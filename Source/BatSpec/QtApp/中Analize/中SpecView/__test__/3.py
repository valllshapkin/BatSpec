import sys
import random
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QSplitter

from BatSpec.QtUp.Image import AdaptiveImageItem
from BatSpec.QtUp.PolyROI import ROIData, ROIGroup, FastROILayer, ROIController

# Импорты наших модулей
from BatSpec.QtApp.中Analize.中SpecView.Logic import ROIProcessor
from BatSpec.QtApp.中Analize.中SpecView.__old__.Widget import ToolMode, ROICreatorTool, ToolPanelWidget, ProcessingPanelWidget

class ROIAppController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Студия: Модульная архитектура")
        self.resize(1300, 800)

        self.current_mode = ToolMode.SELECT

        # 1. Подготовка данных и логики
        self._init_data()
        self.processor = ROIProcessor(self.full_data)

        # 2. Построение UI
        self._build_ui()

        # 3. Настройка графики
        self._setup_graphics()

        # 4. Соединение сигналов
        self._connect_signals()

    def _init_data(self):
        n_freq, n_time = 600, 100_000
        np.random.seed(42)
        data = np.random.randn(n_freq, n_time).astype(np.float32)
        t = np.linspace(0, 4 * np.pi, n_time)
        data += 0.5 * np.outer(np.sin(np.linspace(0, np.pi, n_freq)), np.sin(t))
        self.full_data = (data - data.min()) / (data.max() - data.min())

        rois = [
            ROIData(i, random.randint(0, 99950), random.randint(0, 580), 
                    random.randint(10, 50), random.randint(5, 20)) 
            for i in range(10_000)
        ]
        self.group = ROIGroup("Основной слой", rois, pg.mkColor('g'), pg.mkColor('y'))

    def _build_ui(self):
        main_splitter = QSplitter(Qt.Vertical)
        self.setCentralWidget(main_splitter)

        # Область 1: График
        self.plot_widget = pg.PlotWidget(background='#111')
        self.plot_item = self.plot_widget.getPlotItem()
        main_splitter.addWidget(self.plot_widget)

        # Нижняя панель
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        
        # Область 2: Инструменты
        self.tool_panel = ToolPanelWidget()
        bottom_layout.addWidget(self.tool_panel, stretch=1)

        # Область 3: Обработка
        self.process_panel = ProcessingPanelWidget()
        bottom_layout.addWidget(self.process_panel, stretch=3)

        main_splitter.addWidget(bottom_widget)
        main_splitter.setSizes([600, 250])

    def _setup_graphics(self):
        self.img_item = AdaptiveImageItem()
        self.plot_item.addItem(self.img_item)
        self.img_item.attachTo(self.plot_item)
        self.img_item.setFullData(self.full_data, x_range=(0, 100_000), y_range=(0, 600))

        self.roi_layer = FastROILayer(100_000, 600)
        self.plot_item.addItem(self.roi_layer)
        
        self.roi_controller = ROIController(self.plot_item, self.roi_layer)
        self.roi_controller.set_groups([self.group])

        # Добавляем инструмент рисования
        self.creator_tool = ROICreatorTool(self.plot_item, self._on_roi_drawn)
        self.plot_item.addItem(self.creator_tool)

        self.plot_item.setXRange(0, 2000, padding=0)
        self.plot_item.setYRange(0, 600, padding=0)

    def _connect_signals(self):
        # UI -> Контроллер
        self.tool_panel.mode_changed.connect(self._change_mode)
        self.process_panel.action_requested.connect(self._handle_processing)

        # Графика -> Контроллер
        self.plot_item.scene().sigMouseClicked.connect(self._on_scene_clicked)
        
        # Контроллер ROI -> UI
        self.roi_controller.signals.roi_selected.connect(
            lambda name, idx, x, y, w, h: self.process_panel.set_status(f"Выделен ROI #{idx} (w={w:.0f}, h={h:.0f})")
        )
        self.roi_controller.signals.roi_deselected.connect(
            lambda _: self.process_panel.set_status("Нет выделения")
        )

    # ── Логика взаимодействия ─────────────────────────────────────────────

    def _change_mode(self, mode: ToolMode):
        self.current_mode = mode
        
        # Активируем/Деактивируем инструмент рисования
        self.creator_tool.active = (mode == ToolMode.DRAW)
        
        # Разрешаем панорамирование (Drag графика) только в режиме Выделения
        is_panning = (mode == ToolMode.SELECT)
        self.plot_item.getViewBox().setMouseEnabled(x=is_panning, y=is_panning)
        
        # Сброс выделения, если переключились на рисование или ластик
        if mode != ToolMode.SELECT:
            self.roi_controller.deselect()

        # Меняем курсор
        cursors = {
            ToolMode.SELECT: Qt.ArrowCursor,
            ToolMode.DRAW: Qt.CrossCursor,
            ToolMode.ERASE: Qt.ForbiddenCursor
        }
        self.plot_widget.setCursor(cursors[mode])

    def _on_scene_clicked(self, ev):
        """Маршрутизация кликов по графику в зависимости от режима."""
        if self.current_mode == ToolMode.SELECT:
            self.roi_controller.on_click(ev)
            
        elif self.current_mode == ToolMode.ERASE and ev.button() == Qt.LeftButton:
            self._erase_roi_at(ev.scenePos())

    def _on_roi_drawn(self, x: float, y: float, w: float, h: float):
        """Создание нового ROI из инструмента рисования."""
        rois = self.group.rois
        new_idx = max([r.idx for r in rois], default=-1) + 1
        new_roi = ROIData(new_idx, x, y, w, h)
        
        rois.append(new_roi)
        self.roi_layer.set_group(self.group) # Пересобираем слой
        self.roi_controller.signals.rois_changed.emit(self.group.name)

    def _erase_roi_at(self, scene_pos):
        """Удаление ROI под курсором."""
        pos = self.plot_item.vb.mapSceneToView(scene_pos)
        b = int(pos.x()) // self.roi_layer.BUCKET_W
        rois_in_bucket = self.roi_layer._buckets.get(b, [])
        
        hit = next((r for r in reversed(rois_in_bucket) if r.contains(pos.x(), pos.y())), None)
        if hit:
            if self.roi_controller.active_idx == hit.idx:
                self.roi_controller.deselect()
            self.group.rois.remove(hit)
            self.roi_layer.set_group(self.group) # Пересобираем слой
            self.roi_controller.signals.rois_changed.emit(self.group.name)

    def _handle_processing(self, operation: str, padding: int):
        """Запуск бизнес-логики обработки."""
        active_idx = self.roi_controller.active_idx
        if active_idx is None:
            self.process_panel.set_status("Сначала выделите ROI!")
            return
            
        active_roi = next((r for r in self.group.rois if r.idx == active_idx), None)
        if not active_roi:
            return

        # Запрашиваем обработку у Processor'а
        crop_data = self.processor.extract_crop(active_roi, padding=padding)
        
        if operation == "blur":
            crop_data = self.processor.apply_gaussian_blur(crop_data)
        elif operation == "laplace":
            crop_data = self.processor.apply_laplacian(crop_data)
        
        # Отправляем результат обратно в UI
        self.process_panel.show_image(crop_data)
        self.process_panel.set_status(f"Успех: {operation} (Размер: {crop_data.shape})")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ROIAppController()
    win.show()
    sys.exit(app.exec())