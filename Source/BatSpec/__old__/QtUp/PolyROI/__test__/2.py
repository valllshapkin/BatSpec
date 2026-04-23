import sys
import random
import numpy as np
import scipy.ndimage as ndimage
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QButtonGroup,
    QSpinBox, QGroupBox, QSplitter
)

# Импорты из вашей структуры проекта
from BatSpec.QtUp.Image import AdaptiveImageItem
from BatSpec.QtUp.PolyROI import ROIData, ROIGroup, FastROILayer, ROIController


class ROICreatorTool(pg.GraphicsObject):
    """Инструмент для рисования новых ROI на графике."""
    def __init__(self, plot_item: pg.PlotItem, on_create_callback) -> None:
        super().__init__()
        self.pi = plot_item
        self.on_create = on_create_callback
        self.active = False
        self.start_pos = None

        self.draw_rect = pg.RectROI(
            [0, 0], [0, 0], 
            pen=pg.mkPen('y', width=2, style=Qt.DashLine)
        )
        self.pi.addItem(self.draw_rect)
        self.draw_rect.hide()

    def boundingRect(self):
        return self.pi.vb.viewRect()

    def paint(self, *args):
        pass

    def mouseDragEvent(self, ev):
        if not self.active or ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        
        ev.accept()
        pos = self.pi.vb.mapSceneToView(ev.scenePos())
        
        if ev.isStart():
            self.start_pos = pos
            self.draw_rect.setPos(pos)
            self.draw_rect.setSize([0, 0])
            self.draw_rect.show()
            
        elif ev.isFinish():
            self.draw_rect.hide()
            x = min(self.start_pos.x(), pos.x())
            y = min(self.start_pos.y(), pos.y())
            w = abs(self.start_pos.x() - pos.x())
            h = abs(self.start_pos.y() - pos.y())
            
            if w > 2 and h > 2:
                self.on_create(x, y, w, h)
                
        else:
            x = min(self.start_pos.x(), pos.x())
            y = min(self.start_pos.y(), pos.y())
            w = abs(self.start_pos.x() - pos.x())
            h = abs(self.start_pos.y() - pos.y())
            self.draw_rect.setPos([x, y])
            self.draw_rect.setSize([w, h])


class ROIProcessor:
    """Класс для обработки выделенных областей."""
    def __init__(self, full_data: np.ndarray) -> None:
        self.full_data = full_data

    def extract_crop(self, roi: ROIData, padding: int = 0) -> np.ndarray:
        max_y, max_x = self.full_data.shape
        x1 = max(0, int(roi.x) - padding)
        y1 = max(0, int(roi.y) - padding)
        x2 = min(max_x, int(roi.x + roi.w) + padding)
        y2 = min(max_y, int(roi.y + roi.h) + padding)
        return self.full_data[y1:y2, x1:x2].copy()

    def apply_gaussian_blur(self, crop: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        return ndimage.gaussian_filter(crop, sigma=sigma)

    def apply_laplacian(self, crop: np.ndarray) -> np.ndarray:
        return np.abs(ndimage.laplace(crop))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ROI Студия: Инструменты и Обработка")
        self.resize(1200, 800)

        # ── Генерация тестовых данных ──
        n_freq, n_time = 600, 100_000
        np.random.seed(42)
        data = np.random.randn(n_freq, n_time).astype(np.float32)
        t = np.linspace(0, 4 * np.pi, n_time)
        data += 0.5 * np.outer(np.sin(np.linspace(0, np.pi, n_freq)), np.sin(t))
        self.full_data = (data - data.min()) / (data.max() - data.min())

        # ── Генерация ROI ──
        rois = [
            ROIData(
                i, 
                random.randint(0, 99950), 
                random.randint(0, 580), 
                random.randint(10, 50), 
                random.randint(5, 20)
            ) for i in range(10_000)
        ]
        self.group = ROIGroup("Основной слой", rois, pg.mkColor('g'), pg.mkColor('y'))
        
        # ── UI: Основной Splitter ──
        main_splitter = QSplitter(Qt.Vertical)
        self.setCentralWidget(main_splitter)
        
        # Верхняя часть: График
        self.plot_widget = pg.PlotWidget(background='#111')
        self.plot_item = self.plot_widget.getPlotItem()
        main_splitter.addWidget(self.plot_widget)

        # Нижняя часть: Панель инструментов
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        main_splitter.addWidget(bottom_widget)
        main_splitter.setSizes([600, 200])

        # ── Инструменты мыши ──
        tools_group = QGroupBox("Инструменты")
        tools_layout = QVBoxLayout(tools_group)
        
        self.btn_select = QPushButton("🖱️ Выделение")
        self.btn_create = QPushButton("➕ Рисовать")
        self.btn_erase = QPushButton("🗑️ Ластик")
        
        self.tool_btns = QButtonGroup(self)
        for i, btn in enumerate([self.btn_select, self.btn_create, self.btn_erase]):
            btn.setCheckable(True)
            self.tool_btns.addButton(btn, i)
            tools_layout.addWidget(btn)
            
        self.btn_select.setChecked(True)
        tools_layout.addStretch()
        bottom_layout.addWidget(tools_group, stretch=1)

        # ── Анализ и Обработка ──
        logic_group = QGroupBox("Анализ выделенной области")
        logic_layout = QHBoxLayout(logic_group)
        
        oper_layout = QVBoxLayout()
        self.status_label = QLabel("Нет выделения")
        oper_layout.addWidget(self.status_label)
        
        pad_layout = QHBoxLayout()
        pad_layout.addWidget(QLabel("Padding:"))
        self.spin_pad = QSpinBox()
        self.spin_pad.setValue(0)
        pad_layout.addWidget(self.spin_pad)
        oper_layout.addLayout(pad_layout)
        
        self.btn_crop = QPushButton("✂️ Извлечь")
        self.btn_blur = QPushButton("🌫️ Blur")
        self.btn_laplace = QPushButton("〰️ Laplacian")
        
        for btn in [self.btn_crop, self.btn_blur, self.btn_laplace]:
            oper_layout.addWidget(btn)
            
        logic_layout.addLayout(oper_layout)
        
        # Превью результата
        self.preview_view = pg.ImageView()
        self.preview_view.ui.histogram.hide()
        self.preview_view.ui.menuBtn.hide()
        self.preview_view.ui.roiBtn.hide()
        logic_layout.addWidget(self.preview_view, stretch=2)
        
        bottom_layout.addWidget(logic_group, stretch=4)
        
        # ── Инициализация графики ──
        self.img_item = AdaptiveImageItem()
        self.plot_item.addItem(self.img_item)
        self.img_item.attachTo(self.plot_item)
        self.img_item.setFullData(self.full_data, x_range=(0, 100_000), y_range=(0, 600))

        self.roi_layer = FastROILayer(100_000, 600)
        self.plot_item.addItem(self.roi_layer)
        
        # ИСПРАВЛЕНИЕ: Передаем только 2 аргумента
        self.controller = ROIController(self.plot_item, self.roi_layer)
        self.controller.set_groups([self.group])

        # Подключаем кастомный статус-лейбл к сигналам контроллера
        self.controller.signals.roi_selected.connect(
            lambda name, idx, x, y, w, h: self.status_label.setText(f"Выделен ROI #{idx} (w={w:.0f}, h={h:.0f})")
        )
        self.controller.signals.roi_deselected.connect(
            lambda name: self.status_label.setText("Нет выделения")
        )
        
        self.creator_tool = ROICreatorTool(self.plot_item, self._on_roi_created)
        self.plot_item.addItem(self.creator_tool)
        
        self.processor = ROIProcessor(self.full_data)
        
        # ── Сигналы слотов ──
        self.plot_item.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self.tool_btns.idClicked.connect(self._on_tool_changed)
        
        self.btn_crop.clicked.connect(lambda: self._process_roi("crop"))
        self.btn_blur.clicked.connect(lambda: self._process_roi("blur"))
        self.btn_laplace.clicked.connect(lambda: self._process_roi("laplace"))
        
        self.plot_item.setXRange(0, 2000, padding=0)
        self.plot_item.setYRange(0, 600, padding=0)

    def _on_tool_changed(self, tool_id: int) -> None:
        self.creator_tool.active = (tool_id == 1)
        is_panning = (tool_id == 0)
        self.plot_item.getViewBox().setMouseEnabled(x=is_panning, y=is_panning)
        
        cursors = [Qt.ArrowCursor, Qt.CrossCursor, Qt.ForbiddenCursor]
        self.plot_widget.setCursor(cursors[tool_id])
        
        if tool_id != 0:
            self.controller.deselect()

    def _on_scene_clicked(self, ev) -> None:
        tool_id = self.tool_btns.checkedId()
        if tool_id == 0:
            self.controller.on_click(ev)
        elif tool_id == 2 and ev.button() == Qt.LeftButton:
            self._erase_roi_at(ev.scenePos())

    def _on_roi_created(self, x: float, y: float, w: float, h: float) -> None:
        rois = self.group.rois
        new_idx = max([r.idx for r in rois], default=-1) + 1
        new_roi = ROIData(new_idx, x, y, w, h)
        rois.append(new_roi)
        
        # ИСПРАВЛЕНИЕ: Безопасное обновление слоя со всеми Numpy массивами
        self.roi_layer.set_group(self.group)
        self.controller.signals.rois_changed.emit(self.group.name)

    def _erase_roi_at(self, scene_pos) -> None:
        pos = self.plot_item.vb.mapSceneToView(scene_pos)
        b = int(pos.x()) // self.roi_layer.BUCKET_W
        rois_in_bucket = self.roi_layer._buckets.get(b, [])
        
        hit = next((r for r in reversed(rois_in_bucket) if r.contains(pos.x(), pos.y())), None)
        if hit:
            if self.controller.active_idx == hit.idx:
                self.controller.deselect()
            self.group.rois.remove(hit)
            
            # ИСПРАВЛЕНИЕ: Безопасное обновление слоя 
            self.roi_layer.set_group(self.group)
            self.controller.signals.rois_changed.emit(self.group.name)

    def _process_roi(self, operation: str) -> None:
        if self.controller.active_idx is None:
            self.status_label.setText("Сначала выделите ROI!")
            return
            
        active_roi = next((r for r in self.group.rois if r.idx == self.controller.active_idx), None)
        if not active_roi:
            return

        crop_data = self.processor.extract_crop(active_roi, padding=self.spin_pad.value())
        
        if operation == "blur":
            crop_data = self.processor.apply_gaussian_blur(crop_data)
        elif operation == "laplace":
            crop_data = self.processor.apply_laplacian(crop_data)
        
        self.preview_view.setImage(crop_data.T, autoRange=True, autoLevels=True)
        self.status_label.setText(f"Применено: {operation} (Размер: {crop_data.shape})")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())