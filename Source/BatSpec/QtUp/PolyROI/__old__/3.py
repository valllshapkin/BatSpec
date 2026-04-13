import sys
import logging
import random
import numpy as np
import scipy.ndimage as ndimage
import pyqtgraph as pg

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QCursor, QColor, QPen
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QLabel, QPushButton, QButtonGroup, 
                               QSpinBox, QGroupBox, QSplitter)

# Импорты ваших оптимизированных классов из предыдущих шагов
from BatSpec.QtUp.Image import AdaptiveImageItem
from BatSpec.QtUp.PolyROI import ROIController, ROIData, FastROILayer

# ==============================================================================
# 1. ИНСТРУМЕНТ СОЗДАНИЯ ROI (ROICreatorTool)
# ==============================================================================
class ROICreatorTool(pg.GraphicsObject):
    """
    Невидимый слой, который перехватывает перетаскивание мыши для рисования нового ROI.
    Активируется только когда выбран соответствующий инструмент.
    """
    def __init__(self, plot_item, roi_layer, rois_list):
        super().__init__()
        self.pi = plot_item
        self.layer = roi_layer
        self.rois = rois_list
        self.active = False
        
        # Временный прямоугольник при рисовании
        self.draw_rect = pg.RectROI([0, 0], [0, 0], pen=pg.mkPen('y', width=2, style=Qt.DashLine))
        self.draw_rect.setZValue(100)
        self.pi.addItem(self.draw_rect)
        self.draw_rect.hide()
        
        self.start_pos = None

    def boundingRect(self):
        return self.layer.boundingRect()

    def paint(self, *args):
        pass # Сам слой прозрачный, рисуем только draw_rect

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
            end_pos = pos
            
            # Вычисляем финальные координаты (поддержка рисования в любую сторону)
            x = min(self.start_pos.x(), end_pos.x())
            y = min(self.start_pos.y(), end_pos.y())
            w = abs(self.start_pos.x() - end_pos.x())
            h = abs(self.start_pos.y() - end_pos.y())
            
            # Защита от случайных микро-кликов
            if w > 2 and h > 2:
                new_idx = max([r.idx for r in self.rois], default=-1) + 1
                new_roi = ROIData(new_idx, x, y, w, h)
                self.rois.append(new_roi)
                self.layer._add_to_index(new_roi)
                self.layer.update()
                print(f"Создан новый ROI #{new_idx}")
                
        else: # Процесс перетаскивания
            x = min(self.start_pos.x(), pos.x())
            y = min(self.start_pos.y(), pos.y())
            w = abs(self.start_pos.x() - pos.x())
            h = abs(self.start_pos.y() - pos.y())
            self.draw_rect.setPos([x, y])
            self.draw_rect.setSize([w, h])


# ==============================================================================
# 2. ИНСТРУМЕНТ УДАЛЕНИЯ ROI (ROIEraserTool)
# ==============================================================================
class ROIEraserTool:
    """Удаляет ROI по клику. Взаимодействует с пространственным хэшем FastROILayer."""
    def __init__(self, plot_item, roi_layer, rois_list, controller):
        self.pi = plot_item
        self.layer = roi_layer
        self.rois = rois_list
        self.controller = controller # Чтобы снять выделение, если удаляем активный ROI
        self.active = False

    def on_click(self, ev):
        if not self.active or ev.button() != Qt.LeftButton:
            return
            
        pos = self.pi.vb.mapSceneToView(ev.scenePos())
        px, py = pos.x(), pos.y()

        b = int(px) // self.layer.BUCKET_W
        hit = None
        # Ищем с конца (с верхнего)
        for r in reversed(self.layer._buckets.get(b, [])):
            if r.contains(px, py):
                hit = r
                break

        if hit is not None:
            # Если удаляем выделенный в данный момент ROI
            if self.controller.active_idx == hit.idx:
                self.controller.deselect()
            
            # Удаляем из списка
            self.rois.remove(hit)
            
            # Удаляем из пространственного кэша
            for bucket_idx in range(int(hit.x) // self.layer.BUCKET_W, int(hit.x + hit.w) // self.layer.BUCKET_W + 1):
                if bucket_idx in self.layer._buckets and hit in self.layer._buckets[bucket_idx]:
                    self.layer._buckets[bucket_idx].remove(hit)
            
            self.layer.update()
            print(f"Удален ROI #{hit.idx}")


# ==============================================================================
# 3. ЛОГИКА ОБРАБОТКИ (ROIProcessor)
# ==============================================================================
class ROIProcessor:
    """
    Абстрагированная математика/логика.
    Не изменяет исходный массив `data`, только копирует нужные участки.
    """
    def __init__(self, full_data):
        self.full_data = full_data # Огромный массив 600x100000

    def extract_crop(self, roi: ROIData, padding: int = 0) -> np.ndarray:
        """Вырезает область ROI из огромного массива с учетом паддинга и границ."""
        max_y, max_x = self.full_data.shape
        
        x1 = max(0, int(roi.x) - padding)
        y1 = max(0, int(roi.y) - padding)
        x2 = min(max_x, int(roi.x + roi.w) + padding)
        y2 = min(max_y, int(roi.y + roi.h) + padding)
        
        # Возвращаем копию (!), чтобы случайно не изменить оригинал
        return self.full_data[y1:y2, x1:x2].copy()

    def apply_gaussian_blur(self, crop_data: np.ndarray, sigma=2.0) -> np.ndarray:
        return ndimage.gaussian_filter(crop_data, sigma=sigma)

    def apply_laplacian(self, crop_data: np.ndarray) -> np.ndarray:
        # Лапласиан может давать отрицательные значения, берем модуль
        lap = ndimage.laplace(crop_data)
        return np.abs(lap)


# ==============================================================================
# MAIN И КОМПОНОВКА UI
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Студия: Создание, Удаление, Обработка")
        self.resize(1200, 800)
        
        # ----------------------------------------------------------------------
        # ГЕНЕРАЦИЯ ДАННЫХ
        # ----------------------------------------------------------------------
        print("Генерация данных изображения (600x100000)...")
        self.n_freq, self.n_time = 600, 100_000
        np.random.seed(42)
        base = np.random.randn(self.n_freq, self.n_time).astype(np.float32)
        t = np.linspace(0, 4*np.pi, self.n_time)
        base += 0.5 * np.outer(np.sin(np.linspace(0, np.pi, self.n_freq)), np.sin(t))
        self.full_data = (base - base.min()) / (base.max() - base.min())

        self.rois = []
        for i in range(10_000):
            w, h = random.randint(10, 50), random.randint(5, 20)
            x, y = random.randint(0, self.n_time - w), random.randint(0, self.n_freq - h)
            self.rois.append(ROIData(i, x, y, w, h))

        # ----------------------------------------------------------------------
        # ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА
        # ----------------------------------------------------------------------
        main_splitter = QSplitter(Qt.Vertical)
        self.setCentralWidget(main_splitter)

        # 1. Верхняя панель: Основной график
        pg.setConfigOptions(antialias=False, useOpenGL=False)
        self.plot_widget = pg.PlotWidget(background='#111')
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setLabel("bottom", "Время")
        self.plot_item.setLabel("left", "Частота")
        main_splitter.addWidget(self.plot_widget)

        # Подключаем графику
        self.img_item = AdaptiveImageItem()
        self.plot_item.addItem(self.img_item)
        self.img_item.attachTo(self.plot_item)
        self.img_item.setFullData(self.full_data, x_range=(0, self.n_time), y_range=(0, self.n_freq))

        self.roi_layer = FastROILayer(self.rois, max_x=self.n_time, max_y=self.n_freq)
        self.plot_item.addItem(self.roi_layer)

        # 2. Нижняя панель: Инструменты (Слева) и Логика (Справа)
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(5, 5, 5, 5)
        main_splitter.addWidget(bottom_widget)
        main_splitter.setSizes([600, 200]) # Пропорции сплиттера

        # --- ЛЕВАЯ ПАНЕЛЬ: ИНСТРУМЕНТЫ ---
        tools_group = QGroupBox("Инструменты")
        tools_layout = QVBoxLayout(tools_group)
        
        self.btn_select = QPushButton("🖱️ Выделение")
        self.btn_create = QPushButton("➕ Рисовать ROI")
        self.btn_erase  = QPushButton("🗑️ Ластик")
        
        self.btn_select.setCheckable(True)
        self.btn_create.setCheckable(True)
        self.btn_erase.setCheckable(True)
        self.btn_select.setChecked(True)

        tool_btns = QButtonGroup(self)
        tool_btns.addButton(self.btn_select, 0)
        tool_btns.addButton(self.btn_create, 1)
        tool_btns.addButton(self.btn_erase, 2)
        
        tools_layout.addWidget(self.btn_select)
        tools_layout.addWidget(self.btn_create)
        tools_layout.addWidget(self.btn_erase)
        tools_layout.addStretch()
        bottom_layout.addWidget(tools_group, stretch=1)

        # --- ПРАВАЯ ПАНЕЛЬ: ВНУТРЕННЯЯ ЛОГИКА ROI ---
        logic_group = QGroupBox("Анализ выделенной области")
        logic_layout = QHBoxLayout(logic_group)

        # Кнопки операций
        oper_layout = QVBoxLayout()
        self.status_label = QLabel("Нет выделения")
        self.status_label.setStyleSheet("color: #a00; font-weight: bold;")
        
        # Настройка Padding
        pad_layout = QHBoxLayout()
        pad_layout.addWidget(QLabel("Padding (px):"))
        self.spin_pad = QSpinBox()
        self.spin_pad.setRange(0, 500)
        self.spin_pad.setValue(10)
        pad_layout.addWidget(self.spin_pad)
        
        self.btn_crop = QPushButton("✂️ Извлечь область")
        self.btn_blur = QPushButton("🌫️ Gaussian Blur")
        self.btn_laplace = QPushButton("〰️ Laplacian")
        
        oper_layout.addWidget(self.status_label)
        oper_layout.addLayout(pad_layout)
        oper_layout.addWidget(self.btn_crop)
        oper_layout.addWidget(self.btn_blur)
        oper_layout.addWidget(self.btn_laplace)
        oper_layout.addStretch()
        
        logic_layout.addLayout(oper_layout)

        # Окно предпросмотра вырезанного куска
        self.preview_view = pg.ImageView()
        self.preview_view.ui.histogram.hide() # Прячем лишние элементы UI ImageView
        self.preview_view.ui.menuBtn.hide()
        self.preview_view.ui.roiBtn.hide()
        logic_layout.addWidget(self.preview_view, stretch=2)
        
        bottom_layout.addWidget(logic_group, stretch=4)

        # ----------------------------------------------------------------------
        # СВЯЗЫВАЕМ ЛОГИКУ КЛАССОВ (Glue Code)
        # ----------------------------------------------------------------------
        self.processor = ROIProcessor(self.full_data)
        self.controller = ROIController(self.plot_item, self.roi_layer, self.rois, self.status_label)
        
        self.creator_tool = ROICreatorTool(self.plot_item, self.roi_layer, self.rois)
        self.plot_item.addItem(self.creator_tool) # Добавляем перехватчик мыши на сцену
        
        self.eraser_tool = ROIEraserTool(self.plot_item, self.roi_layer, self.rois, self.controller)

        # Подключаем сигналы мыши PyQtGraph
        self.plot_item.scene().sigMouseClicked.connect(self._on_scene_clicked)
        
        # Переключение инструментов
        tool_btns.buttonClicked.connect(self._on_tool_changed)

        # Кнопки процессора
        self.btn_crop.clicked.connect(lambda: self._process_roi("crop"))
        self.btn_blur.clicked.connect(lambda: self._process_roi("blur"))
        self.btn_laplace.clicked.connect(lambda: self._process_roi("laplace"))

        # Начальный вид
        self.plot_item.setXRange(0, 2000, padding=0)
        self.plot_item.setYRange(0, self.n_freq, padding=0)

    # --------------------------------------------------------------------------
    # СОБЫТИЯ И УПРАВЛЕНИЕ РЕЖИМАМИ
    # --------------------------------------------------------------------------
    def _on_tool_changed(self, btn):
        # Сбрасываем все состояния
        self.creator_tool.active = False
        self.eraser_tool.active = False
        self.plot_item.getViewBox().setMouseEnabled(x=True, y=True) # Включаем зум/пан
        self.plot_widget.setCursor(Qt.ArrowCursor)

        if btn == self.btn_select:
            pass # Обычный режим
            
        elif btn == self.btn_create:
            self.creator_tool.active = True
            self.controller.deselect() # Снимаем выделение при рисовании
            self.plot_widget.setCursor(Qt.CrossCursor)
            # Отключаем перемещение камеры мышью, чтобы можно было рисовать
            self.plot_item.getViewBox().setMouseEnabled(x=False, y=False) 
            
        elif btn == self.btn_erase:
            self.eraser_tool.active = True
            self.plot_widget.setCursor(Qt.ForbiddenCursor) # Имитируем ластик

    def _on_scene_clicked(self, ev):
        if self.btn_select.isChecked():
            self.controller.on_click(ev)
        elif self.btn_erase.isChecked():
            self.eraser_tool.on_click(ev)
        # Рисование (Create) обрабатывается внутри ROICreatorTool через mouseDragEvent

    # --------------------------------------------------------------------------
    # ВЫПОЛНЕНИЕ ОБРАБОТКИ
    # --------------------------------------------------------------------------
    def _process_roi(self, operation):
        if self.controller.active_idx is None:
            self.status_label.setText("Сначала выделите ROI!")
            return
            
        # Получаем данные о текущем выделенном ROI
        # Находим его в массиве (так как active_idx хранит именно .idx, а не позицию в списке)
        active_roi = next((r for r in self.rois if r.idx == self.controller.active_idx), None)
        if not active_roi: return

        # 1. Вырезаем область через Processor
        pad = self.spin_pad.value()
        crop_data = self.processor.extract_crop(active_roi, padding=pad)

        # 2. Применяем фильтр
        if operation == "blur":
            crop_data = self.processor.apply_gaussian_blur(crop_data, sigma=3.0)
        elif operation == "laplace":
            crop_data = self.processor.apply_laplacian(crop_data)
            
        # 3. Отображаем результат в ImageView (ось X и Y транспонируются для ImageView)
        self.preview_view.setImage(crop_data, autoRange=True)
        self.status_label.setText(f"Применено: {operation} (Размер: {crop_data.shape})")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())