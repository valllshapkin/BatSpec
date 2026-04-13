from enum import Enum
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QButtonGroup, QSpinBox, QGroupBox, QLabel
)

class ToolMode(Enum):
    SELECT = 0
    DRAW = 1
    ERASE = 2

class ROICreatorTool(pg.GraphicsObject):
    """Графический инструмент для отрисовки штрихового прямоугольника при создании ROI."""
    def __init__(self, plot_item: pg.PlotItem, on_create_callback) -> None:
        super().__init__()
        self.pi = plot_item
        self.on_create = on_create_callback
        self.active = False
        self.start_pos = None

        self.draw_rect = pg.RectROI([0, 0], [0, 0], pen=pg.mkPen('y', width=2, style=Qt.DashLine))
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


class ToolPanelWidget(QGroupBox):
    """Область инструментов (Переключение режимов)."""
    mode_changed = Signal(ToolMode)

    def __init__(self, parent=None) -> None:
        super().__init__("Режим мыши", parent)
        layout = QVBoxLayout(self)

        self.btn_select = QPushButton("🖱️ Выделение и перемещение")
        self.btn_create = QPushButton("➕ Рисовать (Зажать ЛКМ)")
        self.btn_erase = QPushButton("🗑️ Ластик (Клик ЛКМ)")

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        for mode, btn in zip(ToolMode, [self.btn_select, self.btn_create, self.btn_erase]):
            btn.setCheckable(True)
            self.btn_group.addButton(btn, mode.value)
            layout.addWidget(btn)

        self.btn_select.setChecked(True)
        layout.addStretch()

        self.btn_group.idClicked.connect(lambda id_: self.mode_changed.emit(ToolMode(id_)))


class ProcessingPanelWidget(QGroupBox):
    """Область обработки (Вырезание, Блюр и т.д.)."""
    action_requested = Signal(str, int)  # (название операции, значение padding)

    def __init__(self, parent=None) -> None:
        super().__init__("Анализ выделенной области", parent)
        layout = QHBoxLayout(self)

        # Левая часть (Кнопки и настройки)
        controls_layout = QVBoxLayout()
        self.status_label = QLabel("Нет выделения")
        controls_layout.addWidget(self.status_label)

        pad_layout = QHBoxLayout()
        pad_layout.addWidget(QLabel("Padding (px):"))
        self.spin_pad = QSpinBox()
        self.spin_pad.setRange(0, 500)
        pad_layout.addWidget(self.spin_pad)
        controls_layout.addLayout(pad_layout)

        btn_crop = QPushButton("✂️ Извлечь (Crop)")
        btn_blur = QPushButton("🌫️ Gaussian Blur")
        btn_laplace = QPushButton("〰️ Laplacian")

        btn_crop.clicked.connect(lambda: self.action_requested.emit("crop", self.spin_pad.value()))
        btn_blur.clicked.connect(lambda: self.action_requested.emit("blur", self.spin_pad.value()))
        btn_laplace.clicked.connect(lambda: self.action_requested.emit("laplace", self.spin_pad.value()))

        for btn in [btn_crop, btn_blur, btn_laplace]:
            controls_layout.addWidget(btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout, stretch=1)

        # Правая часть (Превью)
        self.preview_view = pg.ImageView()
        self.preview_view.ui.histogram.hide()
        self.preview_view.ui.menuBtn.hide()
        self.preview_view.ui.roiBtn.hide()
        layout.addWidget(self.preview_view, stretch=2)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def show_image(self, img_data) -> None:
        self.preview_view.setImage(img_data.T, autoRange=True, autoLevels=True)