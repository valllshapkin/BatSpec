import sys
import random

import numpy as np
import scipy.ndimage as ndimage
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QButtonGroup,
    QSpinBox, QGroupBox, QSplitter,
)

from MySide6.Image import AdaptiveImageItem
from MySide6.PolyROI import ROIData, ROIGroup, FastROILayer, ROIController, ROIStatusLabel
from MySide6.Tools import ToolManager


class ROIProcessor:
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

        n_freq, n_time = 600, 100_000
        np.random.seed(42)
        data = np.random.randn(n_freq, n_time).astype(np.float32)
        t = np.linspace(0, 4 * np.pi, n_time)
        data += 0.5 * np.outer(np.sin(np.linspace(0, np.pi, n_freq)), np.sin(t))
        data = (data - data.min()) / (data.max() - data.min())
        self.full_data = data

        rois = [
            ROIData(i,
                    random.randint(0, 99950), random.randint(0, 580),
                    random.randint(10, 50),   random.randint(5, 20))
            for i in range(10_000)
        ]
        self.group = ROIGroup("Основной слой", rois, pg.mkColor("g"), pg.mkColor("y"))
        self.processor = ROIProcessor(self.full_data)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.setCentralWidget(splitter)

        self.plot_widget = pg.PlotWidget(background="#111")
        self.plot_item   = self.plot_widget.getPlotItem()
        splitter.addWidget(self.plot_widget)

        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        splitter.addWidget(bottom)
        splitter.setSizes([600, 200])

        tools_group  = QGroupBox("Инструменты")
        tools_layout = QVBoxLayout(tools_group)
        self.btn_select = QPushButton("🖱️ Выделение")
        self.btn_create = QPushButton("➕ Рисовать")
        self.btn_erase  = QPushButton("🗑️ Ластик")
        self.tool_btns  = QButtonGroup(self)
        for i, btn in enumerate([self.btn_select, self.btn_create, self.btn_erase]):
            btn.setCheckable(True)
            self.tool_btns.addButton(btn, i)
            tools_layout.addWidget(btn)
        self.btn_select.setChecked(True)
        tools_layout.addStretch()
        bottom_layout.addWidget(tools_group, stretch=1)

        logic_group  = QGroupBox("Анализ выделенной области")
        logic_layout = QHBoxLayout(logic_group)
        oper_layout  = QVBoxLayout()

        pad_layout = QHBoxLayout()
        pad_layout.addWidget(QLabel("Padding:"))
        self.spin_pad = QSpinBox()
        pad_layout.addWidget(self.spin_pad)

        self.btn_crop   = QPushButton("✂️ Извлечь")
        self.btn_blur   = QPushButton("🌫️ Blur")
        self.btn_laplace = QPushButton("〰️ Laplacian")

        oper_layout.addLayout(pad_layout)
        for btn in [self.btn_crop, self.btn_blur, self.btn_laplace]:
            oper_layout.addWidget(btn)
        logic_layout.addLayout(oper_layout)

        self.preview_view = pg.ImageView()
        self.preview_view.ui.histogram.hide()
        self.preview_view.ui.menuBtn.hide()
        self.preview_view.ui.roiBtn.hide()
        logic_layout.addWidget(self.preview_view, stretch=2)
        bottom_layout.addWidget(logic_group, stretch=4)

        self.img_item = AdaptiveImageItem()
        self.plot_item.addItem(self.img_item)
        self.img_item.attachTo(self.plot_item)
        self.img_item.setFullData(self.full_data, (0, n_time), (0, n_freq))

        self.roi_layer  = FastROILayer(n_time, n_freq)
        self.plot_item.addItem(self.roi_layer)

        self.controller = ROIController(self.plot_item, self.roi_layer)
        self.controller.set_groups([self.group])

        self.status_label = ROIStatusLabel(self.controller)
        oper_layout.insertWidget(0, self.status_label)

        self.tool_manager = ToolManager(
            plot_item  = self.plot_item,
            widget     = self.plot_widget,
            controller = self.controller,
            roi_layer  = self.roi_layer,
            all_groups = [self.group],
        )
        self.tool_manager.active_tool.activate(self.plot_item, self.plot_widget)

        self.plot_item.scene().sigMouseClicked.connect(self.tool_manager.on_click)
        self.tool_btns.idClicked.connect(self.tool_manager.set_tool)
        self.btn_crop.clicked.connect(lambda: self._process_roi("crop"))
        self.btn_blur.clicked.connect(lambda: self._process_roi("blur"))
        self.btn_laplace.clicked.connect(lambda: self._process_roi("laplace"))

        self.plot_item.setXRange(0, 2000, padding=0)
        self.plot_item.setYRange(0, n_freq, padding=0)

    def _process_roi(self, operation: str) -> None:
        if self.controller.active_idx is None:
            self.status_label.setText("Сначала выделите ROI!")
            return
        roi = next(
            (r for r in self.group.rois if r.idx == self.controller.active_idx), None
        )
        if roi is None:
            return
        crop = self.processor.extract_crop(roi, padding=self.spin_pad.value())
        if operation == "blur":
            crop = self.processor.apply_gaussian_blur(crop)
        elif operation == "laplace":
            crop = self.processor.apply_laplacian(crop)
        self.preview_view.setImage(crop.T, autoRange=True, autoLevels=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())