import sys
import random

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QComboBox, QHBoxLayout,
)

from BatSpec.QtUp.Image import AdaptiveImageItem
from BatSpec.QtUp.PolyROI import ROIData, ROIGroup, FastROILayer, ROIController, ROIStatusLabel


def main() -> None:
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Тест групп ROI (8 ручек, без вращения)")
    win.resize(1100, 700)

    central = QWidget()
    win.setCentralWidget(central)
    layout = QVBoxLayout(central)

    # ── toolbar ──────────────────────────────
    toolbar = QHBoxLayout()
    combo_groups = QComboBox()
    combo_groups.setFixedWidth(250)
    toolbar.addWidget(QLabel("Активная группа:"))
    toolbar.addWidget(combo_groups)
    layout.addLayout(toolbar)

    # ── plot ─────────────────────────────────
    plot_widget = pg.PlotWidget(background="w")
    plot_item = plot_widget.getPlotItem()
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    layout.addWidget(plot_widget)

    n_freq, n_time = 600, 100_000
    np.random.seed(42)
    data = np.random.randn(n_freq, n_time).astype(np.float32)
    t = np.linspace(0, 4 * np.pi, n_time)
    data += 0.5 * np.outer(np.sin(np.linspace(0, np.pi, n_freq)), np.sin(t))
    data = (data - data.min()) / (data.max() - data.min())

    img_item = AdaptiveImageItem()
    plot_item.addItem(img_item)
    img_item.attachTo(plot_item)
    img_item.setFullData(data, x_range=(0, n_time), y_range=(0, n_freq))

    def generate_rois(count: int) -> list[ROIData]:
        rois = []
        for i in range(count):
            w, h = random.randint(10, 50), random.randint(5, 20)
            x, y = random.randint(0, n_time - w), random.randint(0, n_freq - h)
            rois.append(ROIData(i, x, y, w, h))
        return rois

    groups = [
        ROIGroup("Тип A (Зеленые, 10000 шт)", generate_rois(10000), QColor(80, 200, 120), QColor(0, 180, 0)),
        ROIGroup("Тип B (Красные, 5000 шт)",  generate_rois(5000),  QColor(220, 80, 80),  QColor(255, 0, 0)),
    ]
    for g in groups:
        combo_groups.addItem(g.name)

    roi_layer = FastROILayer(max_x=n_time, max_y=n_freq)
    plot_item.addItem(roi_layer)

    # controller не знает ни о каком лейбле
    controller = ROIController(plot_item, roi_layer)
    controller.set_groups(groups)

    # лейбл сам подписывается на сигналы
    status_label = ROIStatusLabel(controller)
    toolbar.addWidget(status_label, 1)

    plot_item.scene().sigMouseClicked.connect(controller.on_click)
    combo_groups.currentIndexChanged.connect(controller.switch_group)

    plot_item.setXRange(0, 2000, padding=0)
    plot_item.setYRange(0, n_freq, padding=0)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()