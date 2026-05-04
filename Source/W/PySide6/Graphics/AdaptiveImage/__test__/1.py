import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow
from W.PySide6.Graphics.AdaptiveImage import AdaptiveImageItem

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.resize(800, 600)

    plot_widget = pg.PlotWidget()
    win.setCentralWidget(plot_widget)
    plot_item = plot_widget.getPlotItem()

    img_item = AdaptiveImageItem()
    plot_item.addItem(img_item)
    img_item.attachTo(plot_item)

    data = np.random.randn(100, 1000).astype(np.float32)
    img_item.setFullData(data, x_range=(0.0, 10.0), y_range=(0.0, 100.0))

    win.show()
    sys.exit(app.exec())
