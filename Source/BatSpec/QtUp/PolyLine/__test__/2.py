import pyqtgraph as pg
import numpy as np
from BatSpec.Visualize import run_visualizer

from PySide6.QtWidgets import QApplication

app = QApplication()

plot_widget = pg.PlotWidget()

# Генерируем координаты 10 000 точек
x = np.random.normal(size=10000)
y = np.random.normal(size=10000)

# Создаем ОДИН слой для всех точек
scatter = pg.ScatterPlotItem(x=x, y=y, size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))

plot_widget.addItem(scatter)

# Нужно нарисовать 1000 вертикальных отрезков
lines_x = []
lines_y = []

for i in range(1000):
    lines_x.extend([i, i, np.nan]) # x1, x2, разрыв
    lines_y.extend([0, 10, np.nan]) # y1, y2, разрыв

# Создаем ОДНУ кривую
lines_layer = pg.PlotCurveItem(lines_x, lines_y, connect='all', pen='y')
plot_widget.addItem(lines_layer)

plot_widget.show()

app.exec()