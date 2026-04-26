import sys

from PySide6.QtWidgets import QApplication, QMainWindow
import numpy as np
import pyqtgraph as pg
from BatSpec.QtUp.Image import AdaptiveImageItem



app = QApplication(sys.argv)

# Создаём главное окно
win = QMainWindow()
win.setWindowTitle("AdaptiveImageItem – 600×100000")
win.resize(900, 600)

# Графический виджет
plot_widget = pg.PlotWidget()
win.setCentralWidget(plot_widget)
plot_item = plot_widget.getPlotItem()

# Создаём адаптивный элемент и добавляем на график
img_item = AdaptiveImageItem()


plot_item.addItem(img_item)
img_item.attachTo(plot_item)

# Генерируем массив 600×100000
n_freq = 300          # строки (например, частоты)
n_time = 37644        # столбцы (например, время)
# Данные: случайный шум + слабая структура для наглядности
np.random.seed(42)
data = np.random.randn(n_freq, n_time).astype(np.float32)
# Добавим синусоидальную модуляцию по времени
t = np.linspace(0, 4*np.pi, n_time)
data += 0.5 * np.outer(np.sin(np.linspace(0, np.pi, n_freq)), np.sin(t))
# Нормализуем в диапазон [0,1] для корректного отображения с уровнями по умолчанию
data = (data - data.min()) / (data.max() - data.min())

# Задаём данные и оси (время от 0 до 10000, частота от 0 до 600)
img_item.setFullData(data, x_range=(0, n_time), y_range=(0, n_freq))

# Можно также установить уровни отображения, если нужно (по умолчанию (0,1))
# img_item.setLevels((0, 1))  # уже установлено в __init__

win.show()
sys.exit(app.exec())

