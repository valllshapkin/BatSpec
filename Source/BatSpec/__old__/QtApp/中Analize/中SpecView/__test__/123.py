import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox, QLabel
import pyqtgraph as pg

# Глобальная настройка: данные (rows, cols) -> (height, width)
# Это критически важно, чтобы массив 200000x600 отображался правильно (не повернутым)
pg.setConfigOption('imageAxisOrder', 'row-major')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 + PyQtGraph Huge Array Fix")
        self.resize(1200, 800)

        # 1. Генерация массива (200,000 x 600)
        # Используем float32 для экономии памяти (~480 МБ)
        print("Генерация данных...")
        self.data = np.random.normal(size=(200000, 600)).astype(np.float32)
        
        # Добавим структуру для проверки разрешения
        self.data[100000:100050, :] = 5  # Тонкая горизонтальная линия
        self.data[:, 300:305] = 5        # Тонкая вертикальная линия
        print("Данные сгенерированы.")

        # Интерфейс
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Виджет pyqtgraph
        self.graphics_view = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_view)

        self.plot_item = self.graphics_view.addPlot(title="Zoom Test")

        # --- СОЗДАНИЕ ImageItem С ПРАВИЛЬНЫМИ НАСТРОЙКАМИ ---
        
        # Создаем элемент. В конструкторе или через setOpts включаем авто-даунсэмплинг.
        # Это решает проблему "размытия" оси, по которой не приближаем.
        self.image_item = pg.ImageItem(autoDownsample=True)
        
        # Добавляем на график
        self.plot_item.addItem(self.image_item)

        # Устанавливаем данные
        self.image_item.setImage(self.data)

        # Настройка интерполяции (nearest убирает размытие)
        self.image_item.setOpts(interpolation='nearest')

        # Инверсия Y, чтобы ноль был сверху
        self.plot_item.invertY(True)

        # Настройка колормапа
        self.image_item.setColorMap('viridis')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())