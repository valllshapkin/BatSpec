import sys
import random
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QColor

from W.PySide6.Graphics.FastGraph import FastElement, FastRectLayer, FastCurveLayer, FastPointLayer

if __name__ == "__main__":
    # Явная типизация переменных графического интерфейса
    app: QApplication = QApplication(sys.argv)
    win: QMainWindow = QMainWindow()
    win.resize(1000, 800)
    
    central: QWidget = QWidget()
    win.setCentralWidget(central)
    layout: QVBoxLayout = QVBoxLayout(central)

    plot_widget: pg.PlotWidget = pg.PlotWidget(background='#111')
    layout.addWidget(plot_widget)

    # ==========================================
    # 1. Тестируем FastRectLayer (Прямоугольники)
    # ==========================================
    rect_layer: FastRectLayer = FastRectLayer()
    rect_layer.set_colors("group_rect", QColor(0, 255, 0), QColor(0, 255, 0, 50))
    plot_widget.addItem(rect_layer)

    for i in range(100):
        x: float = random.uniform(0, 500)
        y: float = random.uniform(0, 100)
        w: float = random.uniform(5, 20)
        h: float = random.uniform(5, 15)
        rect_layer.add_element(FastElement(f"r_{i}", "group_rect", [x, y, w, h]))

    rect_layer.rebuild()

    # ==========================================
    # 2. Тестируем FastCurveLayer (Линии/Кривые)
    # ==========================================
    curve_layer: FastCurveLayer = FastCurveLayer()
    curve_layer.set_colors("group_curve", QColor(0, 200, 255))
    plot_widget.addItem(curve_layer)

    for i in range(5): 
        pts: list[list[float]] = []
        start_y: float = random.uniform(150, 250) 
        for j in range(25):
            cx: float = float(j * 20)
            cy: float = start_y + random.uniform(-30, 30)
            pts.append([cx, cy])
        curve_layer.add_element(FastElement(f"c_{i}", "group_curve", pts))

    curve_layer.rebuild()

    # ==========================================
    # 3. Тестируем FastPointLayer (Точки)
    # ==========================================
    point_layer: FastPointLayer = FastPointLayer(point_radius=5)
    point_layer.set_colors("group_point", QColor(255, 100, 100))
    plot_widget.addItem(point_layer)

    for i in range(200):
        px: float = random.uniform(0, 500)
        py: float = random.uniform(300, 400)
        point_layer.add_element(FastElement(f"p_{i}", "group_point", [px, py]))

    point_layer.rebuild()

    # ==========================================
    # Принудительно вызываем автомасштабирование
    # ==========================================
    plot_widget.autoRange()

    win.show()
    sys.exit(app.exec())
