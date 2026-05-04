import sys
import random
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton
from PySide6.QtGui import QColor

from W.PySide6.Graphics.FastGraph import (
    FastElement, FastRectLayer, FastCurveLayer, FastPointLayer, FastGraphEditor
)

if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    win: QMainWindow = QMainWindow()
    win.resize(1200, 800)
    
    central: QWidget = QWidget()
    win.setCentralWidget(central)
    
    # Создаем UI: Слева кнопки, справа график
    main_layout: QHBoxLayout = QHBoxLayout(central)
    
    ui_panel: QWidget = QWidget()
    ui_layout: QVBoxLayout = QVBoxLayout(ui_panel)
    main_layout.addWidget(ui_panel, stretch=1)

    plot_widget: pg.PlotWidget = pg.PlotWidget(background='#111')
    main_layout.addWidget(plot_widget, stretch=4)

    # Инициализация слоев
    rect_layer: FastRectLayer = FastRectLayer()
    rect_layer.set_colors("g_rect", QColor(0, 255, 0), QColor(0, 255, 0, 50))
    plot_widget.addItem(rect_layer)

    curve_layer: FastCurveLayer = FastCurveLayer()
    curve_layer.set_colors("g_curve", QColor(0, 200, 255))
    plot_widget.addItem(curve_layer)

    point_layer: FastPointLayer = FastPointLayer(point_radius=5)
    point_layer.set_colors("g_point", QColor(255, 100, 100))
    plot_widget.addItem(point_layer)

    # Инициализация редактора
    editor = FastGraphEditor(plot_widget)

    # Заполнение графиков случайными данными
    for i in range(10):
        # Rects
        x, y = random.uniform(0, 500), random.uniform(0, 400)
        w, h = random.uniform(20, 60), random.uniform(20, 60)
        rect_layer.add_element(FastElement(f"r_{i}", "g_rect", [x, y, w, h]))

        # Points
        px, py = random.uniform(0, 500), random.uniform(0, 400)
        point_layer.add_element(FastElement(f"p_{i}", "g_point", [px, py]))

    # Curves
    for i in range(3): 
        pts: list[list[float]] = []
        start_y = random.uniform(100, 300) 
        for j in range(10):
            cx = float(j * 50)
            cy = start_y + random.uniform(-40, 40)
            pts.append([cx, cy])
        curve_layer.add_element(FastElement(f"c_{i}", "g_curve", pts))

    rect_layer.rebuild()
    curve_layer.rebuild()
    point_layer.rebuild()

    plot_widget.autoRange()

    # --- Создание кнопок ---
    btn_edit_rect = QPushButton("Выделить случайный Rect")
    btn_edit_curve = QPushButton("Выделить случайную Curve")
    btn_edit_point = QPushButton("Выделить случайный Point")
    btn_commit = QPushButton("Сохранить все изменения")

    ui_layout.addWidget(btn_edit_rect)
    ui_layout.addWidget(btn_edit_curve)
    ui_layout.addWidget(btn_edit_point)
    ui_layout.addStretch()
    ui_layout.addWidget(btn_commit)

    # --- Логика выбора случайного элемента и отправки в редактор ---
    def edit_random(layer):
        # Ищем элементы, которые сейчас НЕ редактируются
        available = [eid for eid in layer.elements.keys() if eid not in editor.active_rois]
        if available:
            el_id = random.choice(available)
            editor.edit_element(layer, el_id)

    btn_edit_rect.clicked.connect(lambda: edit_random(rect_layer))
    btn_edit_curve.clicked.connect(lambda: edit_random(curve_layer))
    btn_edit_point.clicked.connect(lambda: edit_random(point_layer))
    
    # Применяем все изменения обратно в FastGraph
    btn_commit.clicked.connect(editor.commit_all)

    win.show()
    sys.exit(app.exec())
