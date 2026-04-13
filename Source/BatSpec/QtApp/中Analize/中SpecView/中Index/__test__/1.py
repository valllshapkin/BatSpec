import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QLabel

# --- Импорты ваших классов ---
# Замените пути на актуальные, если они отличаются
from BatSpec.QtApp.中Analize.中SpecView.中Index.Logic import INDEX_STATE
from BatSpec.QtApp.中Analize.中SpecView.中Index.Widget import (
    IndexLayer, RoiToolSelector, RoiGroupSelector, 
    RoiStatusLabel, KeyboardToolController
)
from BatSpec.QtUp.PolyROI import ROIGroup, ROIData


def create_mock_roi_groups() -> list[ROIGroup]:
    """Создаем 2 тестовые группы ROI для координат от 0 до 100"""
    # Группа 1
    g1_rois = [
        ROIData(0, 10, 10, 20, 20),
        ROIData(1, 60, 40, 15, 30)
    ]
    # Группа 2
    g2_rois = [
        ROIData(0, 30, 70, 40, 15)
    ]
    
    return [
        ROIGroup("Группа А (Зеленые)", g1_rois, pg.mkColor('g'), pg.mkColor('y')),
        ROIGroup("Группа Б (Красные)", g2_rois, pg.mkColor('r'), pg.mkColor('w'))
    ]


class TestWindow(QMainWindow, INDEX_STATE.Trigger):
    """
    Главное окно для тестирования слоя Index.
    Наследуем Trigger, чтобы управлять блокировкой ViewBox при рисовании.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Тестирование независимого IndexLayer")
        self.resize(800, 600)

        # 1. Создаем базовые элементы PyQtGraph
        self.plot_widget = pg.PlotWidget(background='#111')
        self.setCentralWidget(self.plot_widget)

        self.plot_item = self.plot_widget.getPlotItem()
        self.view_box = self.plot_item.getViewBox()

        # 2. Создаем фейковый фон (чтобы видеть размеры сетки: 100x100)
        np.random.seed(42)
        fake_image_data = np.random.normal(size=(100, 100))
        self.img_item = pg.ImageItem(fake_image_data)
        self.plot_item.addItem(self.img_item)
        self.plot_item.setLimits(xMin=-50, xMax=150, yMin=-50, yMax=150)
        self.plot_item.setXRange(0, 100, padding=0)
        self.plot_item.setYRange(0, 100, padding=0)

        # 3. Настраиваем глобальный Стейт (размер поля и тестовые данные)
        INDEX_STATE.setSize((100, 100))
        INDEX_STATE.setRoiGroups(create_mock_roi_groups())

        # 4. Инициализируем ваш изолированный слой, передав базовые типы
        self.index_layer = IndexLayer(self.plot_item, self.view_box)

        # 5. Инициализируем UI элементы
        self.tool_selector = RoiToolSelector()
        self.group_selector = RoiGroupSelector()
        self.status_label = RoiStatusLabel()

        # Настраиваем ToolBar
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        toolbar.addWidget(QLabel(" Инструмент: "))
        toolbar.addWidget(self.tool_selector)
        toolbar.addSeparator()
        
        toolbar.addWidget(QLabel(" Слой ROI: "))
        toolbar.addWidget(self.group_selector)
        toolbar.addSeparator()
        
        toolbar.addWidget(QLabel(" Подсказка: зажми [Shift] для рисования, [Alt] для стирания"))

        # Настраиваем StatusBar
        self.statusBar().addWidget(self.status_label)

    def onChangeToolMode(self):
        """
        Так как вы закомментировали setMouseEnabled в IndexLayer,
        мы управляем блокировкой графика (чтобы не тянулся фон при рисовании) здесь.
        """
        is_select = (INDEX_STATE.tool_mode == "select")
        self.view_box.setMouseEnabled(x=is_select, y=is_select)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Регистрация перехватчика горячих клавиш (Shift/Alt) на всё приложение
    kbd_controller = KeyboardToolController()
    app.installEventFilter(kbd_controller)

    win = TestWindow()
    win.show()
    
    sys.exit(app.exec())