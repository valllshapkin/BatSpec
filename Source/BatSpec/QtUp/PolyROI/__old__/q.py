import sys
import logging
import random
import numpy as np
import pyqtgraph as pg

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                               QWidget, QLabel, QComboBox, QHBoxLayout)


from MySide6.Image import AdaptiveImageItem
from MySide6.PolyROI import ROIController, ROIData, ROIGroup, FastROILayer

# ------------------------------------------------------------------------------
# Главное окно (UI)
# ------------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = QMainWindow()
    win.setWindowTitle("Множественные группы ROI (8 ручек, без вращения Alt)")
    win.resize(1100, 700)
    
    central = QWidget()
    win.setCentralWidget(central)
    layout = QVBoxLayout(central)
    
    # --- Панель управления ---
    toolbar = QHBoxLayout()
    
    combo_groups = QComboBox()
    combo_groups.setFixedWidth(250)
    toolbar.addWidget(QLabel("Активная группа:"))
    toolbar.addWidget(combo_groups)
    
    status_label = QLabel("Нет выделения")
    status_label.setStyleSheet("color: #333; font-weight: bold; margin-left: 20px;")
    toolbar.addWidget(status_label)
    toolbar.addStretch()
    
    layout.addLayout(toolbar)

    # --- График ---
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    plot_widget = pg.PlotWidget()
    plot_widget.setBackground('w')
    layout.addWidget(plot_widget)
    plot_item = plot_widget.getPlotItem()
    plot_item.showGrid(x=True, y=True, alpha=0.3)

    # 1. Генерируем массив (изображение)
    print("Генерация данных изображения...")
    n_freq, n_time = 600, 100_000
    np.random.seed(42)
    data = np.random.randn(n_freq, n_time).astype(np.float32)
    t = np.linspace(0, 4*np.pi, n_time)
    data += 0.5 * np.outer(np.sin(np.linspace(0, np.pi, n_freq)), np.sin(t))
    data = (data - data.min()) / (data.max() - data.min())

    img_item = AdaptiveImageItem()
    plot_item.addItem(img_item)
    img_item.attachTo(plot_item)
    img_item.setFullData(data, x_range=(0, n_time), y_range=(0, n_freq))

    # 2. Генерируем массивы ROI (Группы)
    print("Генерация массивов ROI...")
    def generate_rois(count):
        rois = []
        for i in range(count):
            w, h = random.randint(10, 50), random.randint(5, 20)
            x, y = random.randint(0, n_time - w), random.randint(0, n_freq - h)
            rois.append(ROIData(i, x, y, w, h))
        return rois

    # Создаем две группы с разными цветами
    group1 = ROIGroup(
        name="Тип A (Зеленые, 10 000 шт)", 
        rois=generate_rois(10_000),
        base_color=QColor(80, 200, 120, 180),   # Бледно-зеленый
        active_color=QColor(0, 180, 0)          # Ярко-зеленый
    )
    
    group2 = ROIGroup(
        name="Тип B (Красные, 5 000 шт)", 
        rois=generate_rois(5_000),
        base_color=QColor(220, 80, 80, 180),    # Бледно-красный
        active_color=QColor(255, 0, 0)          # Ярко-красный
    )
    
    groups = [group1, group2]
    
    for g in groups:
        combo_groups.addItem(g.name)

    # 3. Подключаем слой и контроллер
    roi_layer = FastROILayer(max_x=n_time, max_y=n_freq)
    plot_item.addItem(roi_layer)

    controller = ROIController(plot_item, roi_layer, groups, status_label)
    plot_item.scene().sigMouseClicked.connect(controller.on_click)
    
    # 4. Обработка переключения в ComboBox
    combo_groups.currentIndexChanged.connect(controller.switch_group)
    
    # Инициализация (включаем первую группу по умолчанию)
    controller.switch_group(0)

    plot_item.setXRange(0, 2000, padding=0)
    plot_item.setYRange(0, n_freq, padding=0)

    print("Готово! Переключайте группы сверху.")
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
