import sys
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from typing import Any

# --- Внедряем наш новый универсальный фреймворк ---
import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType

# --- Твои реальные импорты ---
from BatSpec.QtUp.Reactive.reactive_dict import ReactiveDict

from BatSpec.Core.Functions import Function1D, TimeFunc, FreqFunc
from BatSpec.Core.Physical.Units import PintUnit, UREG


# =====================================================================
# 1. ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ ДАННЫХ И API
# =====================================================================

plot_store: ReactiveDict[str, Function1D] = ReactiveDict()

def update_function(name: str, func_obj: Function1D) -> None:
    plot_store[name] = func_obj

def remove_function(name: str) -> None:
    if name in plot_store:
        del plot_store[name]

def clear_all_functions() -> None:
    plot_store.clear()


# =====================================================================
# 2. ВИДЖЕТ ВИЗУАЛИЗАЦИИ
# =====================================================================

class Function1DVisualizer(QtWidgets.QGroupBox):
    def __init__(self, title: str = "1D Function Visualizer", parent: QtWidgets.QWidget | None = None):
        super().__init__(title, parent)

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        # --- ПАНЕЛЬ УПРАВЛЕНИЯ ---
        self.control_layout = QtWidgets.QHBoxLayout()
        self.norm_cb = QtWidgets.QCheckBox("Нормировать Y (Z-Score)")
        self.norm_cb.setChecked(True)
        self.norm_cb.toggled.connect(self._redraw_plot)
        self.control_layout.addWidget(self.norm_cb)
        self.control_layout.addStretch()
        self.main_layout.addLayout(self.control_layout)

        # --- СПИСОК ЧЕКБОКСОВ ---
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setMaximumHeight(120)  
        self.main_layout.addWidget(self.list_widget)
        self.list_widget.itemChanged.connect(self._redraw_plot)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(10, 10)) 
        self.main_layout.addWidget(self.plot_widget)

        self.sync_timer = QtCore.QTimer(self)
        self.sync_timer.timeout.connect(self._sync_store)
        self.sync_timer.start(200)

        self.colors = [
            '#00FF00', '#FF00FF', '#00FFFF', '#FFFF00', 
            '#FF5555', '#5555FF', '#FFAA00', '#FFFFFF'
        ]

    def _sync_store(self):
        current_keys = set(plot_store.keys())
        list_keys = set(self.list_widget.item(i).text() for i in range(self.list_widget.count()))
        
        added = current_keys - list_keys
        removed = list_keys - current_keys
        
        if added or removed:
            self.list_widget.blockSignals(True)
            
            for key in removed:
                items = self.list_widget.findItems(key, QtCore.Qt.MatchExactly)
                for item in items:
                    self.list_widget.takeItem(self.list_widget.row(item))
            
            for key in added:
                item = QtWidgets.QListWidgetItem(key)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked) # По умолчанию включаем
                self.list_widget.addItem(item)
                
            self.list_widget.blockSignals(False)
            
            if removed or added:
                self._redraw_plot()

    def _redraw_plot(self, *args) -> None:
        self.plot_widget.clear()

        checked_items = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                checked_items.append(item.text())

        do_norm = self.norm_cb.isChecked()

        if not checked_items:
            self.plot_widget.setTitle("Выберите графики для отображения")
            self.plot_widget.getPlotItem().setLabel('bottom', 'Ось X')
            self.plot_widget.getPlotItem().setLabel('left', 'Z-Score' if do_norm else 'Значения')
            return

        self.plot_widget.setTitle("")
        
        color_idx = 0
        first_func = None

        for key in checked_items:
            func = plot_store.get(key)
            if func is None:
                continue
            
            if first_func is None:
                first_func = func

            try:
                x_tensor = func._axis__a
                y_tensor = func._value_a

                if y_tensor.ndim > 1:
                    y_tensor = y_tensor[0]
                
                x_data = ma.to_numpy(x_tensor)
                y_data = ma.to_numpy(y_tensor) 
                
                # --- СТАТИСТИЧЕСКАЯ НОРМАЛИЗАЦИЯ ---
                if do_norm:
                    mean_val = np.mean(y_data)
                    std_val = np.std(y_data)
                    # Избегаем деления на ноль для константных графиков
                    if std_val < 1e-12:
                        std_val = 1.0
                    y_data = (y_data - mean_val) / std_val

                color = self.colors[color_idx % len(self.colors)]
                
                curve = self.plot_widget.plot(
                    x=x_data, 
                    y=y_data, 
                    pen=pg.mkPen(color=color, width=1.5),
                    name=key
                )
                curve.setDownsampling(ds=True, auto=True, method='peak')
                curve.setClipToView(True)

                color_idx += 1

            except Exception as e:
                print(f"Ошибка отрисовки графика {key}: {e}")

        if first_func is not None:
            x_unit_str = str(first_func._axis__u)
            if isinstance(first_func, TimeFunc):
                x_label = "Время"
            elif isinstance(first_func, FreqFunc):
                x_label = "Частота"
            else:
                x_label = "Ось X"

            self.plot_widget.getPlotItem().setLabel('bottom', x_label, units=x_unit_str)
            self.plot_widget.getPlotItem().setLabel('left', 'Z-Score (σ)' if do_norm else 'Значения')

# =====================================================================
# 3. ДЕМОНСТРАТОР (ОТЛАДКА)
# =====================================================================

class DemoWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Line Function1D Visualizer")
        self.resize(800, 600)

        self.visualizer = Function1DVisualizer()
        self.setCentralWidget(self.visualizer)
        
        try:
            import torch
            self.ctx = ArrayContext(Framework.TORCH, DeviceType.CPU, None)
        except ImportError:
            self.ctx = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)

        t_np = np.linspace(0, 1, 1000)
        # Сумасшедший разброс порядков:
        y1_np = np.sin(2 * np.pi * 5 * t_np) * 10000.0  
        y2_np = np.cos(2 * np.pi * 5 * t_np) * 0.0001
        
        t_tensor = ma.convert_to(t_np, self.ctx)
        
        func1 = TimeFunc(values=(ma.convert_to(y1_np, self.ctx), UREG.dimensionless), axis=t_tensor)
        func2 = TimeFunc(values=(ma.convert_to(y2_np, self.ctx), UREG.dimensionless), axis=t_tensor)
        
        update_function("Синус (Ампл 10000)", func1)
        update_function("Косинус (Ампл 0.0001)", func2)

def run_standalone_demo() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_standalone_demo()