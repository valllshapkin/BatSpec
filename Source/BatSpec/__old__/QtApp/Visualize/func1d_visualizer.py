import sys
import numpy as np
import torch
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from typing import Any

# --- Твои реальные импорты ---
from BatSpec.QtUp.Reactive.reactive_dict import ReactiveDict
from BatSpec.QtUp.Selector import Selector

# Подставьте актуальные пути
from NewSpec.Core.Functions import Function1D
from NewSpec.Core.Units import PintUnit, UREG


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

        self.selector = Selector(title="Выбрать график:", parent=self)
        self.selector.set_dictionary(plot_store)
        self.main_layout.addWidget(self.selector)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.main_layout.addWidget(self.plot_widget)

        self.plot_curve = self.plot_widget.plot()
        self.plot_curve.setDownsampling(ds=True, auto=True, method='peak')
        self.plot_curve.setClipToView(True)
        self.plot_curve.setPen(pg.mkPen(color="#00FF00", width=1.5)) 

        self.selector.signals.keySelected.connect(self._redraw_plot)
        self.selector.signals.activeValueChanged.connect(self._redraw_plot)

        self._redraw_plot()

    def _redraw_plot(self, *args) -> None:
        func: Function1D | None = self.selector.active_value()
        active_key: str | None = self.selector.active_key()

        if func is None or active_key is None:
            self.plot_curve.setData([], [])
            self.plot_widget.setTitle("Нет данных для отображения")
            self.plot_widget.getPlotItem().setLabel('bottom', 'Ось X')
            self.plot_widget.getPlotItem().setLabel('left', 'Ось Y')
            return

        try:
            # 1. Извлекаем тензоры PyTorch и юниты из объекта Function1D
            x_tensor = func._axis__a
            y_tensor = func._value_a
            x_unit_str = str(func._axis__u)
            y_unit_str = str(func._value_u)

            # 2. Обработка батчей: если данные многомерные, берем первый элемент
            title_suffix = ""
            if y_tensor.ndim > 1:
                batch_size = y_tensor.shape[0]
                title_suffix = f" (элемент 0 из {batch_size})"
                y_tensor = y_tensor[0] # Визуализируем только первый элемент батча
            
            # 3. Конвертируем тензоры в NumPy массивы для PyQtGraph
            x_data = x_tensor.detach().cpu().numpy()
            y_data = y_tensor.detach().cpu().numpy()
            
            # Обновляем график
            self.plot_curve.setData(x=x_data, y=y_data)

            # Обновляем UI
            self.plot_widget.setTitle(f"График: {active_key}{title_suffix}")
            self.plot_widget.getPlotItem().setLabel('bottom', 'Ось X', units=x_unit_str)
            self.plot_widget.getPlotItem().setLabel('left', 'Значение', units=y_unit_str)

        except Exception as e:
            self.plot_widget.setTitle(f"Ошибка формата данных: {e}")
            self.plot_curve.setData([], [])


# =====================================================================
# 3. ДЕМОНСТРАТОР (ОТЛАДКА)
# =====================================================================

class DemoWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Function1D Visualizer (PyTorch edition)")
        self.resize(800, 600)

        self.visualizer = Function1DVisualizer()
        self.setCentralWidget(self.visualizer)

        # --- Создаем статический график (Осциллограмма) ---
        t_np = np.linspace(0, 1, 1000)
        y_time_np = np.sin(2 * np.pi * 10 * t_np)
        
        # Конвертируем в тензоры
        t_tensor = torch.tensor(t_np, dtype=torch.float64)
        y_time_tensor = torch.tensor(y_time_np, dtype=torch.float64)

        # Создаем Function1D по новому API: (values=(Tensor, Unit), axis=(Tensor, Unit))
        time_func = Function1D(
            values=(y_time_tensor, UREG.pascal),
            axis=(t_tensor, UREG.second)
        )
        update_function("Осциллограмма (Статика)", time_func)

        # --- Подготовка для анимированного графика (Спектр) ---
        f_np = np.linspace(0, 100, 500)
        self.f_tensor = torch.tensor(f_np, dtype=torch.float64) # Ось частот неизменна
        self.phase = 0.0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_live_data)
        self.timer.start(50)

    def update_live_data(self):
        self.phase += 0.2
        
        width = 2.0 + np.sin(self.phase) * 1.5 
        # Numpy для генерации, т.к. self.f_tensor.numpy() может быть медленно
        y_freq_np = np.exp(-((self.f_tensor.numpy() - 10)**2) / width)
        
        # Конвертируем только изменяющиеся значения в тензор
        y_freq_tensor = torch.tensor(y_freq_np, dtype=torch.float64)
        
        freq_func = Function1D(
            values=(y_freq_tensor, UREG.pascal / UREG.hertz**0.5),
            axis=(self.f_tensor, UREG.hertz)
        )
        
        update_function("Спектр (Real-Time)", freq_func)


def run_standalone_demo() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DemoWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    run_standalone_demo()