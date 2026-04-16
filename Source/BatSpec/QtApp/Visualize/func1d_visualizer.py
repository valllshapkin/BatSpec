import sys
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from typing import Any

# --- Твои реальные импорты ---
from BatSpec.QtUp.Reactive.reactive_dict import ReactiveDict
from BatSpec.QtUp.Selector import Selector
from BatSpec.Work.Function import Function1D 
from BatSpec.Work.Units import PintUnit, UREG


# =====================================================================
# 1. ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ ДАННЫХ И API
# =====================================================================

# Создаем единый реактивный словарь для хранения графиков
plot_store: ReactiveDict[str, Function1D] = ReactiveDict()

def update_function(name: str, func_obj: Function1D) -> None:
    """API для добавления или обновления функции на графике."""
    plot_store[name] = func_obj

def remove_function(name: str) -> None:
    """Удаляет функцию из селектора и хранилища."""
    if name in plot_store:
        del plot_store[name]

def clear_all_functions() -> None:
    """Очищает все графики."""
    plot_store.clear()


# =====================================================================
# 2. ВИДЖЕТ ВИЗУАЛИЗАЦИИ
# =====================================================================

class Function1DVisualizer(QtWidgets.QGroupBox):
    """
    Виджет, который связывает ReactiveDict, Selector и pyqtgraph.
    """
    def __init__(self, title: str = "1D Function Visualizer", parent: QtWidgets.QWidget | None = None):
        super().__init__(title, parent)

        # 1. Настройка Layout
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        # 2. Реактивный Selector
        self.selector = Selector(title="Выбрать график:", parent=self)
        self.selector.set_dictionary(plot_store)
        self.main_layout.addWidget(self.selector)

        # 3. Настройка PyQtGraph
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.main_layout.addWidget(self.plot_widget)

        self.plot_curve = self.plot_widget.plot()
        self.plot_curve.setDownsampling(ds=True, auto=True, method='peak')
        self.plot_curve.setClipToView(True)
        self.plot_curve.setPen(pg.mkPen(color="#00FF00", width=1.5)) 

        # 4. Подписка на сигналы Selector'а
        self.selector.signals.keySelected.connect(self._redraw_plot)
        self.selector.signals.activeValueChanged.connect(self._redraw_plot)

        # Отрисовка начального состояния (если в словаре уже что-то есть)
        self._redraw_plot()

    def _redraw_plot(self, *args) -> None:
        """Метод извлечения данных из Function1D и передачи их в pyqtgraph."""
        func: Function1D | None = self.selector.active_value()
        active_key: str | None = self.selector.active_key()

        if func is None or active_key is None:
            self.plot_curve.setData([], [])
            self.plot_widget.setTitle("Нет данных для отображения")
            self.plot_widget.getPlotItem().setLabel('bottom', 'Ось X')
            self.plot_widget.getPlotItem().setLabel('left', 'Ось Y')
            return

        try:
            # Извлекаем массивы JAX/Numpy
            x_data = np.asarray(func._axis__a)
            y_data = np.asarray(func._value_a)

            # Извлекаем единицы измерения PintUnit в виде строки
            x_unit_str = str(func._axis__u)
            y_unit_str = str(func._value_u)

            # Обновляем график
            self.plot_curve.setData(x=x_data, y=y_data)

            # Обновляем UI
            self.plot_widget.setTitle(f"График: {active_key}")
            self.plot_widget.getPlotItem().setLabel('bottom', 'Ось X', units=x_unit_str)
            self.plot_widget.getPlotItem().setLabel('left', 'Значение', units=y_unit_str)

        except Exception as e:
            self.plot_widget.setTitle(f"Ошибка формата данных: {e}")
            self.plot_curve.setData([], [])


# =====================================================================
# 3. ДЕМОНСТРАТОР (ОТЛАДКА)
# =====================================================================

class DemoWindow(QtWidgets.QMainWindow):
    """Окно для демонстрации работы API и виджета."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Function1D Visualizer")
        self.resize(800, 600)

        # Добавляем виджет
        self.visualizer = Function1DVisualizer()
        self.setCentralWidget(self.visualizer)

        # --- Создаем статический график (Осциллограмма) ---
        t = np.linspace(0, 1, 1000)
        y_time = np.sin(2 * np.pi * 10 * t) # Синусоида 10 Гц
        
        # Сигнатура: _value_a, _value_u, _axis__a, _axis__u
        time_func = Function1D(y_time, UREG.pascal, t, UREG.second)
        update_function("Осциллограмма (Статика)", time_func)

        # --- Подготовка для анимированного графика (Спектр) ---
        self.f = np.linspace(0, 100, 500)
        self.phase = 0.0 # Накопитель фазы для анимации

        # Настраиваем таймер
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_live_data)
        self.timer.start(50) # 20 FPS (каждые 50 мс)

    def update_live_data(self):
        """Вызывается по таймеру, обновляет ReactiveDict новыми данными."""
        self.phase += 0.2  # Двигаем время вперед
        
        # Генерируем "дышащий" пик Гаусса на частоте 10 Гц
        # Ширина пика пульсирует от 0.5 до 3.5
        width = 2.0 + np.sin(self.phase) * 1.5 
        y_freq = np.exp(-((self.f - 10)**2) / width)
        
        # Создаем НОВЫЙ объект Function1D (так требует иммутабельность/реактивность)
        freq_func = Function1D(
            _value_a=y_freq, 
            _value_u=UREG.pascal / UREG.hertz**0.5, # Па/√Гц
            _axis__a=self.f, 
            _axis__u=UREG.hertz
        )
        
        # Обновляем словарь. Selector перехватит `activeValueChanged` и перерисует график!
        update_function("Спектр (Real-Time)", freq_func)


def run_standalone_demo() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DemoWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    run_standalone_demo()