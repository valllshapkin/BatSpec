import sys
import numpy as np
import torch
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QPointF

# --- Твои импорты ---
from BatSpec.QtUp.Reactive.reactive_dict import ReactiveDict
from BatSpec.QtUp.Selector import Selector
from BatSpec.QtUp.Image import AdaptiveImageItem

# Подставьте актуальные пути
from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.Physical.Units import UREG


# =====================================================================
# 1. ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ ДАННЫХ И API
# =====================================================================

plot2d_store: ReactiveDict[str, SpecFunc] = ReactiveDict()

def update_spec2d(name: str, func_obj: SpecFunc) -> None:
    plot2d_store[name] = func_obj

def remove_spec2d(name: str) -> None:
    if name in plot2d_store:
        del plot2d_store[name]

# =====================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ
# =====================================================================

class Syncer(QtCore.QObject):
    """Синхронизирует регион на Minimap и область видимости MainGraph."""
    def __init__(self, minimap_region: pg.LinearRegionItem, graph_view: pg.ViewBox, parent=None):
        super().__init__(parent)
        self._region = minimap_region
        self._view = graph_view
        self._updating = False

        self._region.sigRegionChanged.connect(self._sync_main_from_mini)
        self._view.sigRangeChanged.connect(self._sync_mini_from_main)

    def _sync_main_from_mini(self):
        if self._updating: return
        self._updating = True
        self._view.setXRange(*self._region.getRegion(), padding=0)
        self._updating = False

    def _sync_mini_from_main(self):
        if self._updating: return
        self._updating = True
        self._region.setRegion(self._view.viewRange()[0])
        self._updating = False

class HoverTracker(QtWidgets.QLabel):
    """Отображает координаты и значение спектрограммы под курсором."""
    def __init__(self, view_box: pg.ViewBox, parent=None):
        super().__init__("No data", parent)
        self.setMinimumWidth(300)
        
        self._view_box = view_box
        self._matrix: np.ndarray | None = None   
        self._t_min = self._t_max = 0.0
        self._f_min = self._f_max = 0.0
        self._unit = ""

        scene = self._view_box.scene()
        if scene is not None:
            scene.sigMouseMoved.connect(self._on_mouse_moved)

    # ИЗМЕНЕНИЕ: Аргумент теперь называется matrix_yx для ясности (Y=freq, X=time)
    def set_data(self, matrix_yx: np.ndarray, t0: float, t1: float, f0: float, f1: float, unit: str):
        self._matrix = matrix_yx 
        self._t_min, self._t_max = t0, t1
        self._f_min, self._f_max = f0, f1
        self._unit = unit
        self.setText("Ready")

    def clear_data(self):
        self._matrix = None
        self.setText("No data")

    @QtCore.Slot(QPointF)
    def _on_mouse_moved(self, scene_pos: QPointF):
        if self._matrix is None:
            return

        if not self._view_box.sceneBoundingRect().contains(scene_pos):
            self.setText("Outside graph area")
            return

        view_pos = self._view_box.mapSceneToView(scene_pos)
        t, f = view_pos.x(), view_pos.y()

        if not (self._t_min <= t <= self._t_max and self._f_min <= f <= self._f_max):
            self.setText(f"t={t:.3f} s, f={f:.1f} Hz  [outside bounds]")
            return

        # ИЗМЕНЕНИЕ: Логика индексации теперь соответствует формату (freq, time)
        freq_bins, time_bins = self._matrix.shape
        t_idx = int(np.clip((t - self._t_min) / (self._t_max - self._t_min) * time_bins, 0, time_bins - 1))
        f_idx = int(np.clip((f - self._f_min) / (self._f_max - self._f_min) * freq_bins, 0, freq_bins - 1))
        
        value = float(self._matrix[f_idx, t_idx]) # Правильный порядок: [freq_idx, time_idx]
        self.setText(f"t = {t:.3f} s | f = {f:.1f} Hz | val = {value:.2f} {self._unit}")

# =====================================================================
# 3. ОСНОВНОЙ ВИДЖЕТ ВИЗУАЛИЗАТОРА СПЕКТРОГРАММ
# =====================================================================

class Spec2DVisualizer(QtWidgets.QGroupBox):
    def __init__(self, title: str = "2D Spectrogram Visualizer", parent=None):
        super().__init__(title, parent)
        self._data_loaded = False
        self._is_y_locked = False

        self._setup_ui()
        self._setup_pyqtgraph()
        self._connect_signals()

        self._on_colormap_changed(self.cm_combo.currentText())
        self._redraw_all()

    def _setup_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 8, 4, 4)

        self.controls_layout = QtWidgets.QHBoxLayout()
        
        self.selector = Selector(title="Данные:")
        self.selector.set_dictionary(plot2d_store)
        self.controls_layout.addWidget(self.selector)

        self.controls_layout.addWidget(QtWidgets.QLabel("Цвет:"))
        self.cm_combo = QtWidgets.QComboBox()
        self.cm_combo.addItems(pg.colormap.listMaps())
        if "viridis" in pg.colormap.listMaps():
            self.cm_combo.setCurrentText("viridis")
        self.controls_layout.addWidget(self.cm_combo)

        self.lock_y_cb = QtWidgets.QCheckBox("Lock Y Axis")
        self.controls_layout.addWidget(self.lock_y_cb)
        
        self.controls_layout.addStretch()
        self.main_layout.addLayout(self.controls_layout)

    def _setup_pyqtgraph(self):
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.main_layout.addWidget(self.graphics_layout, stretch=1)

        self.view_box = pg.ViewBox()
        self.spec_plot = self.graphics_layout.addPlot(row=0, col=0, viewBox=self.view_box)
        self.spec_plot.getAxis('left').setWidth(40)
        self.spec_plot.setLabel('bottom', "Время", units="s")
        self.spec_plot.setLabel('left', "Частота", units="Hz")

        self.image_item = AdaptiveImageItem()
        self.image_item.attachTo(self.spec_plot)
        self.spec_plot.addItem(self.image_item)

        self.minimap = self.graphics_layout.addPlot(row=1, col=0)
        self.minimap.setMaximumHeight(60)
        self.minimap.hideAxis('bottom')
        self.minimap.getAxis('left').setWidth(40)
        self.minimap.getAxis('left').setStyle(showValues=False)
        self.minimap.setMouseEnabled(x=False, y=False)

        self.minimap_curve = self.minimap.plot(pen=pg.mkPen('c', width=1), fillLevel=0, brush=(0, 255, 255, 80))
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.minimap.addItem(self.region)

        self.syncer = Syncer(self.region, self.view_box, parent=self)
        
        self.hover_tracker = HoverTracker(self.view_box)
        self.controls_layout.addWidget(self.hover_tracker)

    def _connect_signals(self):
        self.cm_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.lock_y_cb.toggled.connect(self._on_lock_y_toggled)

        self.selector.signals.keySelected.connect(self._redraw_all)
        self.selector.signals.activeValueChanged.connect(self._redraw_all)

    def _redraw_all(self, *args):
        func: SpecFunc | None = self.selector.active_value()
        
        if func is None:
            self._clear_all()
            return

        matrix_tensor, unit_obj = func.values
        f_tensor, _ = func.freq
        t_tensor, _ = func.time

        if matrix_tensor.ndim > 2:
            matrix_tensor = matrix_tensor[0]

        matrix_np = matrix_tensor.detach().cpu().numpy()  # Форма (freq, time)
        f_axis = f_tensor.detach().cpu().numpy()
        t_axis = t_tensor.detach().cpu().numpy()
        unit_str = str(unit_obj)

        t0, t1 = float(t_axis[0]), float(t_axis[-1])
        f0, f1 = float(f_axis[0]), float(f_axis[-1])

        # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ ---
        # ImageItem ожидает данные (Y, X), т.е. (freq, time).
        # Наша матрица matrix_np УЖЕ имеет эту форму, поэтому транспонирование НЕ НУЖНО.
        self.image_item.setFullData(matrix_np, (t0, t1), (f0, f1))

        minimap_data = np.mean(matrix_np, axis=0) # Усреднение по частотам (axis=0)
        self.minimap_curve.setData(x=t_axis, y=minimap_data)

        # Передаем в HoverTracker тоже не-транспонированную матрицу
        self.hover_tracker.set_data(matrix_np, t0, t1, f0, f1, unit_str)
        self._update_camera_limits(t0, t1, f0, f1)


    def _update_camera_limits(self, t0: float, t1: float, f0: float, f1: float):
        self.spec_plot.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self.minimap.setLimits(xMin=None, xMax=None)

        self.minimap.setXRange(t0, t1, padding=0)
        self.region.blockSignals(True)
        self.region.setBounds([t0, t1])
        if not self._data_loaded:
            self.region.setRegion([t0, t1])
        self.region.blockSignals(False)

        if not self._data_loaded:
            self.spec_plot.setXRange(t0, t1, padding=0)
            if self._is_y_locked:
                self.spec_plot.setYRange(f0, f1, padding=0)
            self._data_loaded = True
        else:
            current_x = self.view_box.viewRange()[0]
            if current_x[1] <= t0 or current_x[0] >= t1:
                self.spec_plot.setXRange(t0, t1, padding=0)
                self.region.blockSignals(True)
                self.region.setRegion([t0, t1])
                self.region.blockSignals(False)
            
            if self._is_y_locked:
                self.spec_plot.setYRange(f0, f1, padding=0)

        duration = t1 - t0
        if duration > 1:
            self.spec_plot.setLimits(xMin=t0, xMax=t1, yMin=f0, yMax=f1)
        else:
            self.spec_plot.setLimits(xMin=None, xMax=None, yMin=f0, yMax=f1)
            
        self.minimap.setLimits(xMin=t0, xMax=t1)

    def _clear_all(self):
        self.image_item.clear()
        self.minimap_curve.setData([], [])
        self.spec_plot.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self.minimap.setLimits(xMin=None, xMax=None)
        self._data_loaded = False
        self.hover_tracker.clear_data()

    def _on_colormap_changed(self, cmap_name: str):
        if hasattr(self.image_item, 'setColorMap'):
            self.image_item.setColorMap(cmap_name)
            if hasattr(self.image_item, '_doUpdate'):
                self.image_item._doUpdate()

    def _on_lock_y_toggled(self, checked: bool):
        self._is_y_locked = checked
        self.spec_plot.setMouseEnabled(x=True, y=not checked)
        
        func = self.selector.active_value()
        if checked and func is not None:
            f_tensor, _ = func.freq
            f0 = float(f_tensor[0])
            f1 = float(f_tensor[-1])
            self.spec_plot.setYRange(f0, f1, padding=0)

# =====================================================================
# 4. ДЕМОНСТРАТОР (ОТЛАДКА)
# =====================================================================

def run_standalone_demo():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # --- СНАЧАЛА создаем данные в NumPy ---
    t_np = np.linspace(0, 10, 4000)
    f_np = np.linspace(0, 100, 200)
    
    # T, F meshgrid создает матрицу, где ось 0 - это частота (f_np), а ось 1 - время (t_np)
    # Форма будет (200, 4000) -> (freq, time)
    T, F = np.meshgrid(t_np, f_np) 
    matrix_np = np.sin(T * 2) * np.exp(-(F - 50)**2 / 50) 
    
    # Конвертируем в тензоры PyTorch для SpecFunc
    matrix_t = torch.tensor(matrix_np, dtype=torch.float32)
    time_t = torch.tensor(t_np, dtype=torch.float32)
    freq_t = torch.tensor(f_np, dtype=torch.float32)

    # Инициализируем SpecFunc по новым правилам
    spec_func = SpecFunc(
        matrix=(matrix_t, UREG.V),
        freq=freq_t,
        time=time_t
    )
    
    update_spec2d("Test Spectrogram", spec_func)
    
    # Создаем главное окно
    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle("Тест Визуализатора SpecFunc (PyTorch edition)")
    main_window.resize(1000, 800)
    
    # Добавляем наш виджет
    visualizer = Spec2DVisualizer()
    main_window.setCentralWidget(visualizer)
    main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_standalone_demo()