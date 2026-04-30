import sys
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QPointF

# --- Внедряем наш новый универсальный фреймворк ---
import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType

# --- Твои импорты ---
from BatSpec.QtUp.Reactive.reactive_dict import ReactiveDict
from BatSpec.QtUp.Selector import Selector
from BatSpec.QtUp.Image import AdaptiveImageItem

from BatSpec.Core.Functions import Function2D, SpecFunc, CorrelFunc
from BatSpec.Core.Physical.Units import UREG


# =====================================================================
# 1. ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ ДАННЫХ И API
# =====================================================================

plot2d_store: ReactiveDict[str, Function2D] = ReactiveDict()

def update_spec2d(name: str, func_obj: Function2D) -> None:
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
    """Отображает координаты и значение функции 2D под курсором."""
    def __init__(self, view_box: pg.ViewBox, parent=None):
        super().__init__("No data", parent)
        self.setMinimumWidth(350)
        
        self._view_box = view_box
        self._matrix: np.ndarray | None = None   
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 0.0
        self._unit = ""
        self._x_label = "X"
        self._y_label = "Y"

        scene = self._view_box.scene()
        if scene is not None:
            scene.sigMouseMoved.connect(self._on_mouse_moved)

    def set_data(self, matrix_yx: np.ndarray, x0: float, x1: float, y0: float, y1: float, 
                 unit: str, x_label: str, y_label: str):
        self._matrix = matrix_yx 
        self._x_min, self._x_max = x0, x1
        self._y_min, self._y_max = y0, y1
        self._unit = unit
        self._x_label = x_label
        self._y_label = y_label
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
        x, y = view_pos.x(), view_pos.y()

        if not (self._x_min <= x <= self._x_max and self._y_min <= y <= self._y_max):
            self.setText(f"{self._x_label}={x:.3f}, {self._y_label}={y:.3f} [outside bounds]")
            return

        y_bins, x_bins = self._matrix.shape
        x_idx = int(np.clip((x - self._x_min) / (self._x_max - self._x_min) * x_bins, 0, x_bins - 1))
        y_idx = int(np.clip((y - self._y_min) / (self._y_max - self._y_min) * y_bins, 0, y_bins - 1))
        
        value = float(self._matrix[y_idx, x_idx])
        self.setText(f"{self._x_label} = {x:.3f} | {self._y_label} = {y:.3f} | val = {value:.2f} {self._unit}")

# =====================================================================
# 3. ОСНОВНОЙ ВИДЖЕТ ВИЗУАЛИЗАТОРА 2D
# =====================================================================

class Spec2DVisualizer(QtWidgets.QGroupBox):
    def __init__(self, title: str = "2D Function Visualizer", parent=None):
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
        self.spec_plot.getAxis('left').setWidth(50)
        self.spec_plot.setLabel('bottom', "Ось X")
        self.spec_plot.setLabel('left', "Ось Y")

        self.image_item = AdaptiveImageItem()
        self.image_item.attachTo(self.spec_plot)
        self.spec_plot.addItem(self.image_item)

        self.minimap = self.graphics_layout.addPlot(row=1, col=0)
        self.minimap.setMaximumHeight(60)
        self.minimap.hideAxis('bottom')
        self.minimap.getAxis('left').setWidth(50)
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
        func: Function2D | None = self.selector.active_value()
        
        if func is None:
            self._clear_all()
            return
            
        # 1. Извлекаем сырые данные
        matrix_t, unit_obj = func._matrx_a, func._matrx_u
        y_axis_t, y_unit = func._first_a, func._first_u
        x_axis_t, x_unit = func._sec___a, func._sec___u

        if matrix_t.ndim > 2:
            matrix_t = matrix_t[0]

        # Определяем лейблы на основе типа данных
        if isinstance(func, SpecFunc):
            x_label, y_label = "Время", "Частота"
        elif isinstance(func, CorrelFunc):
            x_label, y_label = "Время", "Задержка"
        else:
            x_label, y_label = "Ось X", "Ось Y"

        self.spec_plot.setLabel('bottom', x_label, units=str(x_unit))
        self.spec_plot.setLabel('left', y_label, units=str(y_unit))

        # 2. Универсальная конвертация в NumPy для PyQtGraph
        matrix_np = ma.to_numpy(matrix_t)
        
        # СУПЕР ВАЖНО ДЛЯ FPS: Гарантируем, что память не фрагментирована (C-Contiguous)
        matrix_np = np.ascontiguousarray(matrix_np)
        
        x_axis = ma.to_numpy(x_axis_t)
        y_axis = ma.to_numpy(y_axis_t)

        unit_str = str(unit_obj)

        x0, x1 = float(x_axis[0]), float(x_axis[-1])
        y0, y1 = float(y_axis[0]), float(y_axis[-1])

        # ImageItem ожидает данные (Y, X). Наша матрица уже имеет эту форму.
        self.image_item.setFullData(matrix_np, (x0, x1), (y0, y1))

        minimap_data = np.mean(matrix_np, axis=0)
        self.minimap_curve.setData(x=x_axis, y=minimap_data)

        self.hover_tracker.set_data(matrix_np, x0, x1, y0, y1, unit_str, x_label, y_label)
        self._update_camera_limits(x0, x1, y0, y1)


    def _update_camera_limits(self, x0: float, x1: float, y0: float, y1: float):
        self.spec_plot.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self.minimap.setLimits(xMin=None, xMax=None)

        self.minimap.setXRange(x0, x1, padding=0)
        self.region.blockSignals(True)
        self.region.setBounds([x0, x1])
        if not self._data_loaded:
            self.region.setRegion([x0, x1])
        self.region.blockSignals(False)

        if not self._data_loaded:
            self.spec_plot.setXRange(x0, x1, padding=0)
            if self._is_y_locked:
                self.spec_plot.setYRange(y0, y1, padding=0)
            self._data_loaded = True
        else:
            current_x = self.view_box.viewRange()[0]
            if current_x[1] <= x0 or current_x[0] >= x1:
                self.spec_plot.setXRange(x0, x1, padding=0)
                self.region.blockSignals(True)
                self.region.setRegion([x0, x1])
                self.region.blockSignals(False)
            
            if self._is_y_locked:
                self.spec_plot.setYRange(y0, y1, padding=0)

        duration = x1 - x0
        # Если шкала X достаточно большая, фиксируем лимиты камеры
        if duration > 1e-3:
            self.spec_plot.setLimits(xMin=x0, xMax=x1, yMin=y0, yMax=y1)
        else:
            self.spec_plot.setLimits(xMin=None, xMax=None, yMin=y0, yMax=y1)
            
        self.minimap.setLimits(xMin=x0, xMax=x1)

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
            f_tensor = func._first_a
            f0 = float(f_tensor[0])
            f1 = float(f_tensor[-1])
            self.spec_plot.setYRange(f0, f1, padding=0)

# =====================================================================
# 4. ДЕМОНСТРАТОР (ОТЛАДКА)
# =====================================================================

def run_standalone_demo():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    try:
        import torch
        ctx = ArrayContext(Framework.TORCH, DeviceType.CPU, None)
    except ImportError:
        ctx = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)

    # --- СНАЧАЛА создаем данные в NumPy ---
    t_np = np.linspace(0, 10, 4000)
    f_np = np.linspace(0, 100, 200)
    
    T, F = np.meshgrid(t_np, f_np) 
    matrix_np = np.sin(T * 2) * np.exp(-(F - 50)**2 / 50) 
    
    matrix_t = ma.convert_to(matrix_np, ctx)
    time_t = ma.convert_to(t_np, ctx)
    freq_t = ma.convert_to(f_np, ctx)

    spec_func = SpecFunc(
        matrix=(matrix_t, UREG.V),
        freq=freq_t,
        time=time_t
    )
    
    fw_name = ctx.isTorch() and 'Torch' or 'NumPy'
    update_spec2d(f"Test Spectrogram ({fw_name})", spec_func)
    
    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle("Тест Визуализатора 2D (MultiArray edition)")
    main_window.resize(1000, 800)
    
    visualizer = Spec2DVisualizer()
    main_window.setCentralWidget(visualizer)
    main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_standalone_demo()
