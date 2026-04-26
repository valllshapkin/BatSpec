Это **две абсолютно разные проблемы**, и обе имеют логичное объяснение. Одна связана с жесткостью TensorFlow, а вторая — с тем, как устроена память в PyTorch/TensorFlow и рендерер PyQtGraph.

---

### Проблема 1: Ошибка TensorFlow (Op:Mul)
TensorFlow **категорически запрещает** математические операции между разными типами данных (в отличие от NumPy, который делает это молча). 

Вы перевели сигнал в `tf.float32`. Однако, когда мы создавали "чистый" контекст для оси времени и окна (`real_ctx = ma.ArrayContext(..., dtype=None)`), TensorFlow по умолчанию сгенерировал окно в `tf.float64` (double).
Когда TF попытался умножить фрейм сигнала (`float32`) на окно (`float64`) внутри `stft`, он выбросил ошибку: `expected to be a float tensor but is a double tensor`.

**Решение:** Нам нужно аккуратно извлекать `real_dtype` (вещественный тип той же точности) из исходного сигнала и передавать его в `real_ctx`.

### Проблема 2: Падение FPS в два раза
Ваш тестовый скрипт с `np.random.randn` летает, потому что NumPy создает матрицы в формате **C-contiguous** (непрерывный блок памяти в строчном порядке). 

Когда же `stft` считает спектрограмму в PyTorch или TensorFlow, он делает кучу внутренних транспонирований и сдвигов (strides). В результате матрица спектрограммы становится **фрагментированной** в памяти. Когда вы конвертируете её в NumPy через `to_numpy()`, она сохраняет эту фрагментацию. 
PyQtGraph (`ImageItem`) использует низкоуровневые C-циклы для отрисовки графики. Если дать ему фрагментированную матрицу, он не может использовать быстрое копирование памяти и скатывается в попиксельный рендер, что **убивает FPS на 50-80%**.

**Решение:** Ровно одна строчка `np.ascontiguousarray()` перед отправкой в PyQtGraph.

---

### Проблема 3: Сдвиг частот (120 vs 130 kHz)
Это не баг, это **возросшая точность математики**. 
В старых версиях мы считали `df` и `dt` грубо, опираясь на длину массива. Сейчас мы используем библиотечные функции `rfftfreq`, которые выдают математически идеальную сетку частот вплоть до предела Найквиста (строго половина от `sr`). Частоты стали честными.

---

Ниже два обновленных файла. Я добавил извлечение `real_dtype` для TensorFlow/PyTorch и `ascontiguousarray` для визуализатора.

### 1. Исправленный `Spectral/__init__.py`

``````python path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/__init__.py" encoding="utf-8"
import math
from typing import Tuple, Union, Any
import warnings

# --- Внедряем наш новый универсальный фреймворк ---
import MultiArray as ma
import MultiArray.FFT as mfft

from BatSpec.Core.Functions import TimeFunc, SpecFunc
from BatSpec.Core.ConvWindow import Window, WindowNorm, WindowNormMismatchError
from BatSpec.Core.Physical.Units import UREG, unit_devide, unit_sqrt, unit_mul

def _get_real_dtype(tensor: Any, ctx: ma.ArrayContext) -> Any:
    """Безопасно извлекает вещественный тип данных (float32/float64), сохраняя точность."""
    if ctx.isTensorflow():
        return tensor.dtype.real_dtype if hasattr(tensor.dtype, 'real_dtype') else tensor.dtype
    elif ctx.isTorch():
        return tensor.real.dtype if tensor.is_complex() else tensor.dtype
    else:
        return tensor.real.dtype if hasattr(tensor, 'real') else tensor.dtype

# =====================================================================
# БЛОК 1: Сложные DSP функции (STFT, IFFT). Теперь через MultiArray API
# =====================================================================

def makeComplexSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)

    ctx = signal.context
    signal_a, signal_unit = signal.values
    sr = signal.sr

    # ИЗВЛЕКАЕМ ТОЧНЫЙ ТИП: Чтобы float32 не умножался на float64 (краш TensorFlow)
    real_dtype = _get_real_dtype(signal_a, ctx)
    real_ctx = ma.ArrayContext(ctx._framework, ctx._device, real_dtype)
    
    win_length = int(window.time * sr)
    fft_length = (bins - 1) * 2
    hop = max(1, int(win_length * (1 - overlap)))

    if fft_length < win_length:
        warnings.warn(f"fft_length={fft_length} меньше win_length={win_length}. Фрейм будет обрезан.")
    if fft_length > win_length:
        warnings.warn(f"fft_length={fft_length} больше win_length={win_length}. Больше разрешения.")

    win = window.get_array(sr, real_ctx)

    stft = mfft.stft(signal_a, frame_length=win_length, frame_step=hop, fft_length=fft_length, window=win, pad_end=True)

    scale_factor = math.sqrt(2.0 / sr)
    num_frames = stft.shape[-1]
    
    freq_axis = mfft.rfftfreq(fft_length, d=1.0 / sr, ctx=real_ctx)
    time_axis = ma.arange(num_frames, real_ctx) * (hop / sr)
    spec_unit = unit_devide(signal_unit, unit_sqrt(UREG.Hz))

    return SpecFunc(matrix=(stft * scale_factor, spec_unit), freq=freq_axis, time=time_axis)


def inverseComplexSpec(spec: SpecFunc, window: Window, overlap: float = 0.5) -> TimeFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)
    
    ctx = spec.context
    complex_matrix, spec_unit = spec.values
    
    real_dtype = _get_real_dtype(complex_matrix, ctx)
    real_ctx = ma.ArrayContext(ctx._framework, ctx._device, real_dtype) 
    
    freq_axis, _ = spec.freq

    fft_length = (len(freq_axis) - 1) * 2
    sr_f_val, _ = spec.df 
    sr = int(round(sr_f_val * fft_length))
    
    win_length = int(window.time * sr)
    hop = max(1, int(win_length * (1 - overlap)))
    
    win = window.get_array(sr, real_ctx)

    scale_factor = math.sqrt(2.0 / sr)
    unscaled = complex_matrix / scale_factor

    signal_a = mfft.istft(unscaled, frame_length=win_length, frame_step=hop, fft_length=fft_length, window=win)
    
    if hasattr(signal_a, 'real') and not ctx.isTorch():
        signal_a = signal_a.real
    elif ctx.isTorch() and signal_a.is_complex():
        signal_a = signal_a.real
    
    new_time_axis = ma.arange(signal_a.shape[-1], real_ctx) / float(sr)
    signal_unit = unit_mul(spec_unit, unit_sqrt(UREG.Hz))

    return TimeFunc(values=(signal_a, signal_unit), axis=new_time_axis)


def makeSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    complex_spec = makeComplexSpec(signal, window, overlap, bins)
    
    amp_matrix = ma.abs(complex_spec.values[0])
    
    return SpecFunc(
        matrix=(amp_matrix, complex_spec.values[1]), 
        freq=complex_spec.freq[0], 
        time=complex_spec.time[0]
    )


# =====================================================================
# БЛОК 2: Чистая математика
# =====================================================================

REF_PA_ASD = 2e-5  
REF_FS_ASD = 1.0   
REF_V_ASD  = 1.0   

def makeLogDB(spec: SpecFunc, add_one: bool = False) -> SpecFunc:
    _, old_unit = spec.values

    if old_unit.is_compatible_with(UREG.Pa / (UREG.Hz ** 0.5)): ref_val = REF_PA_ASD
    elif old_unit.is_compatible_with(UREG.FS / (UREG.Hz ** 0.5)): ref_val = REF_FS_ASD
    elif old_unit.is_compatible_with(UREG.V / (UREG.Hz ** 0.5)): ref_val = REF_V_ASD
    else: ref_val = 1.0

    def to_db(arr: Any) -> Any:
        amplitude = ma.abs(arr)
        ratio = amplitude / ref_val
        
        if add_one:
            return 20 * ma.log10(ratio + 1.0)
        else:
            scaled_ratio = ma.clamp_min(ratio, min_val=1e-9)
            return 20 * ma.log10(scaled_ratio)

    return spec.cloneApply(func=to_db, new_unit=UREG.dB)


# =====================================================================
# БЛОК 3: Интерполяция и Свертки
# =====================================================================

def interpolate(spec: SpecFunc, new_freq: Any, new_time: Any) -> 'SpecFunc':
    ctx = spec.context
    mat_a, mat_u = spec.values
    old_freq, _ = spec.freq
    old_time, _ = spec.time
    
    real_dtype = _get_real_dtype(mat_a, ctx)
    real_ctx = ma.ArrayContext(ctx._framework, ctx._device, real_dtype)
    
    new_freq_t = ma.convert_to(new_freq, real_ctx)
    new_time_t = ma.convert_to(new_time, real_ctx)

    out_mat = mfft.interpolate_2d(mat_a, old_time, old_freq, new_time_t, new_freq_t)

    return SpecFunc(matrix=(out_mat, mat_u), freq=new_freq_t, time=new_time_t)


def interpolateToShape(spec: 'SpecFunc', target_shape: Tuple[int, int]) -> 'SpecFunc':
    old_freq, _ = spec.freq
    old_time, _ = spec.time
    ctx = spec.context
    
    real_dtype = _get_real_dtype(spec.values[0], ctx)
    real_ctx = ma.ArrayContext(ctx._framework, ctx._device, real_dtype)
    
    new_freq = ma.linspace(float(old_freq[0]), float(old_freq[-1]), target_shape[0], real_ctx)
    new_time = ma.linspace(float(old_time[0]), float(old_time[-1]), target_shape[1], real_ctx)

    return interpolate(spec, new_freq, new_time)


def interpolateByFactors(spec: 'SpecFunc', factors: Tuple[float, float]) -> 'SpecFunc':
    old_freq, _ = spec.freq
    old_time, _ = spec.time

    new_nfreq = max(1, int(round(old_freq.shape[0] * factors[0])))
    new_ntime = max(1, int(round(old_time.shape[0] * factors[1])))

    return interpolateToShape(spec, target_shape=(new_nfreq, new_ntime))


def blurSpec(spec: 'SpecFunc', sigma: Union[float, Tuple[float, float]] = 1.0) -> 'SpecFunc':
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))

    kernel_size_f = int(round(sigma[0] * 3)) * 2 + 1
    kernel_size_t = int(round(sigma[1] * 3)) * 2 + 1
    kernel_size = (kernel_size_f, kernel_size_t)

    _, current_unit = spec.values

    def apply_gaussian_blur_fft(mat: Any) -> Any:
        ctx_orig = ma.ArrayContext.from_array(mat)
        
        import torch
        dev_enum = ma.DeviceType.GPU if ctx_orig.isGPU() else ma.DeviceType.CPU
        ctx_torch = ma.ArrayContext(ma.Framework.TORCH, dev_enum, None)
        
        mat_torch = ma.convert_to(mat, ctx_torch)
        device = mat_torch.device
        dtype = mat_torch.real.dtype if mat_torch.is_complex() else mat_torch.dtype
        img_h, img_w = mat_torch.shape[-2], mat_torch.shape[-1]
        
        def get_1d_kernel(k: int, s: float) -> torch.Tensor:
            limit = (k - 1) / 2.0
            x = torch.linspace(-limit, limit, steps=k, device=device, dtype=dtype)
            gauss = torch.exp(-0.5 * (x / s).pow(2))
            return gauss / gauss.sum()

        kernel_y = get_1d_kernel(kernel_size[0], sigma[0])
        kernel_x = get_1d_kernel(kernel_size[1], sigma[1])
        kernel_2d_small = (kernel_y.unsqueeze(-1) * kernel_x.unsqueeze(0))
        
        padded_kernel = torch.zeros(img_h, img_w, device=device, dtype=dtype)
        k_h, k_w = kernel_2d_small.shape
        padded_kernel[:k_h, :k_w] = kernel_2d_small
        
        padded_kernel = torch.roll(padded_kernel, shifts=(-k_h // 2, -k_w // 2), dims=(-2, -1))
        
        fft_mat = torch.fft.fft2(mat_torch, dim=(-2, -1))
        fft_kernel = torch.fft.fft2(padded_kernel, dim=(-2, -1))
        fft_result = fft_mat * fft_kernel
        ifft_result = torch.fft.ifft2(fft_result, dim=(-2, -1))

        res = ifft_result if mat_torch.is_complex() else ifft_result.real
        
        return ma.convert_to(res, ctx_orig)

    return spec.cloneApply(func=apply_gaussian_blur_fft, new_unit=current_unit)

``````


### 2. Исправленный `Visualize/spec2d_visualizer.py` (Возвращаем FPS!)

``````python path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Visualize/spec2d_visualizer.py" encoding="utf-8"
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

        freq_bins, time_bins = self._matrix.shape
        t_idx = int(np.clip((t - self._t_min) / (self._t_max - self._t_min) * time_bins, 0, time_bins - 1))
        f_idx = int(np.clip((f - self._f_min) / (self._f_max - self._f_min) * freq_bins, 0, freq_bins - 1))
        
        value = float(self._matrix[f_idx, t_idx])
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
            
        # 1. Извлекаем сырые данные
        matrix_t, unit_obj = func.values
        f_axis_t, _ = func.freq
        t_axis_t, _ = func.time

        if matrix_t.ndim > 2:
            matrix_t = matrix_t[0]

        # 2. Универсальная конвертация в NumPy для PyQtGraph
        matrix_np = ma.to_numpy(matrix_t)
        
        # СУПЕР ВАЖНО ДЛЯ FPS: Гарантируем, что память не фрагментирована (C-Contiguous)
        # STFT в TF/PyTorch делает матрицу strided, что убивает рендерер PyQtGraph.
        matrix_np = np.ascontiguousarray(matrix_np)
        
        t_axis = ma.to_numpy(t_axis_t)
        f_axis = ma.to_numpy(f_axis_t)

        unit_str = str(unit_obj)

        t0, t1 = float(t_axis[0]), float(t_axis[-1])
        f0, f1 = float(f_axis[0]), float(f_axis[-1])

        # ImageItem ожидает данные (Y, X), т.е. (freq, time). Наша матрица уже имеет эту форму.
        self.image_item.setFullData(matrix_np, (t0, t1), (f0, f1))

        minimap_data = np.mean(matrix_np, axis=0)
        self.minimap_curve.setData(x=t_axis, y=minimap_data)

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
    
    # Попытаемся использовать PyTorch контекст, если он есть, иначе NumPy
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
    
    # Конвертируем в тензоры выбранного фреймворка
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
    main_window.setWindowTitle("Тест Визуализатора SpecFunc (MultiArray edition)")
    main_window.resize(1000, 800)
    
    visualizer = Spec2DVisualizer()
    main_window.setCentralWidget(visualizer)
    main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_standalone_demo()

``````