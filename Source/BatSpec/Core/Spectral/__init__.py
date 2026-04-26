import math
import threading
from contextlib import contextmanager
from typing import Tuple, Union, Any
import warnings

# --- Внедряем наш новый универсальный фреймворк ---
import MultiArray as ma
import MultiArray.FFT as mfft

from BatSpec.Core.Functions import TimeFunc, SpecFunc
from BatSpec.Core.ConvWindow import Window, WindowNorm, WindowNormMismatchError
from BatSpec.Core.Physical.Units import UREG, unit_devide, unit_sqrt, unit_mul


# =====================================================================
# КОНТЕКСТНЫЙ МЕНЕДЖЕР ВЫЧИСЛЕНИЙ
# =====================================================================

_dsp_tls = threading.local()

@contextmanager
def DSPContext(ctx: ma.ArrayContext):
    """
    Контекстный менеджер для переопределения вычислительного бэкенда (фреймворка и устройства).
    """
    old_ctx = getattr(_dsp_tls, 'math_ctx', None)
    _dsp_tls.math_ctx = ctx
    try:
        yield
    finally:
        _dsp_tls.math_ctx = old_ctx

def _get_math_ctx(default_ctx: ma.ArrayContext) -> ma.ArrayContext:
    override = getattr(_dsp_tls, 'math_ctx', None)
    return override if override is not None else default_ctx


def _get_real_dtype(tensor: Any, ctx: ma.ArrayContext) -> Any:
    """Безопасно извлекает вещественный тип данных."""
    if ctx.isTensorflow():
        return tensor.dtype.real_dtype if hasattr(tensor.dtype, 'real_dtype') else tensor.dtype
    elif ctx.isTorch():
        return tensor.real.dtype if tensor.is_complex() else tensor.dtype
    else:
        return tensor.real.dtype if hasattr(tensor, 'real') else getattr(tensor, 'dtype', None)

# =====================================================================
# БЛОК 1: Сложные DSP функции (STFT, IFFT)
# =====================================================================

def makeComplexSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)

    orig_ctx = signal.context
    math_ctx = _get_math_ctx(orig_ctx)

    signal_math = signal.to_context(math_ctx)
    signal_a, signal_unit = signal_math.values
    sr = signal_math.sr

    real_dtype = _get_real_dtype(signal_a, math_ctx)
    real_ctx = ma.ArrayContext(math_ctx._framework, math_ctx._device, real_dtype)
    
    orig_win_length = int(window.time * sr)
    fft_length = (bins - 1) * 2
    hop = max(1, int(orig_win_length * (1 - overlap)))

    win = window.get_array(sr, real_ctx)

    # ====================================================================
    # ЭКСТРЕМАЛЬНЫЙ DSP: Если запрошенное разрешение меньше размера окна, 
    # считаем спектрограмму точным матричным умножением (Direct DTFT)!
    # Это позволяет использовать ГИГАНТСКИЕ окна без потерь данных.
    # ====================================================================
    if fft_length < orig_win_length:
        warnings.warn(f"fft_length ({fft_length}) < win_length ({orig_win_length}). "
                      "Используется точный матричный расчет DTFT (Без обрезки окна!).")
        
        # 1. Единый паддинг (дописываем нули до ровного количества фреймов)
        pad_len = 0
        sig_len = signal_a.shape[-1]
        if sig_len < orig_win_length:
            pad_len = orig_win_length - sig_len
        else:
            rem = (sig_len - orig_win_length) % hop
            pad_len = (hop - rem) % hop if rem != 0 else 0
            
        if pad_len > 0:
            if math_ctx.isTensorflow():
                paddings = [[0, 0]] * (len(signal_a.shape) - 1) + [[0, pad_len]]
                signal_a = math_ctx.fw.pad(signal_a, paddings)
            elif math_ctx.isTorch():
                import torch.nn.functional as F
                signal_a = F.pad(signal_a, (0, pad_len))
            else:
                import numpy as np
                sig_np = ma.to_numpy(signal_a)
                pad_tuple = [(0, 0)] * (sig_np.ndim - 1) + [(0, pad_len)]
                signal_a = ma.convert_to(np.pad(sig_np, pad_tuple), math_ctx)

        # 2. Нарезаем сигнал на фреймы (shape: ..., Frames, WinLength)
        if math_ctx.isTorch():
            frames = signal_a.unfold(-1, orig_win_length, hop)
        elif math_ctx.isTensorflow():
            frames = math_ctx.fw.signal.frame(signal_a, orig_win_length, hop)
        else:
            import numpy as np
            from numpy.lib.stride_tricks import sliding_window_view
            frames_np = sliding_window_view(ma.to_numpy(signal_a), orig_win_length, axis=-1)
            frames_np = frames_np[..., ::hop, :]
            frames = ma.convert_to(frames_np, math_ctx)
            
        # 3. Готовим комплексный базис DTFT (Точное значение частот)
        n_vec = ma.reshape(ma.arange(orig_win_length, real_ctx), (orig_win_length, 1))
        k_vec = ma.reshape(ma.arange(bins, real_ctx), (1, bins))
        
        # фаза = -2 * pi * n * k / N_fft
        phase = (-2.0 * math.pi) * (n_vec * k_vec) / float(fft_length)
        
        # Умножаем базис на окно
        win_col = ma.reshape(win, (orig_win_length, 1))
        E_real = win_col * ma.cos(phase)
        E_imag = win_col * ma.sin(phase)
        
        # 4. Мощь матричного умножения: вычисляем все фреймы разом!
        real_part = ma.matmul(frames, E_real)
        imag_part = ma.matmul(frames, E_imag)
        
        # 5. Собираем обратно в формат [..., Bins, Frames]
        if math_ctx.isTorch():
            import torch
            stft = torch.complex(real_part, imag_part).transpose(-2, -1)
        elif math_ctx.isTensorflow():
            import tensorflow as tf
            stft = tf.linalg.matrix_transpose(tf.complex(real_part, imag_part))
        else:
            import numpy as np
            stft_np = ma.to_numpy(real_part) + 1j * ma.to_numpy(imag_part)
            stft = ma.convert_to(np.swapaxes(stft_np, -2, -1), math_ctx)
            
    else:
        # Стандартный путь: Обычный FFT
        if fft_length > orig_win_length:
            warnings.warn(f"fft_length={fft_length} больше win_length={orig_win_length}. Добавление нулей (Zero-padding).")
        stft = mfft.stft(signal_a, frame_length=orig_win_length, frame_step=hop, fft_length=fft_length, window=win, pad_end=True)

    # --- Общая финализация осей и масштаба ---
    scale_factor = math.sqrt(2.0 / sr)
    num_frames = stft.shape[-1]
    
    freq_axis = mfft.rfftfreq(fft_length, d=1.0 / sr, ctx=real_ctx)
    time_axis = ma.arange(num_frames, real_ctx) * (hop / sr)
    spec_unit = unit_devide(signal_unit, unit_sqrt(UREG.Hz))

    spec_math = SpecFunc(matrix=(stft * scale_factor, spec_unit), freq=freq_axis, time=time_axis)
    
    return spec_math.to_context(orig_ctx)


def inverseComplexSpec(spec: SpecFunc, window: Window, overlap: float = 0.5) -> TimeFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)
    
    orig_ctx = spec.context
    math_ctx = _get_math_ctx(orig_ctx)

    spec_math = spec.to_context(math_ctx)
    complex_matrix, spec_unit = spec_math.values
    
    real_dtype = _get_real_dtype(complex_matrix, math_ctx)
    real_ctx = ma.ArrayContext(math_ctx._framework, math_ctx._device, real_dtype) 
    
    freq_axis, _ = spec_math.freq

    fft_length = (len(freq_axis) - 1) * 2
    sr_f_val, _ = spec_math.df 
    sr = int(round(sr_f_val * fft_length))
    
    orig_win_length = int(window.time * sr)
    hop = max(1, int(orig_win_length * (1 - overlap)))
    
    win = window.get_array(sr, real_ctx)

    # iSTFT физически не может восстановить сигнал с перекрытием, если fft_length < win_length
    # В этом редком случае (если кто-то решил обратить такую спектрограмму) мы вынуждены симметрично обрезать окно.
    if fft_length < orig_win_length:
        start = (orig_win_length - fft_length) // 2
        win = win[start : start + fft_length]
        win_length = fft_length
    else:
        win_length = orig_win_length

    scale_factor = math.sqrt(2.0 / sr)
    unscaled = complex_matrix / scale_factor

    signal_a = mfft.istft(unscaled, frame_length=win_length, frame_step=hop, fft_length=fft_length, window=win)
    
    if hasattr(signal_a, 'real') and not math_ctx.isTorch():
        signal_a = signal_a.real
    elif math_ctx.isTorch() and signal_a.is_complex():
        signal_a = signal_a.real
    
    new_time_axis = ma.arange(signal_a.shape[-1], real_ctx) / float(sr)
    signal_unit = unit_mul(spec_unit, unit_sqrt(UREG.Hz))

    time_math = TimeFunc(values=(signal_a, signal_unit), axis=new_time_axis)
    
    return time_math.to_context(orig_ctx)


def makeSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    orig_ctx = signal.context
    math_ctx = _get_math_ctx(orig_ctx)

    signal_math = signal.to_context(math_ctx)
    complex_spec_math = makeComplexSpec(signal_math, window, overlap, bins)
    
    amp_matrix = ma.abs(complex_spec_math.values[0])
    
    spec_math = SpecFunc(
        matrix=(amp_matrix, complex_spec_math.values[1]), 
        freq=complex_spec_math.freq[0], 
        time=complex_spec_math.time[0]
    )
    
    return spec_math.to_context(orig_ctx)


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