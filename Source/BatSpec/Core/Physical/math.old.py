import warnings
from typing import Any, Tuple, Union

# --- Базовый хелпер ---
def get_fw(x: Any) -> str:
    """Возвращает название фреймворка ('numpy', 'torch', 'tensorflow')."""
    type_str = str(type(x)).lower()
    if 'torch' in type_str: return 'torch'
    if 'tensorflow' in type_str: return 'tensorflow'
    return 'numpy'

def get_device(x: Any) -> Any:
    return getattr(x, 'device', 'cpu')

# ==========================================
# 1. БАЗОВАЯ МАТЕМАТИКА
# ==========================================

def abs(x: Any) -> Any:
    fw = get_fw(x)
    if fw == 'torch': import torch; return torch.abs(x)
    elif fw == 'tensorflow': import tensorflow as tf; return tf.math.abs(x)
    else: import numpy as np; return np.abs(x)

def log10(x: Any) -> Any:
    fw = get_fw(x)
    if fw == 'torch': import torch; return torch.log10(x)
    elif fw == 'tensorflow': 
        import tensorflow as tf
        return tf.math.log(x) / tf.math.log(tf.constant(10.0, dtype=x.dtype))
    else: import numpy as np; return np.log10(x)

def clamp_min(x: Any, min_val: float) -> Any:
    fw = get_fw(x)
    if fw == 'torch': import torch; return torch.clamp(x, min=min_val)
    elif fw == 'tensorflow': import tensorflow as tf; return tf.maximum(x, min_val)
    else: import numpy as np; return np.clip(x, a_min=min_val, a_max=None)

def exp(x: Any) -> Any:
    fw = get_fw(x)
    if fw == 'torch': import torch; return torch.exp(x)
    elif fw == 'tensorflow': import tensorflow as tf; return tf.math.exp(x)
    else: import numpy as np; return np.exp(x)


# ==========================================
# 2. ГЕНЕРАЦИЯ ОСЕЙ И СЕТОК
# ==========================================

def linspace(start: float, end: float, steps: int, ref_tensor: Any) -> Any:
    fw = get_fw(ref_tensor)
    if fw == 'torch':
        import torch
        return torch.linspace(start, end, steps, device=get_device(ref_tensor))
    elif fw == 'tensorflow':
        import tensorflow as tf
        return tf.linspace(start, end, steps)
    else:
        import numpy as np
        return np.linspace(start, end, steps)

def arange(end: int, dtype: Any, ref_tensor: Any) -> Any:
    fw = get_fw(ref_tensor)
    if fw == 'torch':
        import torch
        return torch.arange(end, dtype=dtype, device=get_device(ref_tensor))
    elif fw == 'tensorflow':
        import tensorflow as tf
        return tf.range(end, dtype=dtype)
    else:
        import numpy as np
        return np.arange(end, dtype=dtype)


# ==========================================
# 3. FFT И ЧАСТОТНЫЕ ОСИ (1D и 2D)
# ==========================================

def fftfreq(n: int, d: float, ref_tensor: Any) -> Any:
    fw = get_fw(ref_tensor)
    if fw == 'torch': import torch; return torch.fft.fftfreq(n, d=d, device=get_device(ref_tensor))
    else: import numpy as np; return np.fft.fftfreq(n, d=d)  # Для numpy и TF (TF не имеет удобного fftfreq)

def rfftfreq(n: int, d: float, ref_tensor: Any) -> Any:
    fw = get_fw(ref_tensor)
    if fw == 'torch': import torch; return torch.fft.rfftfreq(n, d=d, device=get_device(ref_tensor))
    else: import numpy as np; return np.fft.rfftfreq(n, d=d)

def fft2(x: Any, dim: Tuple[int, int] = (-2, -1)) -> Any:
    fw = get_fw(x)
    if fw == 'torch': import torch; return torch.fft.fft2(x, dim=dim)
    elif fw == 'tensorflow': import tensorflow as tf; return tf.signal.fft2d(x)
    else: import numpy as np; return np.fft.fft2(x, axes=dim)

def ifft2(x: Any, dim: Tuple[int, int] = (-2, -1)) -> Any:
    fw = get_fw(x)
    if fw == 'torch': import torch; return torch.fft.ifft2(x, dim=dim)
    elif fw == 'tensorflow': import tensorflow as tf; return tf.signal.ifft2d(x)
    else: import numpy as np; return np.fft.ifft2(x, axes=dim)

def rfft2(x: Any, dim: Tuple[int, int] = (-2, -1)) -> Any:
    fw = get_fw(x)
    if fw == 'torch': import torch; return torch.fft.rfft2(x, dim=dim)
    else: import numpy as np; return np.fft.rfft2(x, axes=dim)

def irfft2(x: Any, s: Tuple[int, int], dim: Tuple[int, int] = (-2, -1)) -> Any:
    fw = get_fw(x)
    if fw == 'torch': import torch; return torch.fft.irfft2(x, s=s, dim=dim)
    else: import numpy as np; return np.fft.irfft2(x, s=s, axes=dim)


# ==========================================
# 4. СЛОЖНАЯ ОБРАБОТКА СИГНАЛОВ (STFT / DSP)
# ==========================================

def stft(signal: Any, n_fft: int, hop_length: int, win_length: int, window: Any, center: bool = True) -> Any:
    """Выполняет STFT. Возвращает комплексную матрицу спектрограммы."""
    fw = get_fw(signal)
    
    if fw == 'torch':
        import torch
        return torch.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                          window=window, center=center, return_complex=True)
    elif fw == 'tensorflow':
        import tensorflow as tf
        # tf.signal.stft ожидает, что pad_end аналог center=True, но сдвиги могут отличаться
        return tf.signal.stft(signal, frame_length=win_length, frame_step=hop_length, 
                              fft_length=n_fft, window_fn=lambda _: window, pad_end=center)
    else:
        import scipy.signal as sig
        import numpy as np
        # scipy.signal.stft требует noverlap вместо hop_length
        noverlap = win_length - hop_length
        boundary = 'zeros' if center else None
        
        # Scipy возвращает (f, t, Zxx). Нам нужна только матрица Zxx.
        _, _, Zxx = sig.stft(signal, fs=1.0, window=window, nperseg=win_length, 
                             noverlap=noverlap, nfft=n_fft, boundary=boundary)
        
        # Спектр scipy нормируется по-другому, подгоняем под масштаб PyTorch
        scaling_factor = win_length / 2.0 
        return Zxx * scaling_factor


def istft(stft_matrix: Any, n_fft: int, hop_length: int, win_length: int, window: Any, center: bool = True, length: int = None) -> Any:
    """Выполняет обратное STFT."""
    fw = get_fw(stft_matrix)
    
    if fw == 'torch':
        import torch
        return torch.istft(stft_matrix, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                           window=window, center=center, length=length)
    else:
        import scipy.signal as sig
        noverlap = win_length - hop_length
        boundary = True if center else False
        
        # Снимаем поправку масштаба, которую применили в stft
        scaling_factor = win_length / 2.0
        stft_matrix = stft_matrix / scaling_factor
        
        _, signal = sig.istft(stft_matrix, fs=1.0, window=window, nperseg=win_length, 
                              noverlap=noverlap, nfft=n_fft, boundary=boundary, time_axis=-1)
        if length is not None:
            signal = signal[..., :length]
        return signal


# ==========================================
# 5. ИНТЕРПОЛЯЦИЯ 2D СЕТОК (Без PyTorch F.grid_sample)
# ==========================================

def interpolate_2d(matrix: Any, old_x: Any, old_y: Any, new_x: Any, new_y: Any) -> Any:
    """Интерполирует 2D матрицу. Заменяет медленный и сложный grid_sample."""
    fw = get_fw(matrix)
    
    if fw == 'torch':
        import torch
        import torch.nn.functional as F
        # Ваш старый код с normalize_coords и F.grid_sample здесь
        pass # Для экономии места оставляю структуру. Вставьте свой код с grid_sample для torch
        
    else:
        # Идеальный, быстрый и нативный способ для CPU
        import numpy as np
        from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
        
        is_complex = np.iscomplexobj(matrix)
        
        # Scipy RectBivariateSpline работает с 2D. Для батчей используем RegularGridInterpolator
        if matrix.ndim == 2:
            if is_complex:
                spline_r = RectBivariateSpline(old_y, old_x, matrix.real)
                spline_i = RectBivariateSpline(old_y, old_x, matrix.imag)
                return spline_r(new_y, new_x) + 1j * spline_i(new_y, new_x)
            else:
                spline = RectBivariateSpline(old_y, old_x, matrix)
                return spline(new_y, new_x)
        else:
            # Универсально для любой размерности (батчей)
            interpolator = RegularGridInterpolator((old_y, old_x), matrix, method='linear', bounds_error=False, fill_value=0)
            X, Y = np.meshgrid(new_x, new_y)
            pts = np.stack([Y.ravel(), X.ravel()], axis=-1)
            res = interpolator(pts).reshape(*matrix.shape[:-2], len(new_y), len(new_x))
            return res