Отлично! Понял две главные вещи:
1.  Тебе нужна размерность **`[..., bins, frames]` (Frequency, Time)** по умолчанию для всех фреймворков (так как это стандарт для 2D-свёрток и классического DSP).
2.  Мы поймали две ошибки:
    *   **TensorFlow** пытался передать аргумент `dtype` в нашу лямбду для `window_fn`. Это легко лечится с помощью `*args, **kwargs`.
    *   **PyTorch** упал с `window overlap add min`, так как его нативный `istft` с параметром `center=False` слишком жёстко проверяет границы нулей в окне Ханнинга при делении на огибающую окна. Чтобы сделать поведение PyTorch на 100% идентичным TensorFlow (который просто умножает и складывает, без деления), я перевёл PyTorch на чистый и понятный Overlap-Add через нативный `irfft`.

Теперь все три фреймворка гарантированно отдают `[..., bins, frames]` и работают по единому стандарту!

### 1. Обновленный `FFT.py` (С новой формой и фиксами)

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/MultiArray/FFT.py" encoding="utf-8"
from typing import Any, Tuple
from .Core import ArrayContext
from .Base import convert_to, to_numpy

# ==========================================
# 1. ЧАСТОТНЫЕ ОСИ
# ==========================================

def fftfreq(n: int, d: float, ctx: 'ArrayContext') -> Any:
    if ctx.isTorch():
        return ctx.fw.fft.fftfreq(n, d=d, device=ctx.device)
    else:
        import numpy as np
        arr = np.fft.fftfreq(n, d=d)
        return convert_to(arr, ctx)

def rfftfreq(n: int, d: float, ctx: 'ArrayContext') -> Any:
    if ctx.isTorch():
        return ctx.fw.fft.rfftfreq(n, d=d, device=ctx.device)
    else:
        import numpy as np
        arr = np.fft.rfftfreq(n, d=d)
        return convert_to(arr, ctx)

# ==========================================
# 2. 2D ПРЕОБРАЗОВАНИЯ ФУРЬЕ (FFT)
# ==========================================

def fft2(x: Any, axes: Tuple[int, int] = (-2, -1)) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.fft.fft2(x, dim=axes)
    elif ctx.isTensorflow():
        return ctx.fw.signal.fft2d(x)
    else:
        return ctx.fw.fft.fft2(x, axes=axes)

def ifft2(x: Any, axes: Tuple[int, int] = (-2, -1)) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.fft.ifft2(x, dim=axes)
    elif ctx.isTensorflow():
        return ctx.fw.signal.ifft2d(x)
    else:
        return ctx.fw.fft.ifft2(x, axes=axes)

def rfft2(x: Any, axes: Tuple[int, int] = (-2, -1)) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.fft.rfft2(x, dim=axes)
    elif ctx.isTensorflow():
        return ctx.fw.signal.rfft2d(x)
    else:
        return ctx.fw.fft.rfft2(x, axes=axes)

def irfft2(x: Any, s: Tuple[int, int] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.fft.irfft2(x, s=s, dim=axes)
    elif ctx.isTensorflow():
        return ctx.fw.signal.irfft2d(x)
    else:
        return ctx.fw.fft.irfft2(x, s=s, axes=axes)

# ==========================================
# 3. СЛОЖНАЯ ОБРАБОТКА СИГНАЛОВ (DSP)
# ==========================================

def stft(signal: Any, frame_length: int, frame_step: int, fft_length: int = None, 
         window: Any = None, pad_end: bool = False) -> Any:
    """
    Выполняет STFT с семантикой TensorFlow, НО!
    Возвращает матрицу с формой [..., bins, frames] (Frequency x Time).
    """
    ctx = ArrayContext.from_array(signal)
    if fft_length is None:
        fft_length = frame_length
        
    if ctx.isTensorflow():
        # Добавляем *args и **kwargs, так как TF попытается передать dtype
        win_fn = (lambda *args, **kwargs: window) if window is not None else None
        res = ctx.fw.signal.stft(signal, frame_length=frame_length, frame_step=frame_step, 
                                 fft_length=fft_length, window_fn=win_fn, pad_end=pad_end)
        # TF возвращает [..., frames, bins], транспонируем в [..., bins, frames]
        return ctx.fw.linalg.matrix_transpose(res)
                                  
    elif ctx.isTorch():
        import torch.nn.functional as F
        
        if pad_end:
            sig_len = signal.shape[-1]
            if sig_len < frame_length:
                pad_len = frame_length - sig_len
            else:
                rem = (sig_len - frame_length) % frame_step
                pad_len = (frame_step - rem) % frame_step
            signal = F.pad(signal, (0, pad_len))
            
        stft_matrix = ctx.fw.stft(signal, n_fft=fft_length, hop_length=frame_step, 
                                  win_length=frame_length, window=window, 
                                  center=False, return_complex=True)
        # PyTorch И ТАК возвращает [..., bins, frames], ничего не меняем
        return stft_matrix
        
    else:
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view
        
        np_sig = to_numpy(signal)
        np_win = to_numpy(window) if window is not None else np.ones(frame_length, dtype=np_sig.dtype)
        
        if pad_end:
            sig_len = np_sig.shape[-1]
            if sig_len < frame_length:
                pad_len = frame_length - sig_len
            else:
                rem = (sig_len - frame_length) % frame_step
                pad_len = (frame_step - rem) % frame_step
            
            pad_tuple = [(0, 0)] * (np_sig.ndim - 1) + [(0, pad_len)]
            np_sig = np.pad(np_sig, pad_tuple)
            
        frames = sliding_window_view(np_sig, frame_length, axis=-1)
        frames = frames[..., ::frame_step, :]
        frames = frames * np_win
        
        res = np.fft.rfft(frames, n=fft_length, axis=-1)
        # NumPy возвращает [..., frames, bins], делаем свап осей
        res = np.swapaxes(res, -2, -1)
        
        target_ctx = ArrayContext(ctx._framework, ctx._device, None)
        return convert_to(res, target_ctx)


def istft(stft_matrix: Any, frame_length: int, frame_step: int, fft_length: int = None, 
          window: Any = None, length: int = None) -> Any:
    """
    Выполняет обратное STFT.
    Ожидает матрицу с формой [..., bins, frames] (Frequency x Time).
    """
    ctx = ArrayContext.from_array(stft_matrix)
    if fft_length is None:
        fft_length = frame_length
        
    if ctx.isTensorflow():
        win_fn = (lambda *args, **kwargs: window) if window is not None else None
        # Возвращаем матрицу обратно в формат TF [..., frames, bins]
        stfts = ctx.fw.linalg.matrix_transpose(stft_matrix)
        res = ctx.fw.signal.inverse_stft(stfts, frame_length=frame_length, 
                                         frame_step=frame_step, fft_length=fft_length, 
                                         window_fn=win_fn)
        if length is not None:
            res = res[..., :length]
        return res
        
    elif ctx.isTorch():
        # Встроенный PyTorch istft падает с "window overlap add min" если окна по краям нулевые.
        # Чтобы поведение было на 100% идентично NumPy и TF, мы делаем прямой Overlap-Add.
        # Формат на входе: [..., bins, frames] -> транспонируем для irfft
        stfts = stft_matrix.transpose(-2, -1)
        frames = ctx.fw.fft.irfft(stfts, n=fft_length, dim=-1)
        
        if window is not None:
            frames = frames * window
            
        num_frames = frames.shape[-2]
        expected_len = (num_frames - 1) * frame_step + frame_length
        
        out_shape = list(frames.shape[:-2]) + [expected_len]
        out_sig = ctx.fw.zeros(out_shape, dtype=frames.dtype, device=frames.device)
        
        for i in range(num_frames):
            start = i * frame_step
            out_sig[..., start:start+frame_length] += frames[..., i, :frame_length]
            
        if length is not None:
            out_sig = out_sig[..., :length]
            
        return out_sig
                            
    else:
        import numpy as np
        # Разворачиваем [..., bins, frames] обратно в [..., frames, bins]
        stfts = np.swapaxes(to_numpy(stft_matrix), -2, -1)
        win = to_numpy(window) if window is not None else 1.0
        
        frames = np.fft.irfft(stfts, n=fft_length, axis=-1)
        frames = frames * win
        
        num_frames = frames.shape[-2]
        expected_len = (num_frames - 1) * frame_step + frame_length
        
        out_shape = list(frames.shape[:-2]) + [expected_len]
        out_sig = np.zeros(out_shape, dtype=frames.dtype)
        
        for i in range(num_frames):
            start = i * frame_step
            out_sig[..., start:start+frame_length] += frames[..., i, :frame_length]
            
        if length is not None:
            out_sig = out_sig[..., :length]
            
        target_ctx = ArrayContext(ctx._framework, ctx._device, None)
        return convert_to(out_sig, target_ctx)

# ==========================================
# 4. ИНТЕРПОЛЯЦИЯ 2D СЕТОК
# ==========================================

def interpolate_2d(matrix: Any, old_x: Any, old_y: Any, new_x: Any, new_y: Any) -> Any:
    ctx = ArrayContext.from_array(matrix)
    
    if ctx.isTorch():
        import torch
        import torch.nn.functional as F
        
        is_complex = matrix.is_complex()
        
        def normalize(coords, old_coords):
            c_min, c_max = old_coords[0], old_coords[-1]
            return 2.0 * (coords - c_min) / (c_max - c_min) - 1.0
            
        norm_x = normalize(new_x, old_x)
        norm_y = normalize(new_y, old_y)
        
        grid_y, grid_x = torch.meshgrid(norm_y, norm_x, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        
        orig_shape = matrix.shape
        m = matrix.unsqueeze(0).unsqueeze(0) if len(orig_shape) == 2 else matrix.unsqueeze(1)
            
        if is_complex:
            out_r = F.grid_sample(m.real, grid.to(m.device), align_corners=True)
            out_i = F.grid_sample(m.imag, grid.to(m.device), align_corners=True)
            out = torch.complex(out_r, out_i)
        else:
            out = F.grid_sample(m, grid.to(m.device), align_corners=True)
            
        return out.squeeze(1).squeeze(0) if len(orig_shape) == 2 else out.squeeze(1)
        
    else:
        import numpy as np
        from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
        
        np_mat = to_numpy(matrix)
        nx, ny = to_numpy(new_x), to_numpy(new_y)
        ox, oy = to_numpy(old_x), to_numpy(old_y)
        
        is_complex = np.iscomplexobj(np_mat)
        
        if np_mat.ndim == 2:
            if is_complex:
                spline_r = RectBivariateSpline(oy, ox, np_mat.real)
                spline_i = RectBivariateSpline(oy, ox, np_mat.imag)
                res = spline_r(ny, nx) + 1j * spline_i(ny, nx)
            else:
                spline = RectBivariateSpline(oy, ox, np_mat)
                res = spline(ny, nx)
        else:
            interpolator = RegularGridInterpolator((oy, ox), np_mat, method='linear', bounds_error=False, fill_value=0)
            X, Y = np.meshgrid(nx, ny)
            pts = np.stack([Y.ravel(), X.ravel()], axis=-1)
            res = interpolator(pts).reshape(*np_mat.shape[:-2], len(ny), len(nx))
            
        return convert_to(res, ctx)

``````

### 2. Обновленный тест `test_fft.py`

Единственное изменение здесь — это строка `assert np_stft.shape[-2] == fft_length // 2 + 1`. Поскольку мы поменяли оси, теперь частоты (bins) находятся в предпоследнем измерении `[-2]`, а не в последнем.

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/MultiArray/test_fft.py" encoding="utf-8"
import pytest
import numpy as np

import MultiArray as ma
import MultiArray.FFT as mfft
from MultiArray.Core import ArrayContext, Framework, DeviceType

# ==========================================
# ОБНАРУЖЕНИЕ УСТАНОВЛЕННЫХ ФРЕЙМВОРКОВ
# ==========================================

AVAILABLE_FRAMEWORKS = [Framework.NUMPY]

try: import torch; AVAILABLE_FRAMEWORKS.append(Framework.TORCH)
except ImportError: pass

try: import tensorflow as tf; AVAILABLE_FRAMEWORKS.append(Framework.TENSORFLOW)
except ImportError: pass

try: import jax; AVAILABLE_FRAMEWORKS.append(Framework.JAX)
except ImportError: pass

CPU_CONTEXTS = [
    ArrayContext(fw, DeviceType.CPU, None) for fw in AVAILABLE_FRAMEWORKS 
    if fw != Framework.CUPY
]

def get_fw_name(ctx: ArrayContext) -> str:
    if ctx.isNumpy(): return "NUMPY"
    if ctx.isTorch(): return "TORCH"
    if ctx.isTensorflow(): return "TENSORFLOW"
    if ctx.isJax(): return "JAX"
    return "UNKNOWN"

# ==========================================
# ТЕСТЫ ДЛЯ FFT И DSP
# ==========================================

@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_math_extensions(ctx):
    np_arr = np.array([1.0, 10.0, 100.0], dtype=np.float32)
    arr = ma.convert_to(np_arr, ctx)
    
    # Test log10
    arr_log10 = ma.log10(arr)
    np.testing.assert_allclose(ma.to_numpy(arr_log10), [0.0, 1.0, 2.0], rtol=1e-5)
    
    # Test clamp
    arr_clamp = ma.clamp_min(arr, 5.0)
    np.testing.assert_allclose(ma.to_numpy(arr_clamp), [5.0, 10.0, 100.0])


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_fftfreq(ctx):
    n = 10
    d = 0.1
    
    freqs = mfft.fftfreq(n, d, ctx)
    expected = np.fft.fftfreq(n, d)
    
    np.testing.assert_allclose(ma.to_numpy(freqs), expected, rtol=1e-5)


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_fft2_ifft2(ctx):
    np_data = np.random.rand(4, 4).astype(np.float32)
    data = ma.convert_to(np_data, ctx)
    
    if ctx.isTensorflow():
        import tensorflow as tf
        data = tf.cast(data, tf.complex64)
    
    fft_res = mfft.fft2(data)
    ifft_res = mfft.ifft2(fft_res)
    
    np.testing.assert_allclose(np.real(ma.to_numpy(ifft_res)), np_data, atol=1e-5)


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_stft_istft(ctx):
    signal_len = 1024
    np_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, signal_len)).astype(np.float32)
    
    frame_length = 256
    frame_step = 64
    fft_length = 256
    np_window = np.hanning(frame_length).astype(np.float32)
    
    signal = ma.convert_to(np_signal, ctx)
    window = ma.convert_to(np_window, ctx)
    
    # STFT (Unified Style)
    stft_matrix = mfft.stft(signal, frame_length=frame_length, frame_step=frame_step, 
                            fft_length=fft_length, window=window, pad_end=True)
    
    np_stft = ma.to_numpy(stft_matrix)
    
    # Проверка нашей новой размерности [..., Bins, Frames]
    assert np_stft.shape[-2] == fft_length // 2 + 1
    
    # iSTFT
    recon_signal = mfft.istft(stft_matrix, frame_length=frame_length, frame_step=frame_step, 
                              fft_length=fft_length, window=window, length=signal_len)
    
    np_recon = ma.to_numpy(recon_signal)
    
    np_recon_norm = np_recon / np.max(np.abs(np_recon))
    np_sig_norm = np_signal / np.max(np.abs(np_signal))
    
    # Сравниваем обрезанные края из-за краевых эффектов окна
    np.testing.assert_allclose(np_sig_norm[256:-256], np_recon_norm[256:-256], atol=1e-3)

``````