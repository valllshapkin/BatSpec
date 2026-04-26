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
    Выполняет STFT с семантикой TensorFlow.
    Гарантирует ИДЕНТИЧНЫЙ размер матриц (Frequency x Time) во всех фреймворках!
    """
    ctx = ArrayContext.from_array(signal)
    if fft_length is None:
        fft_length = frame_length

    pad_len = 0
    sig_len = signal.shape[-1]

    # 1. Единый базовый паддинг: дополняем сигнал до ровного количества фреймов
    if pad_end:
        if sig_len < frame_length:
            pad_len = frame_length - sig_len
        else:
            rem = (sig_len - frame_length) % frame_step
            pad_len = (frame_step - rem) % frame_step if rem != 0 else 0

    # 2. Спец-паддинг для PyTorch: так как он нарезает куски по fft_length, ему нужен хвост длиннее
    if ctx.isTorch() and fft_length > frame_length:
        pad_len += (fft_length - frame_length)

    # 3. Применяем нули в конец
    if pad_len > 0:
        if ctx.isTensorflow():
            paddings = [[0, 0]] * (len(signal.shape) - 1) + [[0, pad_len]]
            signal = ctx.fw.pad(signal, paddings)
        elif ctx.isTorch():
            import torch.nn.functional as F
            signal = F.pad(signal, (0, pad_len))
        else:
            import numpy as np
            signal_np = to_numpy(signal)
            pad_tuple = [(0, 0)] * (signal_np.ndim - 1) + [(0, pad_len)]
            signal = np.pad(signal_np, pad_tuple)
            if not ctx.isNumpy():
                signal = convert_to(signal, ctx)

    # 4. Выполняем строгое STFT (без внутренних додумок фреймворков)
    if ctx.isTensorflow():
        win_fn = (lambda *args, **kwargs: window) if window is not None else None
        # pad_end=False, так как мы уже всё западили сами!
        res = ctx.fw.signal.stft(signal, frame_length=frame_length, frame_step=frame_step, 
                                 fft_length=fft_length, window_fn=win_fn, pad_end=False)
        return ctx.fw.linalg.matrix_transpose(res)
                                  
    elif ctx.isTorch():
        import torch.nn.functional as F
        # Защита от симметричного паддинга PyTorch: дополняем окно справа вручную
        if window is not None and fft_length > frame_length:
            pt_window = F.pad(window, (0, fft_length - frame_length))
        else:
            pt_window = window
            
        # Говорим PyTorch, что окно уже нужного размера, чтобы он не лез его центрировать
        stft_matrix = ctx.fw.stft(signal, n_fft=fft_length, hop_length=frame_step, 
                                  win_length=fft_length, window=pt_window, 
                                  center=False, return_complex=True)
        return stft_matrix
        
    else:
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view
        
        np_sig = to_numpy(signal)
        np_win = to_numpy(window) if window is not None else np.ones(frame_length, dtype=np_sig.dtype)
        
        frames = sliding_window_view(np_sig, frame_length, axis=-1)
        frames = frames[..., ::frame_step, :]
        frames = frames * np_win
        
        # np.fft.rfft сам добавляет нули справа, если n > frame_length
        res = np.fft.rfft(frames, n=fft_length, axis=-1)
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
        stfts = ctx.fw.linalg.matrix_transpose(stft_matrix)
        res = ctx.fw.signal.inverse_stft(stfts, frame_length=frame_length, 
                                         frame_step=frame_step, fft_length=fft_length, 
                                         window_fn=win_fn)
        if length is not None:
            res = res[..., :length]
        return res
        
    elif ctx.isTorch():
        stfts = stft_matrix.transpose(-2, -1)
        frames = ctx.fw.fft.irfft(stfts, n=fft_length, dim=-1)
        
        frames = frames[..., :frame_length]
        
        if window is not None:
            frames = frames * window[:frames.shape[-1]]
            
        num_frames = frames.shape[-2]
        expected_len = (num_frames - 1) * frame_step + frame_length
        
        out_shape = list(frames.shape[:-2]) + [expected_len]
        out_sig = ctx.fw.zeros(out_shape, dtype=frames.dtype, device=frames.device)
        
        for i in range(num_frames):
            start = i * frame_step
            out_sig[..., start:start+frames.shape[-1]] += frames[..., i, :]
            
        if length is not None:
            out_sig = out_sig[..., :length]
            
        return out_sig
                            
    else:
        import numpy as np
        stfts = np.swapaxes(to_numpy(stft_matrix), -2, -1)
        win = to_numpy(window) if window is not None else 1.0
        
        frames = np.fft.irfft(stfts, n=fft_length, axis=-1)
        
        frames = frames[..., :frame_length]
        
        if window is not None:
            frames = frames * win[:frames.shape[-1]]
        
        num_frames = frames.shape[-2]
        expected_len = (num_frames - 1) * frame_step + frame_length
        
        out_shape = list(frames.shape[:-2]) + [expected_len]
        out_sig = np.zeros(out_shape, dtype=frames.dtype)
        
        for i in range(num_frames):
            start = i * frame_step
            out_sig[..., start:start+frames.shape[-1]] += frames[..., i, :]
            
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