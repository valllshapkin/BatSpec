from typing import Tuple, Union, Any
import warnings

# --- Импортируем нашу функцию конвертации сырых массивов ---
from BatSpec.Core.Physical.Arrays import convert_to_framework

# --- Межфреймворковые хелперы ---
def _get_fw_info(tensor: Any) -> Tuple[str, Any]:
    """Определяет фреймворк и устройство тензора."""
    type_str = str(type(tensor)).lower()
    dev = getattr(tensor, 'device', None)
    if 'torch' in type_str: return 'torch', dev
    if 'tensorflow' in type_str: return 'tensorflow', dev
    return 'numpy', dev

def _array_abs(x: Any) -> Any:
    type_str = str(type(x)).lower()
    if 'torch' in type_str:
        import torch
        return torch.abs(x)
    elif 'tensorflow' in type_str:
        import tensorflow as tf
        return tf.math.abs(x)
    else:
        import numpy as np
        return np.abs(x)

def _array_log10(x: Any) -> Any:
    type_str = str(type(x)).lower()
    if 'torch' in type_str:
        import torch
        return torch.log10(x)
    elif 'tensorflow' in type_str:
        import tensorflow as tf
        return tf.math.log(x) / tf.math.log(tf.constant(10.0, dtype=x.dtype))
    else:
        import numpy as np
        return np.log10(x)

def _array_clamp_min(x: Any, min_val: float) -> Any:
    type_str = str(type(x)).lower()
    if 'torch' in type_str:
        import torch
        return torch.clamp(x, min=min_val)
    elif 'tensorflow' in type_str:
        import tensorflow as tf
        return tf.maximum(x, min_val)
    else:
        import numpy as np
        return np.clip(x, a_min=min_val, a_max=None)


from BatSpec.Core.Functions import TimeFunc, SpecFunc
from BatSpec.Core.ConvWindow import Window, WindowNorm, WindowNormMismatchError
from BatSpec.Core.Physical.Units import UREG, unit_devide, unit_sqrt, unit_mul


# =====================================================================
# БЛОК 1: Сложные DSP функции (STFT, IFFT). Паттерн "PyTorch Bridge"
# =====================================================================

def makeComplexSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)

    # 1. Запоминаем исходный фреймворк и временно переводим в PyTorch
    fw_target, dev_target = _get_fw_info(signal.values[0])
    sig_torch = signal.to_framework('torch')
    
    import torch
    signal_a, signal_unit = sig_torch.values
    device = signal_a.device
    sr = sig_torch.sr

    # ИСПРАВЛЕНИЕ ЗДЕСЬ: используем convert_to_framework для сырого массива
    win_raw = window.get_array(sr, device=device)
    win = convert_to_framework(win_raw, 'torch', device)
    
    fft_length = (bins - 1) * 2
    win_length = len(win)
    hop = max(1, int(win_length * (1 - overlap)))

    if fft_length < win_length:
        warnings.warn(f"fft_length={fft_length} меньше win_length={win_length}. Фрейм будет обрезан.")
    if fft_length > win_length:
        warnings.warn(f"fft_length={fft_length} больше win_length={win_length}. Больше разрешения.")

    stft = torch.stft(
        signal_a, n_fft=fft_length, hop_length=hop, win_length=win_length,
        window=win, center=True, return_complex=True
    )

    scale = torch.sqrt(torch.tensor(2.0 / sr, dtype=torch.float64, device=device))
    num_frames = stft.shape[-1]
    
    freq_axis = torch.fft.rfftfreq(fft_length, d=1.0 / sr).to(device=device)
    time_axis = (torch.arange(num_frames, dtype=torch.float64, device=device) * hop) / sr
    spec_unit = unit_devide(signal_unit, unit_sqrt(UREG.Hz))

    spec_torch = SpecFunc(matrix=(stft * scale, spec_unit), freq=freq_axis, time=time_axis)
    
    # 2. Возвращаем в исходный формат
    return spec_torch.to_framework(fw_target, dev_target)


def inverseComplexSpec(spec: SpecFunc, window: Window, overlap: float = 0.5) -> TimeFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)
    
    fw_target, dev_target = _get_fw_info(spec.values[0])
    spec_torch = spec.to_framework('torch')

    import torch
    complex_matrix, spec_unit = spec_torch.values
    time_axis, _ = spec_torch.time
    freq_axis, _ = spec_torch.freq
    device = complex_matrix.device

    fft_length = (len(freq_axis) - 1) * 2
    sr_f_val, _ = spec_torch.df 
    sr = int(round(sr_f_val * fft_length))
    
    # ИСПРАВЛЕНИЕ ЗДЕСЬ
    win_raw = window.get_array(sr, device=device)
    win = convert_to_framework(win_raw, 'torch', device)
    
    win_length = len(win)
    hop = max(1, int(win_length * (1 - overlap)))

    scale = torch.sqrt(torch.tensor(2.0 / sr, dtype=torch.float64, device=device))
    unscaled = complex_matrix / scale

    signal_a = torch.istft(
        unscaled, n_fft=fft_length, hop_length=hop, win_length=win_length,
        window=win, center=True              
    )
    
    new_time_axis = torch.arange(signal_a.shape[-1], dtype=torch.float64, device=device) / sr
    signal_unit = unit_mul(spec_unit, unit_sqrt(UREG.Hz))

    sig_torch = TimeFunc(values=(signal_a, signal_unit), axis=new_time_axis)
    return sig_torch.to_framework(fw_target, dev_target)


def makeSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)

    fw_target, dev_target = _get_fw_info(signal.values[0])
    sig_torch = signal.to_framework('torch')

    import torch
    signal_a, signal_unit = sig_torch.values
    device = signal_a.device
    sr = sig_torch.sr

    # ИСПРАВЛЕНИЕ ЗДЕСЬ
    win_raw = window.get_array(sr, device=device)
    win = convert_to_framework(win_raw, 'torch', device)
    
    fft_length = (bins - 1) * 2
    win_length = len(win)
    hop = max(1, int(win_length * (1 - overlap)))

    stft = torch.stft(
        signal_a, n_fft=fft_length, hop_length=hop, win_length=win_length,
        window=win, center=True, return_complex=True
    )

    amplitude_matrix = torch.abs(stft)
    scale = torch.sqrt(torch.tensor(2.0 / sr, dtype=torch.float64, device=device))
    num_frames = amplitude_matrix.shape[-1]
    
    freq_axis = torch.fft.rfftfreq(fft_length, d=1.0 / sr).to(device=device)
    time_axis = (torch.arange(num_frames, dtype=torch.float64, device=device) * hop) / sr
    spec_unit = unit_devide(signal_unit, unit_sqrt(UREG.Hz))

    spec_torch = SpecFunc(matrix=(amplitude_matrix * scale, spec_unit), freq=freq_axis, time=time_axis)
    return spec_torch.to_framework(fw_target, dev_target)


# =====================================================================
# БЛОК 2: Чистая математика. Нативная межфреймворковость!
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
        amplitude = _array_abs(arr)
        ratio = amplitude / ref_val
        
        if add_one:
            return 20 * _array_log10(ratio + 1)
        else:
            scaled_ratio = _array_clamp_min(ratio, min_val=1e-9)
            return 20 * _array_log10(scaled_ratio)

    return spec.cloneApply(func=to_db, new_unit=UREG.dB)


# =====================================================================
# БЛОК 3: Интерполяция и Свертки. Снова "PyTorch Bridge"
# =====================================================================

def interpolate(spec: SpecFunc, new_freq: Any, new_time: Any) -> 'SpecFunc':
    fw_target, dev_target = _get_fw_info(spec.values[0])
    
    spec_torch = spec.to_framework('torch')
    import torch
    import torch.nn.functional as F

    mat_a, mat_u = spec_torch.values
    old_freq, _ = spec_torch.freq
    old_time, _ = spec_torch.time
    
    device = mat_a.device
    
    if not isinstance(new_freq, torch.Tensor): new_freq = torch.tensor(new_freq, device=device)
    if not isinstance(new_time, torch.Tensor): new_time = torch.tensor(new_time, device=device)

    def normalize_coords(new_ax, old_ax):
        min_val, max_val = old_ax[0], old_ax[-1]
        if min_val == max_val: return torch.zeros_like(new_ax)
        return 2.0 * (new_ax - min_val) / (max_val - min_val) - 1.0

    grid_x = normalize_coords(new_time, old_time)  
    grid_y = normalize_coords(new_freq, old_freq)  

    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    grid = torch.stack([mesh_x, mesh_y], dim=-1).unsqueeze(0) 
    
    original_shape = mat_a.shape
    batch_shape = original_shape[:-2]
    
    flat_N = int(torch.prod(torch.tensor(batch_shape))) if len(batch_shape) > 0 else 1
    mat_reshaped = mat_a.reshape(flat_N, 1, old_freq.shape[0], old_time.shape[0])
    
    grid_dtype = mat_reshaped.real.dtype if mat_reshaped.is_complex() else mat_reshaped.dtype
    grid = grid.expand(flat_N, -1, -1, -1).to(grid_dtype)

    if mat_reshaped.is_complex():
        real_part = F.grid_sample(mat_reshaped.real, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        imag_part = F.grid_sample(mat_reshaped.imag, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        out_reshaped = torch.complex(real_part, imag_part)
    else:
        out_reshaped = F.grid_sample(mat_reshaped, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    new_shape = (*batch_shape, new_freq.shape[0], new_time.shape[0])
    out_mat = out_reshaped.reshape(new_shape)

    result_torch = SpecFunc(matrix=(out_mat, mat_u), freq=new_freq, time=new_time)
    return result_torch.to_framework(fw_target, dev_target)


def interpolateToShape(spec: 'SpecFunc', target_shape: Tuple[int, int]) -> 'SpecFunc':
    old_freq, _ = spec.freq
    old_time, _ = spec.time
    
    fw_target, _ = _get_fw_info(old_freq)
    
    if fw_target == 'torch':
        import torch
        new_freq = torch.linspace(old_freq[0], old_freq[-1], target_shape[0], device=old_freq.device)
        new_time = torch.linspace(old_time[0], old_time[-1], target_shape[1], device=old_time.device)
    elif fw_target == 'tensorflow':
        import tensorflow as tf
        new_freq = tf.linspace(old_freq[0], old_freq[-1], target_shape[0])
        new_time = tf.linspace(old_time[0], old_time[-1], target_shape[1])
    else:
        import numpy as np
        new_freq = np.linspace(old_freq[0], old_freq[-1], target_shape[0])
        new_time = np.linspace(old_time[0], old_time[-1], target_shape[1])

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
        fw_target, dev_target = _get_fw_info(mat)
        
        if fw_target != 'torch':
            mat = convert_to_framework(mat, 'torch')
            
        import torch
        device = mat.device
        dtype = mat.real.dtype if mat.is_complex() else mat.dtype
        img_h, img_w = mat.shape[-2], mat.shape[-1]
        
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
        
        fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
        fft_kernel = torch.fft.fft2(padded_kernel, dim=(-2, -1))
        fft_result = fft_mat * fft_kernel
        ifft_result = torch.fft.ifft2(fft_result, dim=(-2, -1))

        res = ifft_result if mat.is_complex() else ifft_result.real
        
        if fw_target != 'torch':
            res = convert_to_framework(res, fw_target, dev_target)
        return res

    return spec.cloneApply(func=apply_gaussian_blur_fft, new_unit=current_unit)

import math

def _get_analytical_gaussian_filter(h: int, w: int, sigma: Tuple[float, float], 
                                    device: Any, dtype: Any, is_complex: bool) -> Any:
    """Аналитически генерирует частотный отклик Гауссова фильтра."""
    import torch
    
    # Частоты по оси X (ширина/время) - циклов на пиксель
    if not is_complex:
        u = torch.fft.rfftfreq(w, d=1.0, device=device) # Только положительные частоты
    else:
        u = torch.fft.fftfreq(w, d=1.0, device=device)  # Все частоты
        
    # Частоты по оси Y (высота/частота)
    v = torch.fft.fftfreq(h, d=1.0, device=device)
    
    # Формула Фурье-образа Гауссианы: H(v, u) = exp(-2 * pi^2 * (sigma_y^2 * v^2 + sigma_x^2 * u^2))
    v_part = (sigma[0] ** 2) * (v ** 2)
    u_part = (sigma[1] ** 2) * (u ** 2)
    
    v_part = v_part.unsqueeze(-1) # Форма (h, 1)
    u_part = u_part.unsqueeze(0)  # Форма (1, w_rfft)
    
    exponent = -2.0 * math.pi**2 * (v_part + u_part)
    return torch.exp(exponent).to(dtype)


def blurSpec(spec: 'SpecFunc', sigma: Union[float, Tuple[float, float]] = 1.0) -> 'SpecFunc':
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))
    _, current_unit = spec.values

    def apply_blur(mat: Any) -> Any:
        fw_target, dev_target = _get_fw_info(mat)
        if fw_target != 'torch': mat = convert_to_framework(mat, 'torch')
        import torch
        
        device = mat.device
        dtype = mat.real.dtype if mat.is_complex() else mat.dtype
        h, w = mat.shape[-2], mat.shape[-1]
        is_cplx = mat.is_complex()
        
        # Сразу получаем идеальный частотный фильтр (без FFT окна!)
        H_filter = _get_analytical_gaussian_filter(h, w, sigma, device, dtype, is_cplx)
        
        if is_cplx:
            fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
            res = torch.fft.ifft2(fft_mat * H_filter, dim=(-2, -1))
        else:
            fft_mat = torch.fft.rfft2(mat, dim=(-2, -1))
            res = torch.fft.irfft2(fft_mat * H_filter, s=(h, w), dim=(-2, -1))
            
        if fw_target != 'torch': res = convert_to_framework(res, fw_target, dev_target)
        return res
    return spec.cloneApply(func=apply_blur, new_unit=current_unit)


def highPassSpec(spec: 'SpecFunc', sigma: Union[float, Tuple[float, float]] = 1.0) -> 'SpecFunc':
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))
    _, current_unit = spec.values

    def apply_hpf(mat: Any) -> Any:
        fw_target, dev_target = _get_fw_info(mat)
        if fw_target != 'torch': mat = convert_to_framework(mat, 'torch')
        import torch
        
        device = mat.device
        dtype = mat.real.dtype if mat.is_complex() else mat.dtype
        h, w = mat.shape[-2], mat.shape[-1]
        is_cplx = mat.is_complex()
        
        H_filter = _get_analytical_gaussian_filter(h, w, sigma, device, dtype, is_cplx)
        
        if is_cplx:
            fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
            res = torch.fft.ifft2(fft_mat * (1.0 - H_filter), dim=(-2, -1))
        else:
            fft_mat = torch.fft.rfft2(mat, dim=(-2, -1))
            res = torch.fft.irfft2(fft_mat * (1.0 - H_filter), s=(h, w), dim=(-2, -1))
            
        if fw_target != 'torch': res = convert_to_framework(res, fw_target, dev_target)
        return res
    return spec.cloneApply(func=apply_hpf, new_unit=current_unit)


def sharpenSpec(spec: 'SpecFunc', sigma: Union[float, Tuple[float, float]] = 1.0, amount: float = 1.0) -> 'SpecFunc':
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))
    _, current_unit = spec.values

    def apply_sharpen(mat: Any) -> Any:
        fw_target, dev_target = _get_fw_info(mat)
        if fw_target != 'torch': mat = convert_to_framework(mat, 'torch')
        import torch
        
        device = mat.device
        dtype = mat.real.dtype if mat.is_complex() else mat.dtype
        h, w = mat.shape[-2], mat.shape[-1]
        is_cplx = mat.is_complex()
        
        H_filter = _get_analytical_gaussian_filter(h, w, sigma, device, dtype, is_cplx)
        H_sharpen = 1.0 + amount * (1.0 - H_filter)
        
        if is_cplx:
            fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
            res = torch.fft.ifft2(fft_mat * H_sharpen, dim=(-2, -1))
        else:
            fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
            res = torch.fft.ifft2(fft_mat * H_sharpen, s=(h, w), dim=(-2, -1))
            
        if fw_target != 'torch': res = convert_to_framework(res, fw_target, dev_target)
        return res
    return spec.cloneApply(func=apply_sharpen, new_unit=current_unit)

