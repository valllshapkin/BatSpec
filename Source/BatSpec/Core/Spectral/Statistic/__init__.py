import numpy as np
import scipy.ndimage as ndimage
import warnings
from typing import Any, Tuple, Union

# --- 1. Импорты специфичных для проекта библиотек ---
from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.Physical.Units import UREG
# Внедряем наш новый универсальный фреймворк
import MultiArray as ma
from MultiArray import ArrayContext

# Пытаемся импортировать наш супербыстрый C++ модуль
try:
    from .FastZScore import fast_median_2d
    HAS_FAST_ZSCORE = True
except ImportError:
    HAS_FAST_ZSCORE = False

ArrayLike = Any

# =====================================================================
def noiseZNormByFreq(spec: SpecFunc, noise_percentile: float = 10.0) -> SpecFunc:
    ctx = spec.context
    time_func = spec.integrateOverFreq()
    energy_profile, _ = time_func.values

    energy_np = ma.to_numpy(energy_profile)
    threshold_val = np.percentile(energy_np, noise_percentile)
    
    threshold = ma.convert_to(np.array(threshold_val), ctx)
    noise_mask = energy_profile <= threshold

    if not np.any(ma.to_numpy(noise_mask)):
        warnings.warn("Не найдено шумовых срезов по процентилю. Используется срез с минимальной энергией.")
        min_energy_idx = ma.argmax(-energy_profile) 
        time_indices = ma.arange(energy_profile.shape[-1], ctx)
        noise_mask = (time_indices == min_energy_idx)

    def z_norm_func(mat: ArrayLike) -> ArrayLike:
        ctx_float = ArrayContext(ctx._framework, ctx._device, mat.dtype)
        mask_float = ma.convert_to(noise_mask, ctx_float)
        
        noise_frame_count = ma.sum(mask_float)
        noise_frame_count = ma.clamp_min(noise_frame_count, 1.0)
        
        noise_only_mat = mat * mask_float
        
        noise_sum = ma.sum(noise_only_mat, axis=-1, keepdims=True)
        noise_mean = noise_sum / noise_frame_count
        
        dev_sq = (mat - noise_mean)**2 * mask_float
        noise_var = ma.sum(dev_sq, axis=-1, keepdims=True) / noise_frame_count
        noise_std = ma.sqrt(noise_var)
        
        safe_std = ma.clamp_min(noise_std, 1e-9)
        return (mat - noise_mean) / safe_std

    return spec.cloneApply(func=z_norm_func, new_unit=UREG.dimensionless)


# =====================================================================
def localNoiseZNorm(spec: SpecFunc, window_time: float = 0.05, window_freq: float = 0.0) -> SpecFunc:
    ctx = spec.context
    dt_val, _ = spec.dt
    df_val, _ = spec.df

    time_bins = max(1, int(round(window_time / float(dt_val))))
    freq_bins = max(1, int(round(window_freq / float(df_val))))

    if time_bins % 2 == 0: time_bins += 1
    if freq_bins % 2 == 0: freq_bins += 1

    def _gpu_moving_median(tensor, k_h, k_w):
        import torch
        import torch.nn.functional as F
        
        pad_h, pad_w = k_h // 2, k_w // 2
        orig_shape = tensor.shape
        
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
            
        padded = F.pad(tensor.unsqueeze(1), (pad_w, pad_w, pad_h, pad_h), mode='reflect').squeeze(1)
        
        if k_h == 1:
            unfolded = padded.unfold(-1, k_w, 1)
            median = unfolded.median(dim=-1).values
            return median.view(orig_shape)
            
        H, W = tensor.shape[-2], tensor.shape[-1]
        res = torch.empty_like(tensor)
        
        bytes_per_element = 4
        chunk_size = max(1, (100 * 1024 * 1024) // (H * k_h * k_w * bytes_per_element))
        
        for w_start in range(0, W, chunk_size):
            w_end = min(W, w_start + chunk_size)
            padded_chunk = padded[..., :, w_start : w_end + 2 * pad_w]
            
            unfolded = padded_chunk.unfold(-2, k_h, 1).unfold(-1, k_w, 1)
            unfolded = unfolded.contiguous().view(*unfolded.shape[:-2], -1)
            
            res[..., :, w_start:w_end] = unfolded.median(dim=-1).values
            
        return res.view(orig_shape)

    def _cpu_moving_median(mat_np, k_h, k_w):
        if HAS_FAST_ZSCORE:
            mat_np_f32 = mat_np.astype(np.float32)
            ph, pw = k_h // 2, k_w // 2
            
            if mat_np_f32.ndim == 2:
                padded = np.pad(mat_np_f32, ((ph, ph), (pw, pw)), mode='symmetric')
                h, w = mat_np_f32.shape
                return fast_median_2d(padded, h, w, k_h, k_w)
                
            elif mat_np_f32.ndim == 3: 
                res = np.empty_like(mat_np_f32)
                for b in range(mat_np_f32.shape[0]):
                    padded = np.pad(mat_np_f32[b], ((ph, ph), (pw, pw)), mode='symmetric')
                    h, w = mat_np_f32[b].shape
                    res[b] = fast_median_2d(padded, h, w, k_h, k_w)
                return res

        if k_h == 1:
            try:
                import pandas as pd
                df = pd.DataFrame(mat_np.T)
                return df.rolling(k_w, center=True, min_periods=1).median().values.T
            except ImportError:
                pass
                
        window_shape = [1] * mat_np.ndim
        window_shape[-2] = k_h 
        window_shape[-1] = k_w
        return ndimage.median_filter(mat_np, size=tuple(window_shape), mode='reflect')

    def local_z_norm_func(mat: ArrayLike) -> ArrayLike:
        if ctx.isTorch():
            import torch
            local_median = _gpu_moving_median(mat, freq_bins, time_bins)
            abs_dev = torch.abs(mat - local_median)
            local_mad = _gpu_moving_median(abs_dev, freq_bins, time_bins)
            
            local_std = torch.clamp(local_mad * 1.4826, min=1e-9)
            return (mat - local_median) / local_std
        else:
            mat_np = ma.to_numpy(mat)
            local_median = _cpu_moving_median(mat_np, freq_bins, time_bins)
            abs_dev = np.abs(mat_np - local_median)
            local_mad = _cpu_moving_median(abs_dev, freq_bins, time_bins)

            local_std = np.maximum(local_mad * 1.4826, 1e-9)
            z_normed_np = (mat_np - local_median) / local_std
            return ma.convert_to(z_normed_np, ctx)

    return spec.cloneApply(func=local_z_norm_func, new_unit=UREG.dimensionless)


# =====================================================================
def blurSpec(spec: SpecFunc, sigma: Union[float, Tuple[float, float]] = 1.0) -> SpecFunc:
    ctx = spec.context
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))
    sigma_f, sigma_t = sigma

    def blur_func(mat: ArrayLike) -> ArrayLike:
        if sigma_f <= 0 and sigma_t <= 0:
            return mat

        if ctx.isTorch():
            import torch
            import torch.nn.functional as F
            
            orig_shape = mat.shape
            tensor = mat.unsqueeze(0).unsqueeze(0) if mat.ndim == 2 else mat.unsqueeze(1)
            
            def get_1d_kernel(s):
                k = int(round(s * 3)) * 2 + 1
                if k < 3: k = 3
                x = torch.arange(k, device=tensor.device, dtype=tensor.dtype) - k // 2
                kernel = torch.exp(-0.5 * (x / s) ** 2)
                return kernel / kernel.sum()
                
            if sigma_f > 0:
                k_h = get_1d_kernel(sigma_f).view(1, 1, -1, 1)
                pad_h = k_h.shape[2] // 2
                pad_mode = 'reflect' if pad_h < tensor.shape[-2] else 'replicate'
                tensor = F.pad(tensor, (0, 0, pad_h, pad_h), mode=pad_mode)
                tensor = F.conv2d(tensor, k_h)
                
            if sigma_t > 0:
                k_w = get_1d_kernel(sigma_t).view(1, 1, 1, -1)
                pad_w = k_w.shape[3] // 2
                pad_mode = 'reflect' if pad_w < tensor.shape[-1] else 'replicate'
                tensor = F.pad(tensor, (pad_w, pad_w, 0, 0), mode=pad_mode)
                tensor = F.conv2d(tensor, k_w)
                
            return tensor.view(orig_shape)
        else:
            mat_np = ma.to_numpy(mat)
            blurred_np = ndimage.gaussian_filter(mat_np, sigma=(sigma_f, sigma_t), mode='reflect')
            return ma.convert_to(blurred_np, ctx)

    _, current_unit = spec.values
    return spec.cloneApply(func=blur_func, new_unit=current_unit)


def multiScaleGeometricMean(
    spec: SpecFunc, 
    sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
) -> SpecFunc:
    """
    МУЛЬТИМАСШТАБНОЕ УМНОЖЕНИЕ (Geometric Mean) с кастомными весами.
    Первый вес относится к оригинальному сигналу, остальные — к соответствующим размытиям (sigmas).
    Веса автоматически нормируются так, чтобы их сумма равнялась 1.0.
    
    Сохраняет математику уничтожения шума и решает проблему минусов через abs().
    Возвращает оригинальный знак в самом конце.
    """
    if len(weights) != len(sigmas) + 1:
        raise ValueError(f"Количество весов ({len(weights)}) должно быть на 1 больше количества sigmas ({len(sigmas)}).")

    ctx = spec.context
    _, current_unit = spec.values
    
    # Нормируем веса (чтобы в сумме они давали 1.0)
    total_w = sum(weights)
    if total_w == 0:
        raise ValueError("Сумма весов не может быть равна нулю.")
    norm_weights = [float(w) / total_w for w in weights]

    def process(mat: ArrayLike) -> ArrayLike:
        # Сохраняем оригинальные знаки (чтобы шум продолжил колебаться вокруг 0, а не ушел вверх)
        if ctx.isTorch(): sign_mat = ctx.fw.sign(mat)
        elif ctx.isTensorflow(): sign_mat = ctx.fw.math.sign(mat)
        else: sign_mat = ctx.fw.sign(mat)

        # 1. Стартуем с МОДУЛЯ оригинального сигнала
        res = (ma.abs(mat) + 1e-9) ** norm_weights[0]
        
        # 2. Итерируемся по сигмам
        for i, sig in enumerate(sigmas):
            w = norm_weights[i + 1]
            
            # Оптимизация: если вес нулевой, пропускаем расчет размытия для этой сигмы
            if w <= 0:
                continue
                
            # РАЗМЫВАЕМ ОРИГИНАЛЬНЫЙ SPEC (с минусами, чтобы они взаимно уничтожали друг друга)
            blurred_spec = blurSpec(spec, sigma=sig)
            blurred_mat, _ = blurred_spec.values
            
            # Умножаем на МОДУЛЬ размытого сигнала в заданной нормированной степени
            res = res * ((ma.abs(blurred_mat) + 1e-9) ** w)
            
        # 3. Возвращаем оригинальный знак!
        return res * sign_mat

    return spec.cloneApply(process, current_unit)

def multiScaleLogProduct(spec: SpecFunc, sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)) -> SpecFunc:
    """
    Точное автоматизированное воспроизведение перемножения кусочных логарифмов.
    Работает как мощный нелинейный гейт:
    - Если хотя бы на одном слое (сигме) пустота (0), то всё умножается на 0 (шум исчезает).
    - Если на всех слоях есть сигнал, значения перемножаются и контраст взлетает в космос.
    """
    # 1. Начинаем с логарифмированного оригинала
    res = makePiecewiseLog(spec)
    
    # 2. В цикле размываем ОРИГИНАЛ, логарифмируем и перемножаем (как ты и писал)
    for sig in sigmas:
        blurred_spec = blurSpec(spec, sigma=sig)
        res = res * makePiecewiseLog(blurred_spec)
        
    return res

# =====================================================================
def makePiecewiseLog(spec: SpecFunc) -> SpecFunc:
    """
    Symmetric Logarithm: Y = sign(X) * log(1 + abs(X)).
    Идеально сжимает пики как в положительную, так и в отрицательную сторону без отсечений!
    """
    def piecewise_log_func(arr: Any) -> Any:
        ctx = ma.ArrayContext.from_array(arr)
        
        abs_arr = ma.abs(arr)
        if ctx.isTorch():
            return ctx.fw.sign(arr) * ctx.fw.log1p(abs_arr)
        elif ctx.isTensorflow():
            return ctx.fw.math.sign(arr) * ctx.fw.math.log1p(abs_arr)
        else:
            return ctx.fw.sign(arr) * ctx.fw.log1p(abs_arr)

    return spec.cloneApply(func=piecewise_log_func, new_unit=UREG.dimensionless)

def medianDenoise(spec: SpecFunc, kernel_time: int = 3, kernel_freq: int = 3) -> SpecFunc:
    """
    Удаляет изолированный 'пиксельный' шум, не размывая края полезного сигнала.
    """
    _, current_unit = spec.values
    ctx = spec.context

    # Убеждаемся, что ядро нечетное
    if kernel_time % 2 == 0: kernel_time += 1
    if kernel_freq % 2 == 0: kernel_freq += 1

    def process(mat: ArrayLike) -> ArrayLike:
        if ctx.isNumpy() and HAS_FAST_ZSCORE:
            # Используем твой C++ модуль!
            mat_np_f32 = ma.to_numpy(mat).astype(np.float32)
            ph, pw = kernel_freq // 2, kernel_time // 2
            
            if mat_np_f32.ndim == 2:
                padded = np.pad(mat_np_f32, ((ph, ph), (pw, pw)), mode='symmetric')
                h, w = mat_np_f32.shape
                res = fast_median_2d(padded, h, w, kernel_freq, kernel_time)
            elif mat_np_f32.ndim == 3: 
                res = np.empty_like(mat_np_f32)
                for b in range(mat_np_f32.shape[0]):
                    padded = np.pad(mat_np_f32[b], ((ph, ph), (pw, pw)), mode='symmetric')
                    h, w = mat_np_f32[b].shape
                    res[b] = fast_median_2d(padded, h, w, kernel_freq, kernel_time)
                    
            return ma.convert_to(res, ctx)
        else:
            # Фолбэк для Torch/TF/обычного numpy
            import scipy.ndimage as ndimage
            mat_np = ma.to_numpy(mat)
            window_shape = [1] * mat_np.ndim
            window_shape[-2] = kernel_freq 
            window_shape[-1] = kernel_time
            res = ndimage.median_filter(mat_np, size=tuple(window_shape), mode='reflect')
            return ma.convert_to(res, ctx)

    return spec.cloneApply(process, current_unit)

def deadZoneFilter(spec: SpecFunc, threshold: float = 2.0) -> SpecFunc:
    """
    Подавляет шум около нуля. Все значения от -threshold до +threshold становятся нулями.
    Значения за пределами плавно 'съезжают' к нулю, сохраняя свой знак.
    """
    _, current_unit = spec.values

    def process(mat: ArrayLike) -> ArrayLike:
        ctx = ma.ArrayContext.from_array(mat)
        
        # Получаем знак (+1 или -1)
        if ctx.isTorch(): sign_mat = ctx.fw.sign(mat)
        elif ctx.isTensorflow(): sign_mat = ctx.fw.math.sign(mat)
        else: sign_mat = ctx.fw.sign(mat)
        
        # Берем модуль, вычитаем порог и отсекаем то, что ушло в минус (это и был шум)
        abs_mat = ma.abs(mat)
        shifted = ma.clamp_min(abs_mat - threshold, 0.0)
        
        # Возвращаем оригинальный знак
        return shifted * sign_mat

    return spec.cloneApply(process, current_unit)