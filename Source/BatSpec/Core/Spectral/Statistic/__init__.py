import numpy as np
import scipy.ndimage as ndimage
import warnings
from typing import Any, Tuple, Union

from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.Physical.Units import UREG
import MultiArray as ma
from MultiArray.Core import ArrayContext

try:
    from .FastZScore import fast_median_2d
    HAS_FAST_ZSCORE = True
except ImportError:
    HAS_FAST_ZSCORE = False

ArrayLike = Any

def noiseZNormByFreq(spec: SpecFunc, noise_percentile: float = 10.0) -> SpecFunc:
    ctx = spec.context
    time_func = spec.integrateOverFreq()
    energy_profile, _ = time_func.values

    energy_np = ma.to_numpy(energy_profile)
    threshold_val = np.percentile(energy_np, noise_percentile)
    
    threshold = ma.convert_to(np.array(threshold_val), ctx)
    noise_mask = energy_profile <= threshold

    if not np.any(ma.to_numpy(noise_mask)):
        warnings.warn("Не найдено шумовых срезов по процентилю.")
        min_energy_idx = ma.argmax(-energy_profile) 
        time_indices = ma.arange(energy_profile.shape[-1], ctx)
        noise_mask = (time_indices == min_energy_idx)

    def z_norm_func(mat: ArrayLike) -> ArrayLike:
        ctx_float = ArrayContext(ctx._framework, ctx._device, mat.dtype)
        mask_float = ma.convert_to(noise_mask, ctx_float)
        # Для корректной размерности broadcasting
        mask_float = ma.reshape(mask_float, (mask_float.shape[0], 1))
        
        noise_frame_count = ma.sum(mask_float)
        noise_frame_count = ma.clamp_min(noise_frame_count, 1.0)
        
        noise_only_mat = mat * mask_float
        noise_sum = ma.sum(noise_only_mat, axis=0, keepdims=True)
        noise_mean = noise_sum / noise_frame_count
        
        dev_sq = (mat - noise_mean)**2 * mask_float
        noise_var = ma.sum(dev_sq, axis=0, keepdims=True) / noise_frame_count
        noise_std = ma.sqrt(noise_var)
        
        safe_std = ma.clamp_min(noise_std, 1e-9)
        return (mat - noise_mean) / safe_std

    return spec.cloneApply(func=z_norm_func, new_unit=UREG.dimensionless)


def localNoiseZNorm(spec: SpecFunc, window_time: float = 0.05, window_freq: float = 0.0) -> SpecFunc:
    ctx = spec.context
    _, dt_val = spec.dt
    _, df_val = spec.df

    time_bins = max(1, int(round(window_time / float(dt_val))))
    freq_bins = max(1, int(round(window_freq / float(df_val))))

    if time_bins % 2 == 0: time_bins += 1
    if freq_bins % 2 == 0: freq_bins += 1

    def _cpu_moving_median(mat_np, k_h, k_w):
        if HAS_FAST_ZSCORE:
            mat_np_f32 = mat_np.astype(np.float32)
            ph, pw = k_h // 2, k_w // 2
            padded = np.pad(mat_np_f32, ((ph, ph), (pw, pw)), mode='symmetric')
            h, w = mat_np_f32.shape
            return fast_median_2d(padded, h, w, k_h, k_w)

        window_shape = [1] * mat_np.ndim
        window_shape[0] = k_h 
        window_shape[1] = k_w
        return ndimage.median_filter(mat_np, size=tuple(window_shape), mode='reflect')

    def local_z_norm_func(mat: ArrayLike) -> ArrayLike:
        mat_np = ma.to_numpy(mat)
        local_median = _cpu_moving_median(mat_np, time_bins, freq_bins) # (T, F)
        abs_dev = np.abs(mat_np - local_median)
        local_mad = _cpu_moving_median(abs_dev, time_bins, freq_bins)

        local_std = np.maximum(local_mad * 1.4826, 1e-9)
        z_normed_np = (mat_np - local_median) / local_std
        return ma.convert_to(z_normed_np, ctx)

    return spec.cloneApply(func=local_z_norm_func, new_unit=UREG.dimensionless)


def blurSpec(spec: SpecFunc, sigma: Union[float, Tuple[float, float]] = 1.0) -> SpecFunc:
    ctx = spec.context
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))
    sigma_t, sigma_f = sigma # (T, F)

    def blur_func(mat: ArrayLike) -> ArrayLike:
        if sigma_f <= 0 and sigma_t <= 0:
            return mat
        mat_np = ma.to_numpy(mat)
        blurred_np = ndimage.gaussian_filter(mat_np, sigma=(sigma_t, sigma_f), mode='reflect')
        return ma.convert_to(blurred_np, ctx)

    return spec.cloneApply(func=blur_func, new_unit=spec._matrx_u)

def makePiecewiseLog(spec: SpecFunc) -> SpecFunc:
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

def multiScaleLogProduct(spec: SpecFunc, sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)) -> SpecFunc:
    """Точное автоматизированное воспроизведение перемножения кусочных логарифмов."""
    res = makePiecewiseLog(spec)
    for sig in sigmas:
        blurred_spec = blurSpec(spec, sigma=sig)
        res = res * makePiecewiseLog(blurred_spec)
    return res

def medianDenoise(spec: SpecFunc, kernel_time: int = 3, kernel_freq: int = 3) -> SpecFunc:
    """Удаляет изолированный 'пиксельный' шум."""
    ctx = spec.context
    if kernel_time % 2 == 0: kernel_time += 1
    if kernel_freq % 2 == 0: kernel_freq += 1

    def process(mat: ArrayLike) -> ArrayLike:
        mat_np = ma.to_numpy(mat)
        if HAS_FAST_ZSCORE:
            mat_np_f32 = mat_np.astype(np.float32)
            ph, pw = kernel_time // 2, kernel_freq // 2
            padded = np.pad(mat_np_f32, ((ph, ph), (pw, pw)), mode='symmetric')
            h, w = mat_np_f32.shape
            res = fast_median_2d(padded, h, w, kernel_time, kernel_freq)
            return ma.convert_to(res, ctx)
        else:
            window_shape = [1] * mat_np.ndim
            window_shape[0] = kernel_time 
            window_shape[1] = kernel_freq
            res = ndimage.median_filter(mat_np, size=tuple(window_shape), mode='reflect')
            return ma.convert_to(res, ctx)

    return spec.cloneApply(process, spec._matrx_u)

def deadZoneFilter(spec: SpecFunc, threshold: float = 2.0) -> SpecFunc:
    """Подавляет шум около нуля. Значения за пределами плавно 'съезжают' к нулю."""
    def process(mat: ArrayLike) -> ArrayLike:
        ctx = ma.ArrayContext.from_array(mat)
        if ctx.isTorch(): sign_mat = ctx.fw.sign(mat)
        elif ctx.isTensorflow(): sign_mat = ctx.fw.math.sign(mat)
        else: sign_mat = ctx.fw.sign(mat)
        
        abs_mat = ma.abs(mat)
        shifted = ma.clamp_min(abs_mat - threshold, 0.0)
        return shifted * sign_mat

    return spec.cloneApply(process, spec._matrx_u)