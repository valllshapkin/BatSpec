from typing import Optional, Any, Tuple, Union
import warnings

# Предполагается, что эти функции лежат рядом
# from .helpers import _get_fw_info  <-- Ваш хелпер из первого блока
from BatSpec.Core.Functions import TimeFunc, SpecFunc
from BatSpec.Core.Physical.Units import UREG, unit_devide
from BatSpec.Core.Physical.Arrays import convert_to_framework

def _get_fw_info(tensor: Any) -> Tuple[str, Any]:
    """Определяет фреймворк и устройство тензора."""
    type_str = str(type(tensor)).lower()
    dev = getattr(tensor, 'device', None)
    if 'torch' in type_str: return 'torch', dev
    if 'tensorflow' in type_str: return 'tensorflow', dev
    return 'numpy', dev

# =====================================================================
def noiseZNormByFreq(spec: SpecFunc, noise_percentile: float = 10.0) -> SpecFunc:
    """
    Адаптивная Z-нормировка по частотным полосам на основе профиля шума.
    КРОСС-ФРЕЙМВОРКОВАЯ ВЕРСИЯ (через PyTorch Bridge).
    """
    # 1. Запоминаем исходный фреймворк и устройство
    fw_target, dev_target = _get_fw_info(spec.values[0])
    
    # 2. Временно переводим в PyTorch для сложной математики
    spec_torch = spec.to_framework('torch')
    import torch
    
    # Интегрируем спектрограмму по частоте
    time_func = spec_torch.integrateOverFreq()
    energy_profile, _ = time_func.values

    # Находим порог шума
    q = torch.tensor(noise_percentile / 100.0, device=energy_profile.device, dtype=energy_profile.dtype)
    threshold = torch.quantile(energy_profile, q)
    noise_mask = energy_profile <= threshold

    if not torch.any(noise_mask):
        noise_mask[torch.argmin(energy_profile)] = True

    def z_norm_func(mat: torch.Tensor) -> torch.Tensor:
        noise_mat = mat[..., :, noise_mask]
        noise_mean = torch.mean(noise_mat, dim=-1, keepdim=True)
        noise_std = torch.std(noise_mat, dim=-1, keepdim=True)

        safe_std = torch.where(noise_std < 1e-9, torch.tensor(1e-9, device=mat.device, dtype=mat.dtype), noise_std)
        return (mat - noise_mean) / safe_std

    # Применяем нормировку в пространстве Torch
    result_torch = spec_torch.cloneApply(func=z_norm_func, new_unit=UREG.dimensionless)
    
    # 3. Возвращаем в исходный NumPy/TensorFlow/PyTorch
    return result_torch.to_framework(fw_target, dev_target)


# =====================================================================
def resampleTimeFunc(time_func: TimeFunc, new_sr: int) -> TimeFunc:
    """
    Ресемплирует временной сигнал. КРОСС-ФРЕЙМВОРКОВАЯ ВЕРСИЯ.
    """
    fw_target, dev_target = _get_fw_info(time_func.values[0])
    tf_torch = time_func.to_framework('torch')
    
    import torch
    import torchaudio.transforms as T
    
    signal_a, signal_u = tf_torch.values
    dt_val, _ = tf_torch.dt
    orig_sr = int(round(1.0 / float(dt_val)))

    if orig_sr == new_sr:
        return time_func

    resampler = T.Resample(orig_sr, new_sr, dtype=signal_a.dtype).to(signal_a.device)
    resampled_a = resampler(signal_a)

    num_samples = resampled_a.shape[-1]
    new_time_axis = torch.arange(num_samples, dtype=torch.float64, device=signal_a.device) / new_sr
    
    res_torch = TimeFunc(
        values=(resampled_a, signal_u),
        axis=new_time_axis
    )
    
    return res_torch.to_framework(fw_target, dev_target)


# =====================================================================
def integrateAndClipEnvelope(
    spec: SpecFunc, 
    threshold: float = 5.0, 
    resample_sr: Optional[int] = None,
    make_zero: bool = True
) -> TimeFunc:
    """
    Интегрирует спектрограмму, нормализует её и отсекает шум по порогу.
    КРОСС-ФРЕЙМВОРКОВАЯ ВЕРСИЯ.
    """
    fw_target, dev_target = _get_fw_info(spec.values[0])
    spec_torch = spec.to_framework('torch')
    import torch
    
    time_func = spec_torch.integrateOverFreq()
    
    freq_axis, freq_unit = spec_torch.freq
    num_freq_bins = freq_axis.shape[-1]
    freq_range = freq_axis[-1] - freq_axis[0]
    
    if freq_range <= 0:
        freq_range = torch.tensor(1e-9, device=freq_axis.device, dtype=freq_axis.dtype)

    val_a, val_u = time_func.values
    
    # Масштабируем
    norm_factor = torch.sqrt(torch.tensor(float(num_freq_bins), device=val_a.device, dtype=val_a.dtype)) * freq_range
    scaled_values = val_a / norm_factor
    new_unit = unit_devide(val_u, freq_unit)

    time_axis_arr, _ = time_func.time  # Исправлено с time_func.axis[0]

    envelope_torch = TimeFunc(
        values=(scaled_values, new_unit),
        axis=time_axis_arr
    )

    # Ресемплинг
    if resample_sr is not None:
        envelope_torch = resampleTimeFunc(envelope_torch, resample_sr)
        # Убеждаемся, что мы все еще в Torch (resample возвращает исходный FW, если не принудить)
        envelope_torch = envelope_torch.to_framework('torch')

    final_values, final_unit = envelope_torch.values
    
    # Клиппинг
    sigma = torch.std(final_values)
    limit = threshold * sigma
    
    clipped_values = torch.clamp(final_values, min=limit)
    
    if make_zero:
        clipped_values = clipped_values - limit
    
    final_time_axis, _ = envelope_torch.time
    res_torch = TimeFunc(
        values=(clipped_values, final_unit),
        axis=final_time_axis
    )
    
    return res_torch.to_framework(fw_target, dev_target)