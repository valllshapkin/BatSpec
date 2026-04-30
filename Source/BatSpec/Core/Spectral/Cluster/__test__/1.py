from pathlib import Path
import numpy as np
import scipy.ndimage
from typing import Union, Tuple

ScriptDir = Path(__file__).parent

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW, TEST_BLHA_WINODW
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.Spectral import makeLogDB, DSPContext, makeRobustSpec
from BatSpec.Core.Spectral.Statistic import noiseZNormByFreq
from BatSpec.Core.Functions import SpecFunc, TimeFunc
from BatSpec.Core.Physical.Units import UREG
from BatSpec.Visualize import update_spec2d, run_visualizer, update_function

# --- Импорты MultiArray ---
from MultiArray.Core import ArrayContext, Framework, DeviceType
import MultiArray as ma

# =====================================================================
# ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# =====================================================================

def getFmaxE(spec: SpecFunc, sigma: Union[float, Tuple[float, float]] = 3.0, norm: bool = True, noise_threshold: float = 1.0) -> TimeFunc:
    """
    Находит частоту максимальной энергии (Peak Frequency).
    
    :param sigma: Сигма для Гауссова фильтра.
                  - float: сглаживание только по оси частот (1D).
                  - tuple (sigma_freq, sigma_time): сглаживание по обеим осям (2D).
    :param noise_threshold: Порог амплитуды. Если амплитуда под пиком меньше порога,
                            частота заменяется на NaN.
    """
    ctx = spec.context
    mat, _ = spec.values
    freq_axis, _ = spec.freq
    
    mat_np = ma.to_numpy(mat).astype(np.float32)
    freq_axis_np = ma.to_numpy(freq_axis).astype(np.float32)
    
    # 1. Сглаживание (1D или 2D)
    if isinstance(sigma, (float, int)):
        sigma_f, sigma_t = float(sigma), 0.0
    else:
        sigma_f, sigma_t = sigma

    if sigma_f > 0 or sigma_t > 0:
        # Создаем вектор сигм, который работает с батчами (первые оси не сглаживаем)
        full_sigma = [0] * mat_np.ndim
        full_sigma[-2] = sigma_f  # Ось частот
        full_sigma[-1] = sigma_t  # Ось времени
        blurred_mat = scipy.ndimage.gaussian_filter(mat_np, sigma=full_sigma)
    else:
        blurred_mat = mat_np
        
    # 2. Находим индекс максимума
    max_idx_np = np.argmax(blurred_mat, axis=-2)
    
    # 3. Мапим индексы на значения реальных частот
    peak_freqs_np = freq_axis_np[max_idx_np]
    
    # 4. Нормализация частоты к [0, 1]
    if norm:
        peak_freqs_np = peak_freqs_np / freq_axis_np[-1]
        
    # 5. Жесткий порог для удаления шума (Thresholding)
    if noise_threshold is not None:
        expanded_idx = np.expand_dims(max_idx_np, axis=-2)
        amp_at_peaks = np.take_along_axis(mat_np, expanded_idx, axis=-2).squeeze(-2)
        mask = amp_at_peaks < noise_threshold
        peak_freqs_np[mask] = np.nan
        
    # Возвращаем TimeFunc в исходном контексте
    return TimeFunc(
        values=(ma.convert_to(peak_freqs_np, ctx), UREG.dimensionless),
        axis=spec.time[0]
    )

# =====================================================================
# ОСНОВНОЙ СКРИПТ
# =====================================================================

@run_visualizer
def main():
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))

    ctx_numpy_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    ctx_torch_gpu = ArrayContext(Framework.TORCH, DeviceType.GPU, None)
    
    print("--- Основные вычисления ---")
    with DSPContext(ctx_torch_gpu):
        spec_robust = makeRobustSpec(
            record,
            window=(TEST_HANN_WINODW, TEST_BLHA_WINODW),
            overlap=0.8,
            bins=300,
            shifts=(-1, 1)  
        ).to_context(ctx_numpy_cpu)
        
    update_spec2d("Robust Spectrogram", makeLogDB(spec_robust, add_one=False))

    zspec = noiseZNormByFreq(spec_robust)
    update_spec2d("Z-Norm Spectrogram", makeLogDB(zspec, add_one=True))
    
    # ---------------------------------------------------------
    # 1. Извлекаем мощность сигнала (интеграл по частоте)
    # ---------------------------------------------------------
    power_func = zspec.integrateOverFreq()
    p_mat, _ = power_func.values
    
    p_np = ma.to_numpy(p_mat)
    p_min = p_np.min(axis=-1, keepdims=True)
    p_max = p_np.max(axis=-1, keepdims=True)
    p_norm_np = (p_np - p_min) / (p_max - p_min + 1e-9)
    
    power_norm = TimeFunc(
        values=(ma.convert_to(p_norm_np, ctx_numpy_cpu), UREG.dimensionless),
        axis=power_func.time[0]
    )
    update_function("Power (Norm 0-1)", power_norm)

    # ---------------------------------------------------------
    # 2. Извлекаем робастную частоту максимальной энергии
    # ---------------------------------------------------------
    
    # Вариант 1: Без сглаживания (шумный, рваный)
    peak_raw = getFmaxE(zspec, sigma=0.0, norm=True, noise_threshold=8.0)
    update_function("Peak Freq (Raw)", peak_raw)

    # Вариант 2: Сглаживание только по частоте (убирает выбросы)
    peak_1d_smooth = getFmaxE(zspec, sigma=4.0, norm=True, noise_threshold=8.0)
    update_function("Peak Freq (1D Smooth)", peak_1d_smooth)
    
    # Вариант 3: Сглаживание по частоте и времени (самый гладкий результат)
    peak_2d_smooth = getFmaxE(zspec, sigma=(4.0, 2.0), norm=True, noise_threshold=8.0)
    update_function("Peak Freq (2D Smooth)", peak_2d_smooth)