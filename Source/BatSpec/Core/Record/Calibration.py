from typing import Any
from jaxtyping import Float

# --- Межфреймворковые хелперы ---

def _array_full_like(x: Any, fill_value: float) -> Any:
    """Создает массив/тензор формы как у x, заполненный значением fill_value."""
    type_str = str(type(x)).lower()
    if 'torch' in type_str:
        import torch
        return torch.full_like(x, fill_value)
    elif 'tensorflow' in type_str:
        import tensorflow as tf
        return tf.fill(tf.shape(x), fill_value)
    else:
        import numpy as np
        return np.full_like(x, fill_value)

def _array_clamp_min(x: Any, min_val: float) -> Any:
    """Ограничивает значения в массиве/тензоре снизу."""
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
        
def _array_exp(x: Any) -> Any:
    """Вычисляет экспоненту для каждого элемента."""
    type_str = str(type(x)).lower()
    if 'torch' in type_str:
        import torch
        return torch.exp(x)
    elif 'tensorflow' in type_str:
        import tensorflow as tf
        return tf.exp(x)
    else:
        import numpy as np
        return np.exp(x)

def _array_expand_dims(x: Any, axis: int) -> Any:
    """Добавляет новое измерение в массив/тензор."""
    type_str = str(type(x)).lower()
    if 'torch' in type_str:
        return x.unsqueeze(axis)
    elif 'tensorflow' in type_str:
        import tensorflow as tf
        return tf.expand_dims(x, axis)
    else:
        import numpy as np
        return np.expand_dims(x, axis)


# --- Основной код ---

ArrayLike = Any

# Подставьте ваши актуальные пути импортов
from BatSpec.Core.Physical.Units import UREG, unit_mul, unit_devide
from BatSpec.Core.Functions import TimeFunc, FreqFunc, SpecFunc
from BatSpec.Core.Spectral import makeComplexSpec, inverseComplexSpec
from BatSpec.Core.ConvWindow import Window, TEST_HANN_WINODW

# =====================================================================
# 1. Аналитические модели приборов (теперь фреймворк-агностичные)
# =====================================================================

class CalibrationModel:
    """Базовый класс для аналитической модели АЧХ прибора."""
    
    def __call__(self, freq_hz: Float[ArrayLike, 'freq']) -> Float[ArrayLike, 'freq']:
        raise NotImplementedError

    def generate_calibration_curve(self, freq_axis: Float[ArrayLike, 'freq']) -> FreqFunc:
        coef_values = self(freq_axis)
        unit_pa_per_fs = unit_devide(UREG.Pa, UREG.FS)
        return FreqFunc(values=(coef_values, unit_pa_per_fs), axis=freq_axis)

        
class FlatResponseModel(CalibrationModel):
    def __init__(self, sensitivity_pa: float = 20.0):
        self.sensitivity_pa = sensitivity_pa

    def __call__(self, freq_hz: Float[ArrayLike, 'freq']) -> Float[ArrayLike, 'freq']:
        return _array_full_like(freq_hz, self.sensitivity_pa)


class PetterssonM500Model(CalibrationModel):
    def __init__(self, base_pa_per_fs: float = 15.0, rolloff_start_hz: float = 20_000.0,
                 rolloff_rate: float = 5.0 / 10_000.0):
        self.base_pa_per_fs   = base_pa_per_fs
        self.rolloff_start_hz = rolloff_start_hz
        self.rolloff_rate     = rolloff_rate

    def __call__(self, freq_hz: Float[ArrayLike, 'freq']) -> Float[ArrayLike, 'freq']:
        boost = _array_clamp_min(freq_hz - self.rolloff_start_hz, 0.0) * self.rolloff_rate
        return self.base_pa_per_fs + boost


class ResonanceMicModel(CalibrationModel):
    def __init__(self, base_pa: float = 25.0, resonance_hz: float = 40_000.0,
                 resonance_width_hz: float = 5_000.0, resonance_depth_pa: float = 15.0):
        self.base_pa            = base_pa
        self.resonance_hz       = resonance_hz
        self.resonance_width_hz = resonance_width_hz
        self.resonance_depth_pa = resonance_depth_pa

    def __call__(self, freq_hz: Float[ArrayLike, 'freq']) -> Float[ArrayLike, 'freq']:
        exponent   = -0.5 * ((freq_hz - self.resonance_hz) / self.resonance_width_hz) ** 2
        resonance  = self.resonance_depth_pa * _array_exp(exponent)
        return self.base_pa - resonance


# =====================================================================
# 2. Главная функция: TimeFunc(FS) -> TimeFunc(Pa)
# =====================================================================

def applyСalibration(signal_fs: TimeFunc, 
                     model: CalibrationModel, 
                     window: Window = TEST_HANN_WINODW, 
                     overlap: float = 0.5, 
                     bins: int = 300) -> TimeFunc:
    """
    Переводит сырой цифровой сигнал (FS) в физические Паскали (Pa),
    учитывая частотно-зависимую калибровку (АЧХ) прибора.
    Работает с NumPy, PyTorch, TensorFlow.
    """
    _, signal_u = signal_fs.values
    if not signal_u.is_compatible_with(UREG.FS):
        raise ValueError(f"Ожидался сигнал в единицах FS, получено: {signal_u}")

    complex_spec = makeComplexSpec(signal_fs, window=window, overlap=overlap, bins=bins)
    freq_axis, _ = complex_spec.freq
    calib_curve  = model.generate_calibration_curve(freq_axis)

    spec_matrix, spec_u = complex_spec.values  # Форма: (... freq time)
    coef_vector, coef_u = calib_curve.values   # Форма: (freq)

    # Применяем универсальный хелпер для добавления измерения
    coef_vector_broadcasted = _array_expand_dims(coef_vector, axis=-1)
    
    # Broadcasting (умножение (... freq time) на (freq, 1)) работает одинаково во всех фреймворках
    calibrated_matrix = spec_matrix * coef_vector_broadcasted
    new_spec_unit = unit_mul(spec_u, coef_u)

    time_axis, _ = complex_spec.time
    calibrated_spec = SpecFunc(
        matrix=(calibrated_matrix, new_spec_unit),
        freq=freq_axis,
        time=time_axis,
    )

    return inverseComplexSpec(calibrated_spec, window, overlap=overlap)