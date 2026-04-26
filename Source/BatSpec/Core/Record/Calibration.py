from jaxtyping import Float, ArrayLike

# --- 1. Импорты специфичных для проекта библиотек ---
from BatSpec.Core.Physical.Units import UREG, unit_mul, unit_devide
from BatSpec.Core.Functions import TimeFunc, FreqFunc, SpecFunc
from BatSpec.Core.Spectral import makeComplexSpec, inverseComplexSpec
from BatSpec.Core.ConvWindow import Window, TEST_HANN_WINODW

# --- 2. Внедряем наш новый универсальный фреймворк ---
import MultiArray as ma

# =====================================================================
# 1. Аналитические модели приборов (теперь с MultiArray)
# =====================================================================

class CalibrationModel:
    """Базовый класс для аналитической модели АЧХ прибора."""
    
    def __call__(self, freq_hz: Float[ArrayLike, 'freq']) -> Float[ArrayLike, 'freq']:
        raise NotImplementedError

    def generate_calibration_curve(self, freq_axis: Float[ArrayLike, 'freq']) -> FreqFunc:
        """Создает калибровочную кривую в том же фреймворке, что и входная ось частот."""
        coef_values = self(freq_axis)
        unit_pa_per_fs = unit_devide(UREG.Pa, UREG.FS)
        return FreqFunc(values=(coef_values, unit_pa_per_fs), axis=freq_axis)

        
class FlatResponseModel(CalibrationModel):
    def __init__(self, sensitivity_pa: float = 20.0):
        self.sensitivity_pa = sensitivity_pa

    def __call__(self, freq_hz: Float[ArrayLike, 'freq']) -> Float[ArrayLike, 'freq']:
        ctx = ma.ArrayContext.from_array(freq_hz)
        return ma.full(freq_hz.shape, self.sensitivity_pa, ctx)


class PetterssonM500Model(CalibrationModel):
    def __init__(self, base_pa_per_fs: float = 15.0, rolloff_start_hz: float = 20_000.0,
                 rolloff_rate: float = 5.0 / 10_000.0):
        self.base_pa_per_fs   = base_pa_per_fs
        self.rolloff_start_hz = rolloff_start_hz
        self.rolloff_rate     = rolloff_rate

    def __call__(self, freq_hz: Float[ArrayLike, 'freq']) -> Float[ArrayLike, 'freq']:
        boost = ma.clamp_min(freq_hz - self.rolloff_start_hz, 0.0) * self.rolloff_rate
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
        resonance  = self.resonance_depth_pa * ma.exp(exponent)
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
    Работает с NumPy, PyTorch, TensorFlow и другими фреймворками.
    """
    _, signal_u = signal_fs.values
    if not signal_u.is_compatible_with(UREG.FS):
        raise ValueError(f"Ожидался сигнал в единицах FS, получено: {signal_u}")

    complex_spec = makeComplexSpec(signal_fs, window=window, overlap=overlap, bins=bins)
    freq_axis, _ = complex_spec.freq
    calib_curve  = model.generate_calibration_curve(freq_axis)

    spec_matrix, spec_u = complex_spec.values  # Форма: (... freq time)
    coef_vector, coef_u = calib_curve.values   # Форма: (freq)

    # Добавляем измерение для broadcast'инга с помощью ma.reshape: (freq) -> (freq, 1)
    new_shape = coef_vector.shape + (1,)
    coef_vector_broadcasted = ma.reshape(coef_vector, new_shape)
    
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
