from pathlib import Path
import numpy as np
from typing import cast


ScriptDir = Path(__file__).parent

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW

# --- Импорты для Legacy Функции ---
import tensorflow as tf
from BatSpec.Core.ConvWindow import Window, WindowNorm, WindowNormMismatchError
from BatSpec.Core.Physical.Units import UREG, unit_devide, unit_sqrt

# --- Основные импорты ---
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.Functions import SpecFunc, TimeFunc
from BatSpec.Core.Spectral import makeSpec, makeLogDB, DSPContext
from BatSpec.Visualize import update_spec2d, run_visualizer

# --- Импорты MultiArray ---
from MultiArray.Core import ArrayContext, Framework, DeviceType
import MultiArray as ma

# =====================================================================
# LEGACY ФУНКЦИЯ (Встроена для сравнения)
# =====================================================================

def makeSpec_legacy(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    """Амплитудная спектрограмма (|STFT| * scale) - СТАРАЯ ВЕРСИЯ НА TF."""
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)

    # Убеждаемся, что на входе NumPy массив
    signal_a, signal_unit = ma.to_numpy(signal.values[0]), signal.values[1]
    sr = signal.sr

    win = ma.to_numpy(window.get_array(sr, ArrayContext(Framework.NUMPY, DeviceType.CPU, None)))
    fft_length = (bins - 1) * 2
    win_length = len(win)
    hop = max(1, int(win_length * (1 - overlap)))

    def win_func(length: int, dtype: tf.DType = tf.float32):
        assert length == win_length
        return tf.convert_to_tensor(win, dtype=dtype)

    stft = cast(tf.Tensor, tf.signal.stft(
        signal_a.astype(np.float32),
        frame_length=win_length,
        frame_step=hop,
        fft_length=fft_length,
        window_fn=win_func,
    ))

    # tf.signal.stft возвращает (Time, Freq)
    amplitude_matrix_tf_shape = np.abs(stft.numpy())
    
    # SpecFunc ожидает (Freq, Time), поэтому ТРАНСПОНИРУЕМ
    amplitude_matrix = amplitude_matrix_tf_shape.T

    scale = np.sqrt(2 / sr)
    num_frames: int = amplitude_matrix.shape[1]
    freq_axis = np.fft.rfftfreq(fft_length, d=1.0 / sr)
    time_axis = (np.arange(num_frames) * hop) / sr

    spec_unit = unit_devide(signal_unit, unit_sqrt(UREG.Hz))

    return SpecFunc(
        matrix=(amplitude_matrix * scale, spec_unit),
        time=time_axis,
        freq=freq_axis,
    )

# =====================================================================
# ОСНОВНОЙ СКРИПТ
# =====================================================================

@run_visualizer
def main():
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    print(record.context)

    ctx_numpy_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    ctx_torch_gpu = ArrayContext(Framework.TENSORFLOW, DeviceType.GPU, None)
    
    window = TEST_HANN_WINODW
    overlap = 0.8
    bins = 300

    print("--- 1. Расчет спектрограммы новым методом (MultiArray) ---")
    with DSPContext(ctx_torch_gpu):
        spec_new = makeSpec(
            record,
            window=window,
            overlap=overlap,
            bins=bins,
        ).to_context(ctx_numpy_cpu)
    update_spec2d("1. MultiArray Spec", makeLogDB(spec_new, add_one=False))
    t, u = spec_new.values
    print(t.shape, u)
    t, u = spec_new.time
    print(t.shape, u)
    t, u = spec_new.freq
    print(t.shape, u)
    print(spec_new.df, spec_new.dt)
    print("--- 2. Расчет спектрограммы старым методом (Legacy TF) ---")
    spec_legacy = makeSpec_legacy(
        record,
        window=window,
        overlap=overlap,
        bins=bins,
    )
    update_spec2d("2. Legacy TF Spec", makeLogDB(spec_legacy, add_one=False))

    print("--- 3. Сравнение результатов ---")
    matrix_new, _ = spec_new.values
    matrix_legacy, _ = spec_legacy.values

    assert matrix_new.shape == matrix_legacy.shape, \
        f"Формы матриц не совпадают! New: {matrix_new.shape}, Legacy: {matrix_legacy.shape}"

    diff_matrix = np.abs(matrix_new - matrix_legacy)
    mae = np.mean(diff_matrix)
    print(f"Средняя абсолютная ошибка (MAE): {mae:.12f}")
    
    # Визуализируем разницу. Если всё черное, значит, результаты идентичны.
    diff_spec = SpecFunc(
        matrix=(diff_matrix, spec_new.values[1]),
        time=spec_new.time[0],
        freq=spec_new.freq[0]
    )

    update_spec2d("3. Difference (Abs)", diff_spec)