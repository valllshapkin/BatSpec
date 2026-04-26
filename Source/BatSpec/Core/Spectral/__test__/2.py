from pathlib import Path
import time
import numpy as np

ScriptDir = Path(__file__).parent

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW, TEST_BLHA_WINODW
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.SaveIntegral import SaveIntegral
from BatSpec.Core.Spectral import makeSpec, makeLogDB, DSPContext
from BatSpec.Core.Functions import TimeFunc, SpecFunc
from BatSpec.Core.Physical.Units import UREG
from BatSpec.Visualize import update_spec2d, run_visualizer

# --- Импорты MultiArray ---
import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType
def makeRobustSpec(
    signal: TimeFunc,
    window,  # одно окно или тюпл окон
    overlap: float = 0.5,
    bins: int = 300,
    shifts: int | tuple[int, ...] = 4
) -> SpecFunc:
    """
    Вычисляет спектрограмму, устойчивую к сингулярностям.

    window: одно окно или тюпл окон — берётся максимум по всем окнам
    shifts: int   — симметричный диапазон ±shifts
            tuple — явный список сдвигов, например (-4, 0, 4)
    """
    windows = window if isinstance(window, tuple) else (window,)

    if isinstance(shifts, int):
        shift_list = [s for s in range(-shifts, shifts + 1) if s != 0]
    else:
        shift_list = [s for s in shifts if s != 0]

    sig_a, sig_u = signal.values

    def apply_shift(shift: int) -> TimeFunc:
        shape_pad = list(sig_a.shape)
        shape_pad[-1] = abs(shift)
        pad = ma.zeros(tuple(shape_pad), ctx)
        if shift > 0:
            shifted_a = ma.concatenate([pad, sig_a[..., :-shift]], axis=-1)
        else:
            shifted_a = ma.concatenate([sig_a[..., -shift:], pad], axis=-1)
        return TimeFunc(values=(shifted_a, sig_u), axis=signal.time[0])

    def elem_max(a, b):
        if ctx.isTorch():
            return ctx.fw.maximum(a, b)
        elif ctx.isTensorflow():
            import tensorflow as tf
            return tf.maximum(a, b)
        else:
            return np.maximum(a, b)

    # Первое окно + нулевой сдвиг — база для mat_max и ctx
    base_spec = makeSpec(signal, windows[0], overlap, bins)
    mat_max, unit = base_spec.values
    ctx = base_spec.context

    for win in windows:
        # Нулевой сдвиг для каждого окна (кроме уже посчитанного windows[0])
        if win is not windows[0]:
            s, _ = makeSpec(signal, win, overlap, bins).values
            mat_max = elem_max(mat_max, s)

        for shift in shift_list:
            s, _ = makeSpec(apply_shift(shift), win, overlap, bins).values
            mat_max = elem_max(mat_max, s)

    return SpecFunc(matrix=(mat_max, unit), freq=base_spec.freq[0], time=base_spec.time[0])


@run_visualizer
def main():
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    
    ctx_torch_gpu = ArrayContext(Framework.TENSORFLOW, DeviceType.GPU, None)
    
    print("--- Основные вычисления ---")
    
    # 1. Оригинальная (классическая) спектрограмма
    t0 = time.time()
    with DSPContext(ctx_torch_gpu):
        spec_standard = makeSpec(record, TEST_HANN_WINODW, overlap=0.8, bins=300)
    print(f"Standard STFT: {time.time() - t0:.3f} сек")
    update_spec2d("1. Standard (С дырками)", makeLogDB(spec_standard, add_one=False))

    # 2. Робастная спектрограмма (Jitter Ensemble)
    t0 = time.time()
    with DSPContext(ctx_torch_gpu):
        spec_robust = makeRobustSpec(
            record,
            window=(TEST_HANN_WINODW, TEST_BLHA_WINODW),
            overlap=0.8,
            bins=300,
            shifts=(-1, 1)  
        )
    print(f"Robust STFT (x6 computations): {time.time() - t0:.3f} сек")
    update_spec2d("2. Robust (Без дырок!)", makeLogDB(spec_robust, add_one=False))