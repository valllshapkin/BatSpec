from pathlib import Path
import time
import numpy as np

ScriptDir = Path(__file__).parent

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW, TEST_BLHA_WINODW
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.SaveIntegral import SaveIntegral
from BatSpec.Core.Spectral import makeSpec, makeLogDB, DSPContext, makeRobustSpec
from BatSpec.Core.Functions import TimeFunc, SpecFunc
from BatSpec.Core.Physical.Units import UREG
from BatSpec.Visualize import update_spec2d, run_visualizer
from MultiArray.Core import ArrayContext, Framework, DeviceType
# --- Импорты MultiArray ---

@run_visualizer
def main():
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    
    
    ctx_torch_gpu = ArrayContext(Framework.TORCH, DeviceType.GPU, None)
    
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