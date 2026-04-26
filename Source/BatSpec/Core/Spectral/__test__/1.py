from pathlib import Path
ScriptDir = Path(__file__).parent
import time
import numpy as np

from BatSpec.Core.ConvWindow import TEST_HANN_WINODW
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.SaveIntegral import SaveIntegral
from BatSpec.Core.Spectral import makeSpec, makeLogDB, DSPContext
from BatSpec.Core.Functions import TimeFunc, SpecFunc
from BatSpec.Core.Physical.Units import UREG
from BatSpec.Visualize import update_function, update_spec2d, run_visualizer

from MultiArray.Core import ArrayContext, Framework, DeviceType


def warmup_gpu():
    """Функция для инициализации драйверов CUDA/XLA перед замерами времени."""
    print("--- Прогрев GPU (Инициализация CUDA) ---")
    
    # Генерируем 1 секунду шума
    dummy_sr = 48000
    dummy_time = np.linspace(0, 1.0, dummy_sr)
    dummy_vals = np.random.randn(dummy_sr).astype(np.float32)
    dummy_sig = TimeFunc(values=(dummy_vals, UREG.FS), axis=dummy_time)
    
    ctx_tf = ArrayContext(Framework.TENSORFLOW, DeviceType.GPU, None)
    ctx_torch = ArrayContext(Framework.TORCH, DeviceType.GPU, None)
    
    try:
        t0 = time.time()
        with DSPContext(ctx_torch):
            makeSpec(dummy_sig, TEST_HANN_WINODW, overlap=0.5, bins=100)
        print(f"Torch GPU прогрет за {time.time() - t0:.3f} сек")
    except Exception as e:
        print("Torch недоступен или ошибка прогрева:", e)
        
    try:
        t0 = time.time()
        with DSPContext(ctx_tf):
            makeSpec(dummy_sig, TEST_HANN_WINODW, overlap=0.5, bins=100)
        print(f"TF GPU прогрет за {time.time() - t0:.3f} сек")
    except Exception as e:
        print("TF недоступен или ошибка прогрева:", e)
        
    print("----------------------------------------\n")


@run_visualizer
def main():
    # 1. Прогрев бэкендов
    warmup_gpu()
    
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    
    # Контексты
    ctx_np_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    ctx_tf_gpu = ArrayContext(Framework.TENSORFLOW, DeviceType.GPU, None)
    ctx_torch_gpu = ArrayContext(Framework.TORCH, DeviceType.GPU, None)
    
    print("--- Основные вычисления ---")
    
    # 1. NumPy
    t0 = time.time()
    with DSPContext(ctx_np_cpu):
        spec_np = makeSpec(record, TEST_HANN_WINODW, overlap=0.8, bins=300)
    print(f"NumPy STFT: {time.time() - t0:.3f} сек")
    update_spec2d("1. NumPy", makeLogDB(spec_np, add_one=False))
    
    # 2. PyTorch GPU
    t0 = time.time()
    with DSPContext(ctx_torch_gpu):
        spec_torch = makeSpec(record, TEST_HANN_WINODW, overlap=0.8, bins=300)
    print(f"Torch GPU STFT: {time.time() - t0:.3f} сек")
    update_spec2d("2. PyTorch", makeLogDB(spec_torch, add_one=False))

    # 3. TensorFlow GPU
    t0 = time.time()
    with DSPContext(ctx_tf_gpu):
        spec_tf = makeSpec(record, TEST_HANN_WINODW, overlap=0.8, bins=300)
    print(f"TF GPU STFT: {time.time() - t0:.3f} сек")
    update_spec2d("3. TensorFlow", makeLogDB(spec_tf, add_one=False))

    # 4. Сборка гибридной спектрограммы
    # Так как makeSpec возвращает объекты в исходном контексте (тут это NumPy),
    # мы можем спокойно использовать классический np.maximum!
    mat_tf, u = spec_tf.values
    mat_torch, _ = spec_torch.values

    print(f'''
DEBUG SHAPE: spec_tf.values[0].shape = {spec_tf.values[0].shape}
DEBUG SHAPE: spec_tf.time[0].shape = {spec_tf.time[0].shape}
DEBUG SHAPE: spec_tf.freq[0].shape = {spec_tf.freq[0].shape}

DEBUG SHAPE: spec_torch.values[0].shape = {spec_torch.values[0].shape}
DEBUG SHAPE: spec_torch.time[0].shape = {spec_torch.time[0].shape}
DEBUG SHAPE: spec_torch.freq[0].shape = {spec_torch.freq[0].shape}

DEBUG SHAPE: spec_np.values[0].shape = {spec_np.values[0].shape}
DEBUG SHAPE: spec_np.time[0].shape = {spec_np.time[0].shape}
DEBUG SHAPE: spec_np.freq[0].shape = {spec_np.freq[0].shape}
''')
    
    mat_max = np.maximum(mat_tf, mat_torch)
    
    spec_hybrid = SpecFunc(
        matrix=(mat_max, u), 
        freq=spec_tf.freq[0], 
        time=spec_tf.time[0]
    )
    
    update_spec2d("4. Hybrid (TF max Torch)", makeLogDB(spec_hybrid, add_one=False))