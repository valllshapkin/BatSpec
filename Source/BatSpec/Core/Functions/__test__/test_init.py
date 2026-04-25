import pytest
import numpy as np

# --- Импорты из твоего проекта ---
from BatSpec.Core.Functions import SpecFunc, TimeFunc, FreqFunc
from BatSpec.Core.Physical.Units import UREG

# --- Импорты нашего нового ядра ---
import MultiArray as ma
from MultiArray import ArrayContext, Framework, DeviceType

# ==========================================
# ОБНАРУЖЕНИЕ ФРЕЙМВОРКОВ И КОНТЕКСТЫ ДЛЯ ТЕСТОВ
# ==========================================

AVAILABLE_FRAMEWORKS = [Framework.NUMPY]
try:
    import torch
    AVAILABLE_FRAMEWORKS.append(Framework.TORCH)
except ImportError:
    torch = None

try:
    import tensorflow as tf
    AVAILABLE_FRAMEWORKS.append(Framework.TENSORFLOW)
except ImportError:
    tf = None

CPU_CONTEXTS = [
    ArrayContext(fw, DeviceType.CPU, None) for fw in AVAILABLE_FRAMEWORKS 
    if fw != Framework.CUPY
]

def get_fw_name(ctx: ArrayContext) -> str:
    return ctx._framework.name

# ==========================================
# ФИКСТУРА ДЛЯ ГЕНЕРАЦИИ ТЕСТОВЫХ ДАННЫХ
# ==========================================

@pytest.fixture(scope="module")
def spec_data():
    """Создает базовые NumPy данные один раз для всех тестов в этом файле."""
    batch_size = 3
    num_freqs = 5
    num_times = 10
    
    freq_np = np.linspace(10, 50, num_freqs, dtype=np.float32)
    time_np = np.linspace(0.0, 1.0, num_times, dtype=np.float32)
    matrix_np = np.random.rand(batch_size, num_freqs, num_times).astype(np.float32)
    
    dummy_unit = UREG.volt if hasattr(UREG, 'volt') else UREG.dimensionless
    
    return {
        "matrix": matrix_np,
        "freq": freq_np,
        "time": time_np,
        "unit": dummy_unit,
        "batch_size": batch_size,
        "num_freqs": num_freqs,
        "num_times": num_times
    }

# ==========================================
# ТЕСТЫ
# ==========================================

def test_initialization_and_properties_numpy(spec_data):
    """Тест 1: Инициализация и базовые свойства на NumPy."""
    spec_np = SpecFunc(
        matrix=(spec_data["matrix"], spec_data["unit"]), 
        freq=spec_data["freq"], 
        time=spec_data["time"]
    )
    
    # Проверяем контекст и типы
    assert spec_np.context.isNumpy()
    assert isinstance(spec_np.values[0], np.ndarray)
    assert spec_np.values[0].shape == (spec_data["batch_size"], spec_data["num_freqs"], spec_data["num_times"])
    
    # Проверяем расчетные свойства
    expected_dt = (spec_data["time"][-1] - spec_data["time"][0]) / (spec_data["num_times"] - 1)
    expected_df = (spec_data["freq"][-1] - spec_data["freq"][0]) / (spec_data["num_freqs"] - 1)
    
    assert spec_np.dt[0] == pytest.approx(expected_dt)
    assert spec_np.dt[1] == UREG.second
    assert spec_np.df[0] == pytest.approx(expected_df)
    assert spec_np.df[1] == UREG.hertz

def test_slicing_and_iteration(spec_data):
    """Тест 2: Магические методы __len__, __getitem__, __iter__."""
    spec_np = SpecFunc(
        matrix=(spec_data["matrix"], spec_data["unit"]),
        freq=spec_data["freq"],
        time=spec_data["time"]
    )
    
    # __len__
    assert len(spec_np) == spec_data["batch_size"]
    
    # __getitem__
    single_spec = spec_np[0]
    assert isinstance(single_spec, SpecFunc)
    assert single_spec.values[0].shape == (spec_data["num_freqs"], spec_data["num_times"])
    
    # __iter__
    count = 0
    for item in spec_np:
        assert isinstance(item, SpecFunc)
        assert item.values[0].shape == (spec_data["num_freqs"], spec_data["num_times"])
        count += 1
    assert count == spec_data["batch_size"]

@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=get_fw_name)
def test_integration(ctx, spec_data):
    """Тест 3: Интегрирование, параметризованное по всем фреймворкам."""
    spec_np = SpecFunc(
        matrix=(spec_data["matrix"], spec_data["unit"]),
        freq=spec_data["freq"],
        time=spec_data["time"]
    )
    # Конвертируем в целевой контекст
    spec_fw = spec_np.to_context(ctx)
    
    # integrateOverFreq
    time_func = spec_fw.integrateOverFreq()
    assert isinstance(time_func, TimeFunc)
    assert time_func.values[0].shape == (spec_data["batch_size"], spec_data["num_times"])
    
    # integrateOverTime
    freq_func = spec_fw.integrateOverTime()
    assert isinstance(freq_func, FreqFunc)
    assert freq_func.values[0].shape == (spec_data["batch_size"], spec_data["num_freqs"])

@pytest.mark.skipif(torch is None, reason="PyTorch не установлен")
def test_cross_framework_conversion_torch(spec_data):
    """Тест 4: Конвертация NumPy <-> PyTorch и проверка данных."""
    spec_np = SpecFunc(
        matrix=(spec_data["matrix"], spec_data["unit"]),
        freq=spec_data["freq"],
        time=spec_data["time"]
    )
    
    # 1. Конвертация в PyTorch
    ctx_torch = ArrayContext(Framework.TORCH, DeviceType.CPU, None)
    spec_torch = spec_np.to_context(ctx_torch)
    
    assert spec_torch.context.isTorch()
    assert isinstance(spec_torch.values[0], torch.Tensor)
    print(f"\nКонвертировано в PyTorch. Тип матрицы: {type(spec_torch.values[0])}")
    
    # 2. Обратная конвертация в NumPy
    ctx_numpy = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    spec_back_np = spec_torch.to_context(ctx_numpy)
    
    assert spec_back_np.context.isNumpy()
    assert isinstance(spec_back_np.values[0], np.ndarray)
    print(f"Конвертировано обратно в NumPy. Тип матрицы: {type(spec_back_np.values[0])}")
    
    # 3. Проверяем, что данные не исказились после round-trip
    np.testing.assert_allclose(spec_back_np.values[0], spec_data["matrix"])