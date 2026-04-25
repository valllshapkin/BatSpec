import pytest
import numpy as np

import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType

# ==========================================
# ОБНАРУЖЕНИЕ УСТАНОВЛЕННЫХ ФРЕЙМВОРКОВ
# ==========================================

AVAILABLE_FRAMEWORKS = [Framework.NUMPY]

try: import torch; AVAILABLE_FRAMEWORKS.append(Framework.TORCH)
except ImportError: pass

try: import tensorflow as tf; AVAILABLE_FRAMEWORKS.append(Framework.TENSORFLOW)
except ImportError: pass

try: import jax; AVAILABLE_FRAMEWORKS.append(Framework.JAX)
except ImportError: pass

try: import cupy; AVAILABLE_FRAMEWORKS.append(Framework.CUPY)
except ImportError: pass

CPU_CONTEXTS = [
    ArrayContext(fw, DeviceType.CPU, None) for fw in AVAILABLE_FRAMEWORKS 
    if fw != Framework.CUPY
]

# Хелпер для Pytest, чтобы красиво называть тесты без доступа к _framework
def get_fw_name(ctx: ArrayContext) -> str:
    if ctx.isNumpy(): return "NUMPY"
    if ctx.isTorch(): return "TORCH"
    if ctx.isTensorflow(): return "TENSORFLOW"
    if ctx.isJax(): return "JAX"
    if ctx.isCupy(): return "CUPY"
    return "UNKNOWN"

# ==========================================
# ТЕСТЫ
# ==========================================

@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_creation_and_conversion(ctx):
    # 1. Создаем массив через наш API
    arr = ma.arange(5, ctx)
    
    # ПРОБЛЕМА РЕШЕНА: Используем fw_equal вместо .framework
    assert ArrayContext.fw_equal(ArrayContext.from_array(arr), ctx)
    
    # 2. Создаем zeros_like
    z = ma.zeros_like(arr)
    assert ArrayContext.fw_equal(ArrayContext.from_array(z), ctx)
    
    # 3. Перегоняем в NumPy и сравниваем с эталоном
    np_arr = ma.to_numpy(arr)
    np_z = ma.to_numpy(z)
    
    assert isinstance(np_arr, np.ndarray)
    np.testing.assert_array_equal(np_arr, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(np_z, np.array([0, 0, 0, 0, 0]))


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_math_operations(ctx):
    np_ref = np.array([[-1.0, 2.0], [3.0, -4.0]])
    arr = ma.convert_to(np_ref, ctx)
    
    arr_abs = ma.abs(arr)
    arr_sum = ma.sum(arr, axis=0)
    arr_max = ma.max(arr)
    
    np.testing.assert_allclose(ma.to_numpy(arr_abs), np.abs(np_ref))
    np.testing.assert_allclose(ma.to_numpy(arr_sum), np.sum(np_ref, axis=0))
    np.testing.assert_allclose(ma.to_numpy(arr_max), np.max(np_ref))


def test_cross_framework_conversion():
    if Framework.TORCH not in AVAILABLE_FRAMEWORKS or Framework.TENSORFLOW not in AVAILABLE_FRAMEWORKS:
        pytest.skip("Нужны Torch и TensorFlow")
        
    ctx_torch = ArrayContext(Framework.TORCH, DeviceType.CPU, None)
    ctx_tf = ArrayContext(Framework.TENSORFLOW, DeviceType.CPU, None)
    
    t_arr = ma.arange(10, ctx_torch)
    # Используем isTorch()
    assert ArrayContext.from_array(t_arr).isTorch()
    
    tf_arr = ma.convert_to(t_arr, ctx_tf)
    # Используем isTensorflow()
    assert ArrayContext.from_array(tf_arr).isTensorflow()
    
    np.testing.assert_array_equal(ma.to_numpy(tf_arr), np.arange(10))


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_shape_manipulations(ctx):
    arr1 = ma.convert_to(np.zeros((2, 3)), ctx)
    arr2 = ma.convert_to(np.ones((2, 3)), ctx)
    
    concat_arr = ma.concatenate([arr1, arr2], axis=0)
    assert ma.to_numpy(concat_arr).shape == (4, 3)
    
    reshaped_arr = ma.reshape(concat_arr, (3, 4))
    assert ma.to_numpy(reshaped_arr).shape == (3, 4)