import pytest
import numpy as np

import MultiArray as ma
import MultiArray.Linalg as ml
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

# Не тестируем Linalg на GPU в рамках базовых тестов, чтобы избежать падений из-за памяти
CPU_CONTEXTS = [
    ArrayContext(fw, DeviceType.CPU, None) for fw in AVAILABLE_FRAMEWORKS 
    if fw != Framework.CUPY
]

def get_fw_name(ctx: ArrayContext) -> str:
    if ctx.isNumpy(): return "NUMPY"
    if ctx.isTorch(): return "TORCH"
    if ctx.isTensorflow(): return "TENSORFLOW"
    if ctx.isJax(): return "JAX"
    return "UNKNOWN"

# ==========================================
# ТЕСТЫ LINALG
# ==========================================

@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_matmul_and_dot(ctx):
    # Двумерные матрицы
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np_b = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    
    a = ma.convert_to(np_a, ctx)
    b = ma.convert_to(np_b, ctx)
    
    res_matmul = ml.matmul(a, b)
    np.testing.assert_allclose(ma.to_numpy(res_matmul), np.matmul(np_a, np_b), rtol=1e-5, atol=1e-5)
    
    # Одномерные векторы
    np_v1 = np.array([1.0, 2.0], dtype=np.float32)
    np_v2 = np.array([3.0, 4.0], dtype=np.float32)
    
    v1 = ma.convert_to(np_v1, ctx)
    v2 = ma.convert_to(np_v2, ctx)
    
    res_dot = ml.dot(v1, v2)
    np.testing.assert_allclose(ma.to_numpy(res_dot), np.dot(np_v1, np_v2), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_inv(ctx):
    # Матрица, которая легко обращается
    np_a = np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float32)
    a = ma.convert_to(np_a, ctx)
    
    a_inv = ml.inv(a)
    np.testing.assert_allclose(ma.to_numpy(a_inv), np.linalg.inv(np_a), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_svd(ctx):
    # Используем вещественную матрицу (float32 критично для SVD в TF)
    np_a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    a = ma.convert_to(np_a, ctx)
    
    u, s, vh = ml.svd(a, full_matrices=False)
    
    np_u = ma.to_numpy(u)
    np_s = ma.to_numpy(s)
    np_vh = ma.to_numpy(vh)
    
    # Проверяем размерности (full_matrices=False)
    assert np_u.shape == (3, 2)
    assert np_s.shape == (2,)
    assert np_vh.shape == (2, 2)
    
    # Проверяем реконструкцию исходной матрицы: A = U * S * Vh
    # У SVD есть проблема неоднозначности знаков у разных бэкендов, поэтому 
    # лучше проверять именно успешность восстановления исходной матрицы.
    reconstructed = np_u @ np.diag(np_s) @ np_vh
    np.testing.assert_allclose(reconstructed, np_a, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_einsum(ctx):
    # Тестируем транспонирование и умножение через einsum: C_ij = A_ki * B_kj
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np_b = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    
    a = ma.convert_to(np_a, ctx)
    b = ma.convert_to(np_b, ctx)
    
    # "ki,kj->ij"
    res = ml.einsum("ki,kj->ij", a, b)
    expected = np.einsum("ki,kj->ij", np_a, np_b)
    
    np.testing.assert_allclose(ma.to_numpy(res), expected, rtol=1e-5, atol=1e-5)
