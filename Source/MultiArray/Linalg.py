from typing import Any, Tuple
from .Core import ArrayContext

# ==========================================
# УРОВЕНЬ 2: ЛИНЕЙНАЯ АЛГЕБРА (Linalg)
# ==========================================

def matmul(a: Any, b: Any) -> Any:
    """Умножение матриц (матричное произведение)."""
    ctx = ArrayContext.from_array(a)
    return ctx.fw.matmul(a, b)

def dot(a: Any, b: Any) -> Any:
    """
    Скалярное произведение.
    Разрешает проблемы PyTorch (где dot работает ТОЛЬКО с 1D-векторами).
    """
    ctx = ArrayContext.from_array(a)
    if ctx.isTensorflow():
        return ctx.fw.tensordot(a, b, axes=1)
    elif ctx.isTorch():
        if a.ndim == 1 and b.ndim == 1:
            return ctx.fw.dot(a, b)
        return ctx.fw.tensordot(a, b, dims=1)
    else:
        return ctx.fw.dot(a, b)

def einsum(equation: str, *operands: Any) -> Any:
    """Обобщенное сокращение тензоров (Einstein summation convention)."""
    ctx = ArrayContext.from_array(operands[0])
    return ctx.fw.einsum(equation, *operands)

def inv(a: Any) -> Any:
    """Обратная матрица квадратной матрицы."""
    ctx = ArrayContext.from_array(a)
    # К счастью, почти все фреймворки договорились класть inv в linalg
    return ctx.fw.linalg.inv(a)

def svd(a: Any, full_matrices: bool = True) -> Tuple[Any, Any, Any]:
    """
    Сингулярное разложение матрицы SVD (Singular Value Decomposition).
    Унифицированно возвращает кортеж: (U, S, Vh).
    
    ВНИМАНИЕ: Разрешает конфликт TensorFlow, который по умолчанию возвращает (S, U, V).
    """
    ctx = ArrayContext.from_array(a)
    
    if ctx.isTensorflow():
        # TensorFlow возвращает (S, U, V), где a = U * diag(S) * V^H
        # NumPy и PyTorch возвращают (U, S, Vh). Vh — это эрмитово-сопряженная матрица V.
        s, u, v = ctx.fw.linalg.svd(a, full_matrices=full_matrices)
        vh = ctx.fw.linalg.adjoint(v)
        return u, s, vh
        
    else:
        # NumPy, PyTorch (torch.linalg.svd), JAX, CuPy
        return ctx.fw.linalg.svd(a, full_matrices=full_matrices)

def eig(a: Any) -> Tuple[Any, Any]:
    """
    Собственные значения и собственные векторы.
    Возвращает: (eigenvalues, eigenvectors).
    """
    ctx = ArrayContext.from_array(a)
    return ctx.fw.linalg.eig(a)
