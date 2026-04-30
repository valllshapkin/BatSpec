import numpy as np
import numpy.typing as npt

def fast_median_2d(
    input_padded: npt.NDArray[np.float32],
    h_orig: int,
    w_orig: int,
    kh: int,
    kw: int
) -> npt.NDArray[np.float32]:
    """
    Вычисляет локальную (скользящую) медиану для 2D массива.
    
    Использует структуру Dual-Heap O(K*logK) вместо O(K*K) и многопоточность C++ (OpenMP).
    Алгоритмически быстрее SciPy в десятки раз.

    Args:
        input_padded (NDArray[np.float32]): Исходный массив, предварительно расширенный (np.pad) 
                                            на `kh // 2` и `kw // 2` с каждой стороны.
        h_orig (int): Оригинальная высота массива (без паддинга).
        w_orig (int): Оригинальная ширина массива (без паддинга).
        kh (int): Размер скользящего окна по оси Y.
        kw (int): Размер скользящего окна по оси X.

    Returns:
        NDArray[np.float32]: Результирующая матрица формы (h_orig, w_orig) с медианами.
    """
    ...
