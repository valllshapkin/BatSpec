import numpy as np
import time
import cv2
from numba import njit, prange
from scipy.ndimage import median_filter
from numpy.lib.stride_tricks import sliding_window_view

# =========================================================
# ГАРАНТИЯ CPU ДЛЯ OPENCV
# Отключаем OpenCL аппаратно, чтобы OpenCV работал только на CPU
# =========================================================
cv2.ocl.setUseOpenCL(False)

# =========================================================
# 1. МЕТОД NUMPY (Strides)
# =========================================================
def z_norm_numpy(arr, window_size):
    wh = window_size // 2
    padded = np.pad(arr, wh, mode='reflect')
    
    try:
        windows = sliding_window_view(padded, (window_size, window_size))
        windows = windows.reshape(arr.shape[0], arr.shape[1], -1)
        med = np.median(windows, axis=-1, keepdims=True)
        dev = np.abs(windows - med)
        mad = np.median(dev, axis=-1, keepdims=True)
        z = (arr[..., None] - med) / (mad * 1.4826 + 1e-8)
        return z.squeeze()
    except MemoryError:
        return "OOM"
    except Exception as e:
        return "Crash"

# =========================================================
# 2. МЕТОД NUMBA V1 (Классическая, из прошлого теста с ravel)
# =========================================================
@njit(parallel=True, fastmath=True)
def z_norm_numba_v1(arr, window_size):
    h, w = arr.shape
    wh = window_size // 2
    result = np.empty_like(arr)
    padded = np.empty((h + 2*wh, w + 2*wh), dtype=arr.dtype)
    padded[wh:-wh, wh:-wh] = arr # Простой паддинг
    
    for i in prange(h):
        for j in range(w):
            # ravel() создает View, что замедляет работу памяти в потоках
            window = padded[i:i+window_size, j:j+window_size].ravel()
            med = np.median(window)
            dev = np.abs(window - med)
            mad = np.median(dev)
            result[i, j] = (arr[i, j] - med) / (mad * 1.4826 + 1e-8)
    return result

# =========================================================
# 3. МЕТОД NUMBA V2 (Ultra: Оптимизация кэша через flatten)
# =========================================================
@njit(parallel=True, fastmath=True)
def z_norm_numba_v2(arr, window_size):
    h, w = arr.shape
    wh = window_size // 2
    result = np.empty_like(arr)
    padded = np.empty((h + 2*wh, w + 2*wh), dtype=arr.dtype)
    padded[wh:-wh, wh:-wh] = arr
    
    for i in prange(h):
        for j in range(w):
            # flatten() создает непрерывный массив (Contiguous), Numba сортирует его молниеносно
            window = padded[i:i+window_size, j:j+window_size].flatten()
            med = np.median(window)
            dev = np.abs(window - med)
            mad = np.median(dev)
            result[i, j] = (arr[i, j] - med) / (mad * 1.4826 + 1e-8)
    return result



# =========================================================
# 4. МЕТОД SCIPY (Двойной фильтр на C)
# =========================================================
def z_norm_scipy(arr, window_size):
    med = median_filter(arr, size=window_size)
    dev = np.abs(arr - med)
    mad = median_filter(dev, size=window_size)
    z = (arr - med) / (mad * 1.4826 + 1e-8)
    return z

# =========================================================
# 5. МЕТОД OPENCV (CV2 Blur)
# =========================================================
def z_norm_cv2(arr, window_size):
    try:
        # В OpenCV ksize (window_size) должен быть нечетным
        med = cv2.medianBlur(arr, window_size)
        dev = np.abs(arr - med)
        mad = cv2.medianBlur(dev, window_size)
        z = (arr - med) / (mad * 1.4826 + 1e-8)
        return z
    except cv2.error:
        # OpenCV падает, если float32 и размер окна > 5
        return "❌ Лимит CV2 (только <=5)"

# =========================================================
# ЗАПУСК ТЕСТОВ
# =========================================================
def run_benchmarks():
    SIZE = 512
    data = np.random.randn(SIZE, SIZE).astype(np.float32)
    windows = [3, 15, 51]
    
    print("=" * 110)
    print("ПРОГРЕВ JIT КОМПИЛЯТОРА NUMBA...")
    t_start = time.time()
    _ = z_norm_numba_v1(data[:10, :10], 3)
    _ = z_norm_numba_v2(data[:10, :10], 3)
    print(f"Компиляция завершена за {time.time() - t_start:.2f} сек. Начинаем тест.")
    print("=" * 110)
    
    print(f"Массив данных: {SIZE}x{SIZE} (~262 тыс. элементов), dtype=float32")
    print(f"OpenCL в OpenCV: {'ВКЛ' if cv2.ocl.useOpenCL() else 'ОТКЛЮЧЕН (Только CPU)'}\n")
    
    header = f"{'Окно':<8} | {'NumPy':<13} | {'Numba V1 (Old)':<16} | {'Numba V2 (Ultra)':<18} | {'SciPy':<10} | {'OpenCV':<15}"
    print(header)
    print("-" * 110)
    
    for w in windows:
        results = []
        
        # 1. NumPy
        if w > 15:
            results.append(f"{'OOM/Crash':<13}")
        else:
            t0 = time.time()
            z_norm_numpy(data, w)
            results.append(f"{time.time() - t0:<13.3f}")
            
        # 2. Numba V1
        t0 = time.time()
        z_norm_numba_v1(data, w)
        results.append(f"{time.time() - t0:<16.3f}")
        
        # 3. Numba V2 (Ultra)
        t0 = time.time()
        z_norm_numba_v2(data, w)
        results.append(f"{time.time() - t0:<18.3f}")
        
        # 4. SciPy
        t0 = time.time()
        z_norm_scipy(data, w)
        results.append(f"{time.time() - t0:<10.3f}")
        
        # 5. OpenCV
        t0 = time.time()
        res_cv = z_norm_cv2(data, w)
        if isinstance(res_cv, str):
            results.append(res_cv) # Текст ошибки
        else:
            results.append(f"{time.time() - t0:<15.3f}")
            
        print(f"{str(w)+'x'+str(w):<8} | {' | '.join(results)}")
    print("-" * 110)

if __name__ == "__main__":
    run_benchmarks()