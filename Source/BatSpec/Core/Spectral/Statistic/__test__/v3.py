import numpy as np
import time
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def z_norm_numba_v3_no_alloc(arr, window_size):
    h, w = arr.shape
    wh = window_size // 2
    ws2 = window_size * window_size
    
    result = np.empty_like(arr)
    padded = np.empty((h + 2*wh, w + 2*wh), dtype=arr.dtype)
    padded[wh:-wh, wh:-wh] = arr
    
    # prange идет по строкам
    for i in prange(h):
        # АЛЛОКАЦИЯ ПАМЯТИ ТОЛЬКО 1 РАЗ НА СТРОКУ (ВМЕСТО 1 РАЗ НА ПИКСЕЛЬ!)
        # Каждый поток получает свой собственный буфер
        temp_window = np.empty(ws2, dtype=arr.dtype)
        temp_dev = np.empty(ws2, dtype=arr.dtype)
        
        for j in range(w):
            # Ручное копирование пикселей окна в буфер (быстрее, чем flatten)
            idx = 0
            for dy in range(window_size):
                for dx in range(window_size):
                    temp_window[idx] = padded[i+dy, j+dx]
                    idx += 1
            
            med = np.median(temp_window)
            
            # Ручное вычисление отклонений без создания новых массивов
            for k in range(ws2):
                temp_dev[k] = np.abs(temp_window[k] - med)
                
            mad = np.median(temp_dev)
            
            result[i, j] = (arr[i, j] - med) / (mad * 1.4826 + 1e-8)
            
    return result

# Быстрый тест только для нее:
SIZE = 512
data = np.random.randn(SIZE, SIZE).astype(np.float32)

print("Прогрев V3...")
_ = z_norm_numba_v3_no_alloc(data[:20, :20], 3)
print("Готово.\n")

for w in [3, 15, 51]:
    t0 = time.time()
    z_norm_numba_v3_no_alloc(data, w)
    print(f"Окно {w}x{w}: {time.time() - t0:.3f} s")