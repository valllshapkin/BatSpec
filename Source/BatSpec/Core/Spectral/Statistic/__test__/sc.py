import numpy as np
import time
from scipy.signal import medfilt2d
from scipy.ndimage import median_filter

SIZE = 512
WINDOW = 51
data = np.random.randn(SIZE, SIZE).astype(np.float32)

print(f"Тест SciPy (Массив {SIZE}x{SIZE}, Окно {WINDOW}x{WINDOW}, float32)\n")

# 1. Справедливый бой (одинаковые условия: padding нулями)
t0 = time.time()
_ = medfilt2d(data, kernel_size=WINDOW)
t_signal = time.time() - t0
print(f"signal.medfilt2d:                 {t_signal:.3f} сек")

t0 = time.time()
_ = median_filter(data, size=WINDOW, mode='constant', cval=0.0)
t_ndimage_const = time.time() - t0
print(f"ndimage.median_filter (constant): {t_ndimage_const:.3f} сек")

# 2. Реальное использование (правильный padding для Z-score)
t0 = time.time()
_ = median_filter(data, size=WINDOW, mode='reflect')
t_ndimage_reflect = time.time() - t0
print(f"ndimage.median_filter (reflect):  {t_ndimage_reflect:.3f} сек")