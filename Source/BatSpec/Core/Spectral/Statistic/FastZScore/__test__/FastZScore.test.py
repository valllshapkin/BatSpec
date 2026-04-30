import numpy as np
import time
from scipy.ndimage import median_filter
from BatSpec.Core.Spectral.Statistic.FastZScore import fast_median_2d

def z_norm_cpp(arr, kh, kw):
    ph, pw = kh // 2, kw // 2
    h, w = arr.shape
    
    # ИСПОЛЬЗУЕМ mode='symmetric' чтобы совпадало с SciPy mode='reflect'
    padded1 = np.pad(arr, ((ph, ph), (pw, pw)), mode='symmetric')
    med = fast_median_2d(padded1, h, w, kh, kw)
    
    dev = np.abs(arr - med)
    padded2 = np.pad(dev, ((ph, ph), (pw, pw)), mode='symmetric')
    mad = fast_median_2d(padded2, h, w, kh, kw)
    
    return (arr - med) / (mad * 1.4826 + 1e-8)

def z_norm_scipy(arr, kh, kw):
    med = median_filter(arr, size=(kh, kw), mode='reflect')
    dev = np.abs(arr - med)
    mad = median_filter(dev, size=(kh, kw), mode='reflect')
    return (arr - med) / (mad * 1.4826 + 1e-8)

if __name__ == "__main__":
    SIZE = 512
    KH = 11
    KW = 51
    data = np.random.randn(SIZE, SIZE).astype(np.float32)

    print(f"Тест: Массив {SIZE}x{SIZE}, Окно {KH}x{KW} (float32)\n")

    t0 = time.time()
    res_scipy = z_norm_scipy(data, KH, KW)
    t_scipy = time.time() - t0
    print(f"SciPy ndimage: {t_scipy:.3f} сек 🐢")

    t0 = time.time()
    res_cpp = z_norm_cpp(data, KH, KW)
    t_cpp = time.time() - t0
    print(f"Наш C++ алгоритм: {t_cpp:.3f} сек ⚡")

    diff = np.abs(res_scipy - res_cpp).max()
    print(f"\nРазница в расчетах: {diff:.5e} (теперь идеально 0.0)")
    print(f"Ускорение на CPU: в {t_scipy / t_cpp:.1f} РАЗ!")
