import numpy as np
import time

try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter as gpu_median_filter
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("CuPy не установлен или нет совместимого GPU NVIDIA.")

from scipy.ndimage import median_filter as cpu_median_filter

# ==========================================
# GPU РЕАЛИЗАЦИЯ (CuPy)
# ==========================================
def z_norm_gpu_cupy(arr_cpu, window_size):
    # 1. Переносим данные из RAM (оперативной памяти) в VRAM (память видеокарты)
    arr_gpu = cp.asarray(arr_cpu, dtype=cp.float32)
    
    # 2. Считаем медиану на тысячах ядер GPU (режим reflect поддерживается!)
    med_gpu = gpu_median_filter(arr_gpu, size=window_size, mode='reflect')
    
    # 3. Векторные вычисления отклонений (GPU делает это мгновенно)
    dev_gpu = cp.abs(arr_gpu - med_gpu)
    
    # 4. Считаем MAD на GPU
    mad_gpu = gpu_median_filter(dev_gpu, size=window_size, mode='reflect')
    
    # 5. Итоговый Z-score
    z_gpu = (arr_gpu - med_gpu) / (mad_gpu * 1.4826 + 1e-8)
    
    # 6. Синхронизируем и возвращаем данные обратно в обычную память (NumPy)
    cp.cuda.Stream.null.synchronize() # Гарантируем завершение вычислений
    return cp.asnumpy(z_gpu)


# ==========================================
# CPU РЕАЛИЗАЦИЯ (Твой любимый ndimage)
# ==========================================
def z_norm_cpu_scipy(arr_cpu, window_size):
    med = cpu_median_filter(arr_cpu, size=window_size, mode='reflect')
    dev = np.abs(arr_cpu - med)
    mad = cpu_median_filter(dev, size=window_size, mode='reflect')
    return (arr_cpu - med) / (mad * 1.4826 + 1e-8)


# ==========================================
# ТЕСТИРОВАНИЕ CPU vs GPU
# ==========================================
def run_gpu_benchmark():
    if not HAS_GPU:
        return
        
    SIZE = 512  # Увеличим массив до 1024x1024 (в 4 раза больше, чем 512x512)
    WINDOW = 51
    data = np.random.randn(SIZE, SIZE).astype(np.float32)
    
    print(f"Массив: {SIZE}x{SIZE} (~1 млн элементов), Окно: {WINDOW}x{WINDOW}\n")
    
    # --- ПРОГРЕВ GPU (Обязательно!) ---
    # При первом вызове CuPy тратит ~1-2 секунды на инициализацию контекста CUDA
    print("Инициализация CUDA и прогрев GPU...")
    _ = z_norm_gpu_cupy(data[:100, :100], 5)
    print("Готово. Начинаем бой.\n")
    
    # --- ТЕСТ GPU ---
    t0 = time.time()
    res_gpu = z_norm_gpu_cupy(data, WINDOW)
    t_gpu = time.time() - t0
    print(f"GPU (CuPy ndimage reflect): {t_gpu:.3f} сек 🚀")
    
    # --- ТЕСТ CPU ---
    print("\nСчитаем на CPU (можно пойти заварить чай)...")
    t0 = time.time()
    res_cpu = z_norm_cpu_scipy(data, WINDOW)
    t_cpu = time.time() - t0
    print(f"CPU (SciPy ndimage reflect): {t_cpu:.3f} сек 🐢")
    
    # Проверка точности
    diff = np.abs(res_gpu - res_cpu).max()
    print(f"\nУскорение: в {t_cpu / t_gpu:.1f} раз!")
    print(f"Разница в расчетах (CPU vs GPU): {diff:.5e} (должно быть близко к 0)")

if __name__ == "__main__":
    run_gpu_benchmark()