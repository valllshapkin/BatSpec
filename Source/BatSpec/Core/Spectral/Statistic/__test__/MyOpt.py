import numpy as np

# Тот самый импорт, который ты просил!
from BatSpec.Core.Spectral.Statistic import FastZScore

def apply_fast_zscore(data: np.ndarray, window_size: int = 51) -> np.ndarray:
    """Удобная обертка на Python для паддинга"""
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        
    h, w = data.shape
    wh = window_size // 2
    
    # Паддинг делаем в Питоне (это быстро)
    padded_data = np.pad(data, wh, mode='reflect')
    
    # Вызываем нашу С++ функцию с автодополнением и стабами!
    return FastZScore.local_zscore_cpu_opt(padded_data, h, w, window_size)

# Проверка:
if __name__ == "__main__":
    matrix = np.random.randn(512, 512).astype(np.float32)
    
    result = apply_fast_zscore(matrix, window_size=51)
    
    print(f"Успех! Форма результата: {result.shape}")