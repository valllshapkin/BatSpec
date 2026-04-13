import numpy as np
from scipy.signal import correlate, find_peaks

def extract_patterns(image, pattern_length=200, threshold_sigma=6.0, max_patterns=5):
    """
    image: np.array формы (600_000, 300)
    pattern_length: длина паттерна вдоль оси 0
    threshold_sigma: порог для поиска пиков корреляции (в сигмах от среднего)
    max_patterns: сколько уникальных базисов искать
    """
    L, H = image.shape
    
    # Копия изображения, из которой мы будем "вычитать" найденные паттерны
    residual = image.copy()
    
    basis_patterns = []
    all_coordinates = []
    reconstructed_components = []
    
    for k in range(max_patterns):
        print(f"--- Поиск паттерна #{k+1} ---")
        
        # 1. Поиск "якоря" (чернового шаблона)
        # Считаем энергию окон (сумма квадратов)
        # Для скорости берем окно с шагом, равным половине длины паттерна
        step = pattern_length // 2
        energies = [np.sum(residual[i:i+pattern_length]**2) 
                    for i in range(0, L - pattern_length, step)]
        
        best_idx = np.argmax(energies) * step
        draft_template = residual[best_idx : best_idx+pattern_length].copy()
        
        # Если максимальная энергия упала (остался только шум), прерываем
        if np.var(draft_template) < np.var(residual) * 1.5:
            print("Остался только шум. Поиск завершен.")
            break

        # 2. Быстрая кросс-корреляция (через FFT)
        # Так как высота совпадает (300 и 300), 2D корреляция valid 
        # по факту является 1D корреляцией скользящего окна
        corr = correlate(residual, draft_template, mode='valid', method='fft')
        corr = corr.flatten()
        
        # 3. Поиск пиков (координат совпадений)
        # Корреляция с гауссовым шумом даст гауссово распределение.
        # Истинные совпадения будут экстремальными выбросами.
        mean_c = np.mean(corr)
        std_c = np.std(corr)
        
        # Ищем пики, которые сильно выше шума
        peaks, _ = find_peaks(corr, height=mean_c + threshold_sigma * std_c, distance=pattern_length//2)
        
        if len(peaks) == 0:
            print("Клоны не найдены, возможно это просто выброс шума.")
            break
            
        print(f"Найдено вхождений паттерна: {len(peaks)}")
        
        # 4. Получение чистого базисного паттерна (Усреднение)
        clean_pattern = np.zeros_like(draft_template)
        for p in peaks:
            clean_pattern += image[p : p+pattern_length]
        clean_pattern /= len(peaks)
        
        # 5. Разложение (создание слоя с этим паттерном) и вычитание (Deflation)
        component_image = np.zeros_like(image)
        for p in peaks:
            component_image[p : p+pattern_length] += clean_pattern
            # Вычитаем из остатка, чтобы на след. итерации искать другие узоры
            residual[p : p+pattern_length] -= clean_pattern
            
        # Сохраняем результаты
        basis_patterns.append(clean_pattern)
        all_coordinates.append(peaks)
        reconstructed_components.append(component_image)

    return basis_patterns, all_coordinates, reconstructed_components, residual

# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (Симуляция данных):
# ==========================================
if __name__ == "__main__":
    L, H = 600_000, 300
    W = 200
    
    print("Генерация тестовых данных...")
    # Фон - гауссов шум
    X = np.random.normal(0, 1, size=(L, H)).astype(np.float32)
    
    # Создаем 2 секретных узора
    pattern1 = np.sin(np.linspace(0, 10, W))[:, None] * np.random.randn(H) * 5
    pattern2 = np.cos(np.linspace(0, 20, W))[:, None] * np.random.randn(H) * 5
    
    # Раскидываем их в случайные места
    true_coords_1 = [5000, 150000, 340000, 550000]
    true_coords_2 = [80000, 210000, 480000]
    
    for c in true_coords_1: X[c:c+W] += pattern1
    for c in true_coords_2: X[c:c+W] += pattern2
    
    print("Запуск алгоритма...")
    bases, coords, components, noise_residual = extract_patterns(X, pattern_length=W)
    
    print("\n--- Итоги ---")
    for i, c in enumerate(coords):
        print(f"Базис {i+1}: найдено по координатам {c}")
        
    # Исходное изображение теперь разложено на сумму:
    # X_reconstructed = sum(components) + noise_residual