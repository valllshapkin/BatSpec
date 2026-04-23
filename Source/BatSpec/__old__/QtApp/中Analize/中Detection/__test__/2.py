import numpy as np
from scipy.signal import fftconvolve, find_peaks
import matplotlib.pyplot as plt
import os


def normalize_for_saving(img):
    """Нормализует изображение в [0, 1] для сохранения в PNG."""
    img_min = img.min()
    img_max = img.max()
    if img_max == img_min:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)

def extract_patterns_v2(image, pattern_length=200, threshold_sigma=6.0, max_patterns=5, min_occurrences=3, save_path="results"):
    """
    Улучшенная версия: быстрая 1D корреляция, подгонка амплитуды и визуализация.
    
    min_occurrences: Минимальное число вхождений, чтобы считать паттерн настоящим.
    save_path: Папка для сохранения PNG-изображений.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    L, H = image.shape
    residual = image.copy()
    
    basis_patterns = []
    all_coordinates = []
    reconstructed_components = []
    
    # Предварительно посчитаем дисперсию шума для критерия остановки
    initial_noise_var = np.var(image)
    
    for k in range(max_patterns):
        print(f"\n--- Поиск паттерна #{k+1} ---")
        
        # 1. УСКОРЕННЫЙ Поиск "якоря" через свертку
        # Это эквивалентно скользящему окну, но в разы быстрее list comprehension
        energies = fftconvolve(residual**2, np.ones((pattern_length, 1)), mode='valid').sum(axis=1)
        
        if len(energies) == 0:
            print("Изображение слишком короткое для поиска паттернов.")
            break
            
        best_idx = np.argmax(energies)
        draft_template = residual[best_idx : best_idx + pattern_length].copy()
        
        # Надежный критерий остановки: если самый энергичный участок не сильно
        # выделяется на фоне общего шума, то мы закончили.
        if np.var(draft_template) < initial_noise_var * 2.0:
            print("Не найдено участков с энергией, значительно превышающей фон. Поиск завершен.")
            break

        # 2. КРИТИЧЕСКОЕ УСКОРЕНИЕ: 1D Кросс-корреляция по каждой колонке
        # Это математически корректно для задачи сдвига только по одной оси.
        total_corr = np.zeros(L - pattern_length + 1)
        for h in range(H):
            # fftconvolve - это быстрая свертка. Корреляция = свертка с развернутым ядром.
            # [::-1] разворачивает шаблон для корреляции.
            col_corr = fftconvolve(residual[:, h], draft_template[::-1, h], 'valid')
            total_corr += col_corr
        
        # 3. Поиск пиков (координат совпадений)
        mean_c, std_c = np.mean(total_corr), np.std(total_corr)
        if std_c == 0: # Если корреляция константная
            print("Корреляция не дала результатов. Остановка.")
            break
            
        peaks, _ = find_peaks(total_corr, height=mean_c + threshold_sigma * std_c, distance=pattern_length // 2)
        
        # 4. НАДЕЖНЫЙ КРИТЕРИЙ ОСТАНОВКИ: если нашли мало вхождений
        if len(peaks) < min_occurrences:
            print(f"Найдено всего {len(peaks)} вхождений, что меньше порога ({min_occurrences}). Считаем это шумом. Поиск завершен.")
            break
            
        print(f"Найдено {len(peaks)} вхождений-кандидатов.")
        
        # 5. Получение чистого базисного паттерна (Усреднение)
        clean_pattern = np.zeros_like(draft_template, dtype=np.float32)
        for p in peaks:
            clean_pattern += image[p : p + pattern_length]
        clean_pattern /= len(peaks)
        
        # Сохраняем базис в PNG
        plt.imsave(os.path.join(save_path, f'basis_{k+1}.png'), normalize_for_saving(clean_pattern), cmap='gray')
        
        # 6. УЛУЧШЕНИЕ ТОЧНОСТИ: Разложение с подгонкой амплитуды и вычитание (Deflation)
        component_image = np.zeros_like(image, dtype=np.float32)
        pattern_norm_sq = np.sum(clean_pattern**2)
        
        final_coords = []
        for p in peaks:
            patch = image[p : p + pattern_length]
            
            # Находим оптимальную амплитуду alpha, чтобы patch ≈ alpha * clean_pattern
            # Решение по методу наименьших квадратов
            alpha = np.sum(patch * clean_pattern) / pattern_norm_sq if pattern_norm_sq > 1e-6 else 0
            
            # Если паттерн анти-коррелирует (alpha < 0) или слишком слабый, игнорируем
            if alpha < 0.5: 
                continue

            scaled_pattern = clean_pattern * alpha
            component_image[p : p + pattern_length] += scaled_pattern
            residual[p : p + pattern_length] -= scaled_pattern
            final_coords.append(p)

        print(f"После уточнения осталось {len(final_coords)} вхождений.")
            
        # Сохраняем результаты итерации
        basis_patterns.append(clean_pattern)
        all_coordinates.append(np.array(final_coords))
        reconstructed_components.append(component_image)
        
        # Сохраняем компонент в PNG
        plt.imsave(os.path.join(save_path, f'component_{k+1}.png'), normalize_for_saving(component_image), cmap='gray')

    # Сохраняем остаток (шум)
    plt.imsave(os.path.join(save_path, 'residual_noise.png'), normalize_for_saving(residual), cmap='gray')

    return basis_patterns, all_coordinates, reconstructed_components, residual

# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (Симуляция данных):
# ==========================================
if __name__ == "__main__":
    L, H = 600_000, 300
    W = 200
    
    print("Генерация тестовых данных...")
    X = np.random.normal(0, 1, size=(L, H)).astype(np.float32)
    
    # Создаем 2 секретных узора
    pattern1 = np.sin(np.linspace(0, 10, W))[:, None] * np.random.randn(H) * 5
    pattern2 = np.cos(np.linspace(0, 20, W))[:, None] * np.random.randn(H) * 3
    
    # Раскидываем их в случайные места с разной яркостью
    true_coords_1 = [5000, 150000, 340000, 550000]
    true_coords_2 = [80000, 210000, 480000]
    
    for c in true_coords_1: X[c:c+W] += pattern1 * (0.8 + np.random.rand() * 0.4) # Яркость от 0.8 до 1.2
    for c in true_coords_2: X[c:c+W] += pattern2 * (0.8 + np.random.rand() * 0.4)
    
    print("Запуск улучшенного алгоритма...")
    bases, coords, components, noise_residual = extract_patterns_v2(X, pattern_length=W, save_path="extraction_results")
    
    print("\n--- Итоги ---")
    if not bases:
        print("Не найдено ни одного базисного паттерна.")
    else:
        for i, c in enumerate(coords):
            print(f"Базис {i+1}: найден по координатам {c}")
    
    # Проверка разложения:
    # X_reconstructed = sum(components) + noise_residual
    # np.allclose(X, X_reconstructed) должен вернуть True