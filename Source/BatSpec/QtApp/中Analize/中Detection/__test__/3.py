import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN

def normalize_for_saving(img):
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min) if img_max > img_min else np.zeros_like(img)

def find_and_decompose_patterns(image, pattern_length=200, n_candidates=2000, 
                                 proj_dim=64, dbscan_eps=0.2, ncc_threshold=0.6, 
                                 save_path="results_ncc"):
    """
    Находит паттерны через случайные проекции, кластеризацию и затем точный поиск NCC.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    L, H = image.shape
    W = pattern_length

    # --- 1. Грубый поиск кандидатов ---
    print("1. Поиск участков-кандидатов по энергии...")
    # Используем float32 для скорости
    img_f32 = image.astype(np.float32)
    energy = np.convolve(np.sum(img_f32**2, axis=1), np.ones(W, dtype=np.float32), 'valid')
    
    # Берем больше кандидатов, чтобы точно не пропустить истинные паттерны
    peaks, _ = find_peaks(energy, distance=W//2)
    if len(peaks) > n_candidates:
        top_peak_indices = peaks[np.argsort(energy[peaks])[-n_candidates:]]
    else:
        top_peak_indices = peaks
        
    if len(top_peak_indices) < 10:
        print("Слишком мало кандидатов.")
        return [], [], []

    print(f"Извлечено {len(top_peak_indices)} кандидатов.")

    # --- 2. Извлечение и Случайные Проекции (Random Projections) ---
    print("2. Сжатие патчей случайными проекциями...")
    patches = np.array([img_f32[i:i+W] for i in top_peak_indices])
    vectorized_patches = patches.reshape(len(patches), -1)
    
    # Генерируем случайную матрицу проекций один раз
    np.random.seed(42)
    proj_matrix = np.random.randn(W * H, proj_dim).astype(np.float32) / np.sqrt(proj_dim)
    
    # Проецируем (это просто матричное умножение, работает мгновенно)
    reduced_vectors = vectorized_patches @ proj_matrix

    # --- 3. Кластеризация (Косинусное расстояние) ---
    print("3. Кластеризация патчей...")
    # metric='cosine' идеально подходит: он игнорирует общую яркость (которую дает шум)
    # и смотрит только на "форму" патча.
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=3, metric='cosine')
    labels = dbscan.fit_predict(reduced_vectors)
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    if n_clusters == 0:
        print("Не найдено кластеров. Попробуйте увеличить dbscan_eps (например, 0.25) или n_candidates.")
        return [], [], []

    print(f"Найдено {n_clusters} уникальных кластеров (типов паттернов).")

    # --- 4. Формирование базисов и ТОЧНЫЙ поиск через NCC ---
    print("4. Формирование чистых базисов и точный поиск по всему изображению (NCC)...")
    basis_patterns = []
    all_coordinates = []
    reconstructed_components = []

    for k in unique_labels:
        if k == -1:
            continue

        class_mask = (labels == k)
        cluster_patches = patches[class_mask]
        
        # Усредняем патчи кластера. Шум усредняется, паттерн становится идеальным!
        clean_basis = np.mean(cluster_patches, axis=0)
        basis_patterns.append(clean_basis)
        
        # --- МАГИЯ NCC (Normalized Cross-Correlation) ---
        # Мы берем чистый базис и ищем его по всей длине L.
        # Это работает в 100 раз быстрее, чем сравнивать все окна напрямую,
        # потому что мы используем математику сверток.
        
        basis_norm = np.linalg.norm(clean_basis)
        if basis_norm == 0:
            continue
            
        # Числитель: сумма косинусных сверток по каждому из 300 столбцов
        S = np.zeros(L - W + 1, dtype=np.float32)
        for c in range(H):
            S += np.correlate(img_f32[:, c], clean_basis[:, c], mode='valid')
            
        # Знаменатель: норма каждого окна в изображении
        # У нас уже есть энергия (сумма квадратов), берем корень
        window_norms = np.sqrt(energy)
        
        # Само NCC (косинусное сходство от -1 до 1)
        NCC = S / (window_norms * basis_norm + 1e-9)
        
        # Ищем пики, где NCC > порога (например, 0.6 означает 60% совпадение формы)
        # distance=W//2 гарантирует, что мы не найдем один паттерн дважды
        final_coords, _ = find_peaks(NCC, height=ncc_threshold, distance=W//2)
        
        all_coordinates.append(final_coords)
        
        print(f"  Базис {k+1}: найдено {len(final_coords)} вхождений (макс. NCC = {NCC[final_coords].max():.2f})")
        
        # Собираем компонентное изображение
        component_image = np.zeros_like(img_f32, dtype=np.float32)
        for coord in final_coords:
            # Опционально: можно умножить базис на NCC(coord) для подгонки амплитуды
            component_image[coord:coord+W] += clean_basis * NCC[coord]
            
        reconstructed_components.append(component_image)
        
        # Визуализация
        plt.imsave(os.path.join(save_path, f'basis_{k+1}.png'), normalize_for_saving(clean_basis), cmap='gray')
        plt.imsave(os.path.join(save_path, f'component_{k+1}.png'), normalize_for_saving(component_image), cmap='gray')
        
        # Сохраняем график NCC для понимания, как хорошо нашелся паттерн
        plt.figure(figsize=(15, 3))
        plt.plot(NCC, color='black', linewidth=0.1)
        plt.plot(final_coords, NCC[final_coords], "x", color='red', markersize=5)
        plt.axhline(ncc_threshold, color='blue', linestyle='--', label='Threshold')
        plt.title(f'NCC Map for Basis {k+1}')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'ncc_plot_{k+1}.png'), dpi=150)
        plt.close()

    return basis_patterns, all_coordinates, reconstructed_components


# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==========================================
if __name__ == "__main__":
    L, H = 600_000, 300
    W = 200
    
    print("Генерация тестовых данных...")
    np.random.seed(10)
    X = np.random.normal(0, 1, size=(L, H)).astype(np.float32)
    
    # Создаем сложные узоры
    pattern1 = np.sin(np.linspace(0, 10, W))[:, None] * np.random.randn(H) * 5
    pattern2 = np.cos(np.linspace(0, 20, W))[:, None] * np.random.randn(H) * 3
    
    true_coords_1 = np.array([5000, 150000, 340000, 550000])
    true_coords_2 = np.array([80000, 210000, 480000])
    
    for c in true_coords_1: X[c:c+W] += pattern1
    for c in true_coords_2: X[c:c+W] += pattern2
    
    print("Запуск алгоритма (Random Projections + NCC)...")
    bases, coords, components = find_and_decompose_patterns(
        X, 
        pattern_length=W, 
        n_candidates=2000,   # Берем с запасом
        proj_dim=64,         # 64 измерения достаточно для кластеризации
        dbscan_eps=0.2,      # Порог косинусного расстояния (0 - идентичны, 1 - ортогональны)
        ncc_threshold=0.6    # Порог качества совпадения при финальном поиске
    )
    
    print("\n--- Итоги ---")
    if not bases:
        print("Не найдено ни одного базисного паттерна.")
    else:
        for i, c in enumerate(coords):
            print(f"Базис {i+1}: найден {len(c)} раз по координатам {c}...")