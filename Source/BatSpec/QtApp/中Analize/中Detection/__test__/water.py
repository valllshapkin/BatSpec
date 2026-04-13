import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# --- Генерация вашей тепловой карты (зарядов) ---
x, y = np.indices((100, 100))
image = np.exp(-((x - 30)**2 + (y - 30)**2) / (2 * 15**2)) + \
        np.exp(-((x - 70)**2 + (y - 70)**2) / (2 * 15**2))


# ==========================================
# ТОТ САМЫЙ АЛГОРИТМ (3 шага)
# ==========================================

# 1. Задаем область: отсекаем всё, что меньше 0.3 (ваша граница снаружи)
threshold = 0.3
mask = image >= threshold

# 2. Находим координаты всех "зарядов" (пиков) СТРОГО внутри нашей маски
# min_distance защита от шума (не искать два пика в 5 пикселях друг от друга)
peaks_coords = peak_local_max(image, min_distance=5, labels=mask)

# Создаем разметку для пиков (1, 2, 3...)
markers = np.zeros_like(image, dtype=np.int32)
for i, (r, c) in enumerate(peaks_coords):
    markers[r, c] = i + 1

# 3. Разделяем область математически строго по градиенту (Watershed)
# Алгоритм "течет" от пиков (-image) вниз до тех пор, пока не упрется в границы маски (0.3).
# В месте встречи двух полей (на седле 0.5) он автоматически ставит границу.
labels = watershed(-image, markers, mask=mask)

# ==========================================


# --- Визуализация ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image, cmap='hot')
axes[0].set_title('Исходное тепловое поле (Плотность)')
axes[0].contour(image, levels=[threshold], colors='cyan', linestyles='dashed') # Показываем линию 0.3

# Раскрашиваем лейблы
cmap = plt.cm.get_cmap('Set1', len(peaks_coords))
axes[1].imshow(image, cmap='gray') # Серый фон
# Показываем разделенные зоны. Там где labels == 0 (фон), делаем прозрачным
axes[1].imshow(np.ma.masked_where(labels == 0, labels), cmap=cmap, alpha=0.8)
axes[1].set_title(f'Разделение по числу пиков (Мин. порог {threshold})')

plt.tight_layout()
plt.show()