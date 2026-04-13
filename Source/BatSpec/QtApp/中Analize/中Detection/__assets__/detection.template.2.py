from typing import Any

from numpy.typing import NDArray

from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.Functions import SpecFunc

from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, removeEchoRichardsonLucy, makeLog, enhanceCurvesGabor, normalizeFrecZ, makeBinarization,
    filterLabelsCurves
)
from BatSpec.Logic.SpecWorker.Morphology.GRAPH import morphClose, morphVerticalConnect
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.Echo import AsymmetricEchoModel

SPEC = getSpectrogramm()
noise = etalonNoise(SPEC, 5)

DENOISED = subEtalonNoise(SPEC, noise, alpha=1)
updateVisLayer("DENOISED", DENOISED)

GABOR = enhanceCurvesGabor(
    DENOISED,
    ksize=17, sigma=1, 
    lambd=5, pre_blur=0
)
updateVisLayer("GABOR", makeLog(GABOR))

RICHLUCY = removeEchoRichardsonLucy(GABOR, AsymmetricEchoModel(
    decay_rate_base=100, f_ref=40000.0,
    freq_exp=2.0, time_power=20.0,
), iterations=40)
updateVisLayer("RICHLUCY", makeLog(RICHLUCY))

BINARY = makeBinarization(normalizeFrecZ(RICHLUCY, noise), trashhold=10)
updateVisLayer("BINARY", BINARY)

CURVES = filterLabelsCurves(BINARY)
updateVisLayer("CURVES", CURVES)

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

def extract_blobs(binary_mask: SpecFunc, min_size: int = 10) -> list[np.ndarray]:
    """Извлекает все связные компоненты (blobs) из бинарной маски."""
    labeled_mask, num_features = ndimage.label(binary_mask.matrix)
    if num_features == 0:
        return []
    
    blobs = []
    props = regionprops(labeled_mask)
    
    for prop in props:
        if prop.area >= min_size:
            # bbox (min_row, min_col, max_row, max_col)
            min_r, min_c, max_r, max_c = prop.bbox
            blob_img = (labeled_mask[min_r:max_r, min_c:max_c] == prop.label)
            blobs.append(blob_img.astype(np.uint8))
            
    return blobs

import hdbscan
from skimage.feature import match_template
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def _compute_blob_similarity(blob1: np.ndarray, blob2: np.ndarray) -> float:
    """
    Надежный воркер для расчета сходства двух блобов.
    """
    # Шаблон должен быть меньше или равен по ОБОИМ измерениям
    if blob1.shape[0] <= blob2.shape[0] and blob1.shape[1] <= blob2.shape[1]:
        template, image = blob1, blob2
    elif blob2.shape[0] <= blob1.shape[0] and blob2.shape[1] <= blob1.shape[1]:
        template, image = blob2, blob1
    else:
        # Блобы имеют несравнимые пропорции (например, один высокий и узкий, другой низкий и широкий).
        # Их сходство в данном контексте равно нулю.
        return 0.0
        
    res = match_template(image.astype(float), template.astype(float))
    return np.max(res)


def build_binary_templates(blobs: list[np.ndarray]) -> list[np.ndarray]:
    """Кластеризует бинарные объекты и создает усредненные шаблоны."""
    if not blobs:
        return []
        
    # --- 1. Фильтруем кандидатов по размеру (берем "золотую середину") ---
    areas = [np.sum(b) for b in blobs]
    p25, p75 = np.percentile(areas, [25, 75])
    good_candidates = [b for b, area in zip(blobs, areas) if p25 <= area <= p75]
    n = len(good_candidates)
    if n < 10: # Если хороших кандидатов мало, используем всех
        good_candidates = blobs
        n = len(blobs)
        if n == 0: return []

    print(f"Отобрано {n} кандидатов для словаря.")
    
    # --- 2. Строим матрицу сходства (параллельно) ---
    corr_matrix = np.eye(n, dtype=np.float32)
    tasks = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_compute_blob_similarity, good_candidates[i], good_candidates[j]) for i, j in tasks]
        for k, future in enumerate(tqdm(futures, desc="Матрица сходства блобов")):
            i, j = tasks[k]
            corr = future.result()
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            
    # --- 3. Кластеризация ---
    distance_matrix = np.clip(1.0 - corr_matrix, 0.0, 1.0).astype(np.float64)
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5, min_samples=2, cluster_selection_method='leaf')
    labels = clusterer.fit_predict(distance_matrix)
    
    # --- 4. Усреднение для создания эталонов ---
    templates = []
    for cluster_id in set(labels):
        if cluster_id == -1: continue
        
        indices = np.where(labels == cluster_id)[0]
        cluster_blobs = [good_candidates[i] for i in indices]
        
        # Усредняем (аналогично createClusterBasis)
        max_h = max(b.shape[0] for b in cluster_blobs)
        max_w = max(b.shape[1] for b in cluster_blobs)
        canvas = np.zeros((max_h, max_w), dtype=np.float32)
        
        for blob in cluster_blobs:
            h, w = blob.shape
            start_h, start_w = (max_h - h) // 2, (max_w - w) // 2
            canvas[start_h:start_h+h, start_w:start_w+w] += blob
        
        avg_blob = canvas / len(cluster_blobs)
        # Снова делаем его бинарным по порогу 0.5
        template = (avg_blob > 0.5).astype(np.uint8)
        templates.append(template)
        
    print(f"Создано {len(templates)} идеальных шаблонов.")
    return templates

from skimage.feature import peak_local_max

def reconstruct_mask_from_templates(
    continuous_spec: SpecFunc, 
    templates: list[np.ndarray], 
    threshold: float = 0.7
) -> SpecFunc:
    """Ищет шаблоны на непрерывной спектрограмме и собирает новую чистую маску."""
    new_mask_matrix = np.zeros_like(continuous_spec.matrix, dtype=np.uint8)
    
    for template in tqdm(templates, desc="Реконструкция маски"):
        # Ищем совпадения на НЕПРЕРЫВНОЙ спектрограмме
        print(continuous_spec.matrix.shape, template.astype(float).shape)
        result = match_template(continuous_spec.matrix, template.astype(float))
        
        # Находим пики (центры совпадений)
        peaks = peak_local_max(result, min_distance=5, threshold_abs=threshold)
        
        # "Печатаем" шаблон на новой маске
        h, w = template.shape
        for r, c in peaks:
            # r, c - это левый верхний угол, где нашлось совпадение
            if r + h > new_mask_matrix.shape[0] or c + w > new_mask_matrix.shape[1]:
                continue
            
            # np.logical_or правильно обрабатывает перекрытия
            region = new_mask_matrix[r:r+h, c:c+w]
            np.logical_or(region, template, out=region)

    # Финальное небольшое "закрытие" для соединения близких частей
    closed_mask = ndimage.binary_closing(new_mask_matrix, iterations=2)
            
    return SpecFunc(closed_mask.astype(np.uint8), continuous_spec.freq, continuous_spec.time)

blobs = extract_blobs(CURVES.value, min_size=20)
templates = build_binary_templates(blobs)
clean_mask = reconstruct_mask_from_templates(DENOISED.value, templates, threshold=0.7)
updateVisLayer("Очищенная_маска_v3", clean_mask)








