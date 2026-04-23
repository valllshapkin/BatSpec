import numpy as np

from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.SpecWorker import makeLog as _makeLog
from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, makeLog, makeLabelsWatershed, normalizeFrecZ, bilateralBlur, gaussianBlur
)
from BatSpec.Logic.TransitionWorker import applyClusterFilters, buildCorrelationMatrix, clusterCorrelationMatrix, integrateFreq, findPeaks, extractCalls
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.Calls import BatTemplateGenerator, applyPatternFilterTime, enhanceCallsWithEtalons

SPEC = getSpectrogramm()
noise = etalonNoise(SPEC, 5)

DENOISED = subEtalonNoise(SPEC, noise, alpha=1)
updateVisLayer("DENOISED", makeLog(DENOISED))

spec = DENOISED.value
power = integrateFreq(spec)
peaks = findPeaks(power, width=0.0005, distance=0.0005)
corr_matrix, calls = buildCorrelationMatrix(peaks, spec, width_multiplier=4.0)
labels = clusterCorrelationMatrix(corr_matrix, min_cluster_size=3)

unique_clusters = set(labels)
num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

print(f"Найдено типов криков (кластеров): {num_clusters}")
print(f"Всего уникальных меток (включая шум -1): {len(unique_clusters)}")
if -1 in unique_clusters:
    print(f"  - Из них шумовые точки (метка -1): {labels.tolist().count(-1) if hasattr(labels, 'tolist') else list(labels).count(-1)}")
    print(f"  - Кластеры без шума: {num_clusters}")
else:
    print("  - Шумовые точки (метка -1) отсутствуют")
print(f"Список найденных кластеров (без -1): {sorted([c for c in unique_clusters if c != -1])}")

# 1. Получаем чистую карту вероятностей (где 1.0 - 100% мышь, 0.0 - нет мыши)
enhanced_spec = enhanceCallsWithEtalons(
    main_spec=spec, 
    peaks=peaks, 
    calls=calls, 
    labels=labels, 
    alpha=0.9  # "Подмешаем" 40% от идеальной формы
)

updateVisLayer("Усиленные_крики_v2", _makeLog(enhanced_spec))