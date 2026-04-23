from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, makeLog, normalizeFrecZ, makeBinarization, extractPatternsNCC,
    gradientSquared, removeEchoWiener, enhanceCurvesGabor, labelWithHDBSCAN, splitDominantPattern
)
from BatSpec.Logic.SpecWorker import makeLog as _makeLog
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.Echo import CausalEchoModel

# Получаем исходную спектрограмму и шум
SPEC = getSpectrogramm()
noise = etalonNoise(SPEC, 5)

# Очищаем от шума
DENOISED = subEtalonNoise(SPEC, noise, alpha=1)
updateVisLayer("DENOISED", makeLog(DENOISED))

# Итеративное выделение паттернов
patterns = []          # список выделенных доминантных паттернов
residual = DENOISED    # текущий остаток, начинаем с очищенной спектрограммы

for i in range(10):
    # Разделяем остаток на доминантный паттерн и новый остаток
    dominant, residual = splitDominantPattern(
        residual,
        n_candidates=300,
        dbscan_eps=0.2,
        ncc_threshold=0.5,
        min_pattern_len=30,
        max_pattern_len=800
    )
    patterns.append(dominant)
    
    # Визуализируем выделенный паттерн (логарифмическая шкала для наглядности)
    updateVisLayer(f"PATTERN_{i+1}", makeLog(dominant))
    
# После цикла можно посмотреть финальный остаток
updateVisLayer("FINAL_RESIDUAL", makeLog(residual))

# Если нужно, patterns[0] — первый (самый сильный) паттерн, patterns[1] — второй и т.д.