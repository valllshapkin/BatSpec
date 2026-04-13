import numpy as np

from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, makeLog, makeLabelsWatershed, normalizeFrecZ, bilateralBlur, gaussianBlur
)
from BatSpec.Logic.TransitionWorker import integrateFreq
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.Calls import BatTemplateGenerator, applyPatternFilter
from scipy.signal import find_peaks

SPEC = getSpectrogramm()
noise = etalonNoise(SPEC, 5)

DENOISED = subEtalonNoise(SPEC, noise, alpha=1)
updateVisLayer("DENOISED", makeLog(DENOISED))


spec = DENOISED.value
generator = BatTemplateGenerator(spec.dt, spec.df, blur_sigma=1.5)
template_fm = generator.generate_FM()

debug = np.zeros_like(template_fm.matrix)
debug[9:-9, 9:-9] = 1
template_fm.matrix = debug
updateVisLayer("template_fm", template_fm)

score_map_spec = applyPatternFilter(spec, template_fm)
updateVisLayer("score_map_spec", score_map_spec)

# 3. Фильтруем всё, что ниже абсолютного порога (например, 0.7)
# Применяем бинаризацию, чтобы получить чистую маску
threshold = 0.7
clean_binary_matrix = (score_map_spec.matrix > threshold).astype(np.uint8)
final_spec = SpecFunc(clean_binary_matrix, spec.freq, spec.time)
updateVisLayer("final_spec", final_spec)

