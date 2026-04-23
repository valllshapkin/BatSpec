from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, makeLog, makeLabelsWatershed, normalizeFrecZ, bilateralBlur, gaussianBlur
)
from BatSpec.Logic.TransitionWorker import integrateFreq
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from scipy.signal import find_peaks

SPEC = getSpectrogramm()
noise = etalonNoise(SPEC, 5)

DENOISED = subEtalonNoise(SPEC, noise, alpha=1)
updateVisLayer("DENOISED", makeLog(DENOISED))

spec = DENOISED.value




