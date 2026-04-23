from typing import Any

from numpy.typing import NDArray

from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.Functions import SpecFunc

from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, removeEchoRichardsonLucy, makeLog, enhanceCurvesGabor, normalizeFrecZ, makeBinarization,
    harmonicPercussive, adaptiveMahalanobisTexture, mahalanobisEtalonTexture, extractRidgesFrangi, makeLabels, filterLabelsHDBSCAN
)
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.Echo import TimeDomainEchoModel

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

RICHLUCY = removeEchoRichardsonLucy(GABOR, TimeDomainEchoModel(
    decay_rate_base=200, f_ref=40000.0,
    freq_exp=2.0, time_power=20.0,
), iterations=40)
updateVisLayer("RICHLUCY", makeLog(RICHLUCY))

BINARY = makeBinarization(normalizeFrecZ(RICHLUCY, noise), trashhold=10)
updateVisLayer("BINARY", BINARY)

LABLES = makeLabels(BINARY)
updateVisLayer("LABLES", LABLES)

HDBSCAN = filterLabelsHDBSCAN(LABLES, DENOISED, min_cluster_size=3, retention_percentile=20)
updateVisLayer("HDBSCAN", HDBSCAN)



