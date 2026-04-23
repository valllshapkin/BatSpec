from typing import Any

from numpy.typing import NDArray

from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.Functions import SpecFunc
# from BatSpec.Logic.SpecWorker import (
#     normalizeFrecZ, removeEchoRichardsonLucy, enhanceCurvesGabor, subEtalonNoise, makeBinarization, makeLog
# )
from BatSpec.Logic.SpecWorker import adaptiveMahalanobisTexture
from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, removeEchoRichardsonLucy, makeLog, enhanceCurvesGabor, normalizeFrecZ, makeBinarization,
    harmonicPercussive, adaptiveMahalanobisTexture, mahalanobisEtalonTexture, extractRidgesFrangi
)
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.Echo import TimeDomainEchoModel


SPEC = getSpectrogramm()
noise = etalonNoise(SPEC, 5)

DENOISED = subEtalonNoise(SPEC, noise, alpha=1)
updateVisLayer("DENOISED", DENOISED)

# RICHLUCY = removeEchoRichardsonLucy(DENOISED, TimeDomainEchoModel(
#     decay_rate_base=200, f_ref=40000.0,
#     freq_exp=2.0, time_power=20.0,
# ), iterations=40)
# updateVisLayer("RICHLUCY", makeLog(RICHLUCY))

FRANGI = extractRidgesFrangi(
    DENOISED,     
    sigmas = (1, 2), 
    alpha = 0.1, 
    beta = 0.9, 
    gamma = 0.0001
)
updateVisLayer("FRANGI", makeLog(FRANGI))

# MULT = FRANGI.cache.get_or_create_node(
#     f"{FRANGI.key} * {DENOISED.key}", 
#     lambda: normalizeFrecZ(FRANGI, noise).value * normalizeFrecZ(DENOISED, noise).value 
# )
# updateVisLayer("MULT", makeLog(MULT))

BINARY = makeBinarization(normalizeFrecZ(FRANGI, noise), trashhold=10)
updateVisLayer("BINARY", BINARY)




