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

# M = adaptiveMahalanobisTexture(SPEC,
#     patch_size = 3, 
#     block_time = 256, 
#     block_freq = 64
# )

M = extractRidgesFrangi(
    SPEC,     
    sigmas = (1, 2, 3), 
    alpha = 0.5, 
    beta = 0.5, 
    gamma = 0.0001
)

updateVisLayer("M", makeLog(M))

BINARY = makeBinarization(M, trashhold=0.0001)
updateVisLayer("BINARY", BINARY)

DENOISED = subEtalonNoise(SPEC, noise, alpha=1)
updateVisLayer("DENOISED", DENOISED)




# GABOR = enhanceCurvesGabor(DENOISED, ksize=17, sigma=1, lambd=5, pre_blur=0)
# updateVisLayer("GABOR", GABOR)

# RICHLUCY = removeEchoRichardsonLucy(GABOR, TimeDomainEchoModel(
#     decay_rate_base=400, f_ref=40000.0,
#     freq_exp=2.0, time_power=20.0,
# ), iterations=40)
# updateVisLayer("RICHLUCY", RICHLUCY)

# GABOR2 = enhanceCurvesGabor(RICHLUCY, ksize=17, sigma=0.5, lambd=4, pre_blur=0)
# updateVisLayer("GABOR2", GABOR2)

# LOG = makeLog(GABOR2)
# updateVisLayer("LOG", LOG)

# NORM = normalizeFrecZ(GABOR2, noise)
# updateVisLayer("NORM", NORM)

# BINARY = makeBinarization(NORM, trashhold=15)
# updateVisLayer("BINARY", BINARY)

