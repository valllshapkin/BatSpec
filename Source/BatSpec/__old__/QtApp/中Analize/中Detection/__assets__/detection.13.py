from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, makeLog, normalizeFrecZ, makeBinarization, enhanceCurvesGabor, 
    gradientSquared, removeEchoWiener, localNormalizeMAD, gaussianBlur
)
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.Echo import CausalEchoModel

SPEC = getSpectrogramm()
noise = etalonNoise(SPEC, 5)

DENOISED = subEtalonNoise(SPEC, noise, alpha=1)
updateVisLayer("DENOISED", DENOISED)

GRAD = gradientSquared(SPEC, blur_sigma=1)
updateVisLayer("GRAD", GRAD)

# ZSPEC = localNormalizeMAD(DENOISED, 
#     window_size = (15, 15),
#     epsilon = 1e-8
# )
# updateVisLayer("ZSPEC", makeLog(ZSPEC))

BINARY = makeBinarization(normalizeFrecZ(GRAD, noise), trashhold=5)
updateVisLayer("BINARY", BINARY)




