from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, makeLog, normalizeFrecZ, makeBinarization, localNormalizeMAD,
    removeEchoWiener, enhanceCurvesGabor, labelWithHDBSCAN, bilateralBlur, filterLabelsCurves
)
from BatSpec.Logic.SpecWorker import makeLog as _makeLog
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.Echo import CausalEchoModel


SPEC = getSpectrogramm()
noise = etalonNoise(SPEC, 5)

DENOISED = subEtalonNoise(SPEC, noise, alpha=1)
updateVisLayer("DENOISED", makeLog(DENOISED))

GABOR = enhanceCurvesGabor(
    DENOISED,
    ksize=17, sigma=1, 
    lambd=5, pre_blur=0
)
updateVisLayer("GABOR", makeLog(GABOR))

WINNER = removeEchoWiener(GABOR, CausalEchoModel(
    decay_rate_base=20, f_ref=40000.0,
    freq_exp=2.0, time_power=20.0,
), eps_factor=1e-5)
updateVisLayer("WINNER", makeLog(WINNER))

ZSPEC = localNormalizeMAD(DENOISED, 
    window_size = (15, 15),
    epsilon = 1e-8
)
updateVisLayer("ZSPEC", makeLog(ZSPEC))

PRODUCT = normalizeFrecZ(WINNER, noise) * ZSPEC
updateVisLayer("PRODUCT", makeLog(ZSPEC))

BINARY = makeBinarization(PRODUCT, trashhold=30)
updateVisLayer("BINARY", BINARY)

CURVES = filterLabelsCurves(BINARY, PRODUCT, threshold=50)
updateVisLayer("CURVES", CURVES)

from BatSpec.Logic.SpecWorker.Morphology.GRAPH import morphSkeleton
SKELETONS = morphSkeleton(CURVES)
updateVisLayer("SKELETONS", SKELETONS)
