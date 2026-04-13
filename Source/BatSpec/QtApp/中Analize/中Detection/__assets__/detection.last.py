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

CLOSE = morphClose(BINARY, radius=2)
updateVisLayer("CLOSE", CLOSE)

VERTICAL = morphVerticalConnect(CLOSE)
updateVisLayer("VERTICAL", VERTICAL)

CURVES = filterLabelsCurves(VERTICAL)
updateVisLayer("CURVES", CURVES)

