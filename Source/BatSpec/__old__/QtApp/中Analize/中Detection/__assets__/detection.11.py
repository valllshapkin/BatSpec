from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, makeLog, normalizeFrecZ, makeBinarization, localNormalizeMAD,
    removeEchoWiener, enhanceCurvesGabor, labelWithHDBSCAN
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

BINARY = makeBinarization(normalizeFrecZ(WINNER, noise) * ZSPEC, trashhold=5)
updateVisLayer("BINARY", BINARY)

LABELS = labelWithHDBSCAN(
    bin_spec=BINARY, 
    spec=makeLog(DENOISED), 
    min_cluster_size=20, 
    epsilon=10.0, 
    scale_weights=(4, 1, 0.5)
)
updateVisLayer("LABELS", LABELS)




