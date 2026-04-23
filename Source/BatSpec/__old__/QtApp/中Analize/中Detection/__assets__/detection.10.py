from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, makeLog, normalizeFrecZ, makeBinarization,
    gradientSquared, removeEchoWiener
)
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.Echo import TimeDomainEchoModel


SPEC = getSpectrogramm()
noise = etalonNoise(SPEC, 5)

DENOISED = subEtalonNoise(SPEC, noise, alpha=1)
updateVisLayer("DENOISED", DENOISED)

WINNER = removeEchoWiener(DENOISED, TimeDomainEchoModel(
    decay_rate_base=20, f_ref=40000.0,
    freq_exp=2.0, time_power=20.0,
), eps_factor=1e-2)
updateVisLayer("WINNER", makeLog(WINNER))

GRAD = gradientSquared(WINNER, blur_sigma=1)
updateVisLayer("GRAD", makeLog(GRAD))

BINARY = makeBinarization(normalizeFrecZ(WINNER, noise), trashhold=5)
updateVisLayer("BINARY", BINARY)




