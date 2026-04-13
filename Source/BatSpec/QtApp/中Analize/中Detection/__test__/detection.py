# import numpy as np

# from BatSpec.API import getSpectrogramm, updateVisLayer
# from BatSpec.Logic.Echo import TimeDomainEchoModel
# from BatSpec.Logic.TransitionWorker import etalonNoise
# from BatSpec.Logic.SpecWorker import (
#     enhanceCurvesGabor, normalizeFrecZ, subEtalonNoise, removeEchoRichardsonLucy,
#     makeBinarization
# )

# SPEC = getSpectrogramm()

# noise = etalonNoise(SPEC, 5)

# spec = subEtalonNoise(SPEC, noise, alpha=2)
# updateVisLayer({"Denoised": spec})

# model = TimeDomainEchoModel(
#     decay_rate_base=20,
#     f_ref=40000.0, 
#     freq_exp=2.0,
#     time_power=20.0,
# )

# spec = removeEchoRichardsonLucy(spec, model, iterations=20)
# updateVisLayer({"RichardsonLucy": spec})

# spec = enhanceCurvesGabor(spec, ksize=17, sigma=1, lambd=4, pre_blur=0.5)
# updateVisLayer({"CurvesGabor": spec})

# spec = normalizeFrecZ(spec, noise)
# updateVisLayer({"NomalizeZ": spec})

# spec = makeBinarization(spec, trashhold=32)
# updateVisLayer({"Binarization": spec})

