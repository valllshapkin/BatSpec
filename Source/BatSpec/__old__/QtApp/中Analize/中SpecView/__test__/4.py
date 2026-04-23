import numpy as np

from BatSpec.Logic.Echo import TimeDomainEchoModel
from BatSpec.Logic.Loader import loadRecord
from BatSpec.Logic.ConvWindow import load_file_window
from BatSpec.Logic.TransitionWorker import makeSpec, etalonNoise
from BatSpec.Logic.SpecWorker import enhanceCurvesGabor, normalizeFrecZ, subEtalonNoise, removeEchoRichardsonLucy
from BatSpec.Logic.SoundWorker import correctDC
from BatSpec.Logic.VisDebug import RegisterGlobFunc; RegisterGlobFunc()
from pathlib import Path

record = loadRecord(Path("/MainData/Repo/golonchenroppi/BatSpecNew/Resources/r09_23/MYODAS_20230624_004924.wav"))
record = correctDC(record)
name, window = load_file_window(Path("/MainData/Repo/golonchenroppi/BatSpecNew/Tests/SomeProject/storage/windows/BlackmanHarris.STFT.1.py")).popitem()
spec = makeSpec(record, window, overlap=0.7, bins=200)
spec.showLog()

noise = etalonNoise(spec, 5)
spec = subEtalonNoise(spec, noise, alpha=2)
spec.showLog()


model = TimeDomainEchoModel(
    decay_rate_base=20,
    f_ref=40000.0, 
    freq_exp=2.0,
    time_power=20.0,
)


spec = removeEchoRichardsonLucy(spec, model, iterations=20)
# spec.show()
spec = enhanceCurvesGabor(spec, ksize=17, sigma=1, lambd=4, pre_blur=0.5)


# spec = enhanceCurvesGabor(spec, ksize=17, sigma=0.5, lambd=5, pre_blur=1)
# spec = enhanceCurvesGabor(spec, ksize=17, sigma=0.5, lambd=5, pre_blur=1)
# spec = enhanceCurvesGabor(spec, ksize=17, sigma=0.5, lambd=5, pre_blur=1)
# spec = enhanceCurvesGabor(spec, ksize=17, sigma=1, lambd=5, pre_blur=1)
# spec = enhanceCurvesGabor(spec, ksize=17, sigma=1, lambd=5, pre_blur=1)
# spec.show()

normalizeFrecZ(spec, noise).cloneApply(lambda arr: np.where(arr > 32, 1, 0)).show()
# normalizeFrecZ(spec, noise).cloneApply(lambda arr: np.where((arr > 32) & (arr < 64), 1, 0)).show()
# normalizeFrecZ(spec, noise).cloneApply(lambda arr: np.where((arr > 16) & (arr < 32), 1, 0)).show()
# normalizeFrecZ(spec, noise).cloneApply(lambda arr: np.where((arr > 8) & (arr < 16), 1, 0)).show()
# normalizeFrecZ(spec, noise).cloneApply(lambda arr: np.where((arr > 0) & (arr < 8), 1, 0)).show()








