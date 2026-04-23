
import numpy as np

from BatSpec.QtApp.Logic import application
from BatSpec.Logic.Loader import loadRecord
from BatSpec.QtApp.Logic import application
from BatSpec.Logic.Loader import loadRecord
from BatSpec.Logic.ConvWindow import Window
from BatSpec.Logic.TransitionWorker import makeSpec, etalonNoise, integrateFreq
from BatSpec.Logic.SpecWorker import subEtalonNoise
from BatSpec.Logic.SoundWorker import correctDC
from scipy.signal.windows import hann

import pyqtgraph as pg
from pathlib import Path
ScriptDir = Path(__file__).parent

with application():
    # Импортируем нашу обертку
    from BatSpec.QtApp.中Analize.中SpecView.Widget import SpecViewWidget
    
    main = SpecViewWidget()
    
    spec = makeSpec(
        correctDC(loadRecord(ScriptDir / "MYODAS_20230624_005134.wav")),
        Window(hann, 0.001, norm="energy"), 
        overlap=0.5, bins=200
    )
    noise = etalonNoise(spec, percentile=5)
    

    main.setMinimapFunc(integrateFreq(subEtalonNoise(spec, noise)))
    main.setSpecFunc(spec.cloneApply(lambda arr: 20 * np.log10(np.clip(arr, 1e-9, None))))

    
    main.show()