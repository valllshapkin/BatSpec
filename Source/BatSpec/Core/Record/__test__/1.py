from pathlib import Path
ScriptDir = Path(__file__).parent
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.SaveIntegral import SaveIntegral


record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
record = correctDC(record)
print(SaveIntegral.Energy(record))

record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
print(SaveIntegral.Energy(record))





