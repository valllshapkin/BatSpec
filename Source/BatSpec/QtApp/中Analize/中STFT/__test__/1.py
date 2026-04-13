from BatSpec.QtApp.Logic import application
import os
from pathlib import Path
ScriptDir = Path(__file__).parent
os.chdir(ScriptDir)

with application():
    from BatSpec.QtApp.中Analize.中STFT.Widget import StftModelTab
    # from BatSpec.QtApp.中Project.Logic import PROJECT_STATE
    # PROJECT_STATE.loadProject(ScriptDir / "SomeProject")
    window = StftModelTab()
    window.show()
