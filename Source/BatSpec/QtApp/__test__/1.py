from BatSpec.QtApp.Logic import application
import os
from pathlib import Path
ScriptDir = Path(__file__).parent
os.chdir(ScriptDir)

with application():
    from BatSpec.QtApp.Widget import MainWindow
    window = MainWindow()
    window.show()
