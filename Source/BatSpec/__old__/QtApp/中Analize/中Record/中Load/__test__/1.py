from BatSpec.QtApp.Logic import application

from pathlib import Path
ScriptDir = Path(__file__).parent

with application():
    from BatSpec.QtApp.中Analize.中Record.中Load.Widget import Load
    from BatSpec.QtApp.中Project.Logic import PROJECT_STATE
    PROJECT_STATE.loadProject(ScriptDir / "SomeProject")
    window = Load()
    window.show()
