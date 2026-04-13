from BatSpec.QtApp.Logic import application

from pathlib import Path
ScriptDir = Path(__file__).parent


with application():
    from BatSpec.QtApp.中Analize.中Record.中MetaData.中Info.Widget import Info
    from BatSpec.QtApp.中Analize.中Record.Logic import RECORD_STATE
    RECORD_STATE.setRecordFile(ScriptDir / "MYODAS_20230624_005134.wav")

    window = Info()
    window.show()
