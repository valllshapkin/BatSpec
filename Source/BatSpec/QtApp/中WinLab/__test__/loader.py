from pathlib import Path
from BatSpec.QtApp.中WinLab.Logic import load_file_window

ScriptDir = Path(__file__).parent
win = {}
win.update(load_file_window(ScriptDir / "BlackmanHarris.STFT.2.py"))
win.update(load_file_window(ScriptDir / "Hann.STFT.2.py"))
win.update(load_file_window(ScriptDir / "Hann.SYM.2.py"))
print(win)