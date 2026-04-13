
from BatSpec.QtApp.Shared.Component import Component, ComponentSettings
from BatSpec.QtUp.Settings import SettingField

from pathlib import Path
ScriptDir = Path(__file__).parent


class GrpahSettings(ComponentSettings):
    COMPONENT_DIR = ScriptDir
    
    mode = SettingField[int](default=0, type=int, key="Analize/Record/Graph")
    # 0 -> Raw (original)
    # 1 -> Centered (median = 0)

GRAPH_SETTINGS = GrpahSettings()






