from pathlib import Path
from W.PySide6.QtThemes import Themes

ScriptDir = Path(__file__).parent

AppThemes = Themes(
    SETTING_PATH = ScriptDir / "__assets__" / "ThemeSettings.ini"
)
