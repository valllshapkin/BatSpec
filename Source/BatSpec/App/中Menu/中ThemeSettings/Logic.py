from pathlib import Path
from BatSpec.QtUp.Themes import Themes

ScriptDir = Path(__file__).parent

# Единый глобальный инстанс тем для всего приложения
AppThemes = Themes(
    SETTING_PATH = ScriptDir / "__assets__" / "ThemeSettings.ini"
)
