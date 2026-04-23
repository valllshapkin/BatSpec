from BatSpec.QtUp import Themes
from pathlib import Path
ScriptDir = Path(__file__).parent

class Settings(Themes.Settings):
    SETTING_PATH = str(ScriptDir / "__assets__" / "ThemeSettings.ini")

Themes.DEBUG = False
Themes.SETTINGS = Settings()