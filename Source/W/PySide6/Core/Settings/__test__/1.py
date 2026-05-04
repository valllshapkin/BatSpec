import sys
from enum import Enum
from pathlib import Path
from PySide6 import QtCore, QtGui # type: ignore
from W.PySide6.Core.Settings import Field, EnumAdapter, LocalesAdapter, QColorAdapter, JsonAdapter

class ThemeType(Enum):
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"

ScriptDir = Path(__file__).parent

class AppConfig(QtCore.QObject):
    settings = QtCore.QSettings(str(ScriptDir / "test.ini"), QtCore.QSettings.Format.IniFormat)
    window_width, S_width = Field(default=800, val_type=int)
    theme, S_theme = Field(default=ThemeType.DARK, adapter=EnumAdapter(ThemeType, fallback=ThemeType.DARK), sig_type=object)
    locales, S_locales = Field(default=[QtCore.QLocale('en_US')], adapter=LocalesAdapter())
    accent_color, S_color = Field(default=QtGui.QColor("#ff0000"), adapter=QColorAdapter(fallback=QtGui.QColor("#ff0000")))
    user_data, S_user_data = Field(default={"id": 0, "role": "guest"}, adapter=JsonAdapter())

if __name__ == "__main__":
    app = QtCore.QCoreApplication(sys.argv)
    config = AppConfig()

    config.S_width.connect(lambda w: print(f"Width: {w}"))
    config.S_theme.connect(lambda t: print(f"Theme: {t.name}"))
    
    config.window_width = 1920
    config.theme = ThemeType.LIGHT
