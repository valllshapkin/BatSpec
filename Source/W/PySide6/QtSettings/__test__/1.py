import sys
from enum import Enum
from PySide6 import QtCore, QtGui # type: ignore
from W.PySide6.QtSettings import Field, EnumAdapter, LocalesAdapter, QColorAdapter, JsonAdapter

# 
class ThemeType(Enum):
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


from pathlib import Path
ScriptDir = Path(__file__).parent

class AppConfig(QtCore.QObject):
    settings = QtCore.QSettings(str(ScriptDir / "test.ini"), QtCore.QSettings.Format.IniFormat)

    # 1. Простые типы (как раньше)
    window_width, S_width = Field(default=800, val_type=int)
    
    # 2. ENUM (Строгая типизация, сломать файл test.ini невозможно)
    theme, S_theme = Field(
        default=ThemeType.DARK, 
        adapter=EnumAdapter(ThemeType, fallback=ThemeType.DARK),
        sig_type=object
    )

    # 3. Списки объектов (локали)
    locales, S_locales = Field(
        default=[QtCore.QLocale('en_US')], 
        adapter=LocalesAdapter()
    )

    # 4. Объекты Qt (Цвета)
    accent_color, S_color = Field(
        default=QtGui.QColor("#ff0000"),
        adapter=QColorAdapter(fallback=QtGui.QColor("#ff0000"))
    )

    # 5. Сложные структуры данных (Словари/JSON)
    # Сохранится в ini как: user_data={"id": 1, "role": "admin"}
    user_data, S_user_data = Field(
        default={"id": 0, "role": "guest"},
        adapter=JsonAdapter()
    )


app = QtCore.QCoreApplication(sys.argv)
QtCore.QCoreApplication.setOrganizationName("MyCompany")
QtCore.QCoreApplication.setApplicationName("MyApp")

config = AppConfig()

# Подписки
config.S_width.connect(lambda w: print(f"📐 Ширина: {w}"))
config.S_theme.connect(lambda t: print(f"🎨 Тема: {t.name}"))
config.S_locales.connect(lambda l: print(f"🌍 Локали: {[x.name() for x in l]}"))
config.S_color.connect(lambda c: print(f"🖌 Цвет: {c.name()}"))
config.S_user_data.connect(lambda d: print(f"👤 Данные: {d}"))

print("\n--- Задаем новые значения ---")
config.window_width = 1920
config.theme = ThemeType.LIGHT  # Передаем Enum, а в INI запишется строка "LIGHT"
config.locales = [QtCore.QLocale('ru_RU')]
config.accent_color = QtGui.QColor("#00ff00")
config.user_data = {"id": 42, "role": "admin"}