from PySide6.QtWidgets import QWidget
from PySide6 import QtCore
from BatSpec.QtUp.Settings import Settings as SettingsProtocol, SettingField
from typing import Any, Self, TYPE_CHECKING
import qt_themes # type: ignore
from BatSpec.Python import delete_init
import darkdetect # type: ignore
import threading
from contextlib import nullcontext as Block

DEBUG = True
THEMES = qt_themes.get_themes()

class Settings(QtCore.QObject, SettingsProtocol if TYPE_CHECKING else object):
    SETTING_PATH: str

    dark = SettingField[str](default="catppuccin_mocha", type=str, key="Theme/Dark")
    light = SettingField[str](default="github_light", type=str, key="Theme/Light")
    one = SettingField[str](default="catppuccin_mocha", type=str, key="Theme/One")
    twoMode = SettingField[bool](default=True, type=bool, key="Theme/TwoMode")
    dark_updated = QtCore.Signal(str)
    light_updated = QtCore.Signal(str)
    one_updated = QtCore.Signal(str)
    twoMode_updated = QtCore.Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self, "SETTING_PATH"): raise NotImplementedError()

        self.settings = QtCore.QSettings(
            self.SETTING_PATH, 
            QtCore.QSettings.Format.IniFormat
        )

    def set_one_and_apply(self, theme: str):
        if self.twoMode: raise RuntimeError()
        self.one = theme
        self.one_updated.emit(theme)
        update_one_theme()
        
    def set_dark_and_apply(self, theme: str):
        if not self.twoMode: raise RuntimeError()
        self.dark = theme
        self.dark_updated.emit(theme)
        update_two_theme()

    def set_light_and_apply(self, theme: str):
        if not self.twoMode: raise RuntimeError()
        self.light = theme
        self.light_updated.emit(theme)
        update_two_theme()
    
    def change_mod(self, mode: bool):
        if mode ^ self.twoMode:
            self.twoMode = mode
            self.twoMode_updated.emit(mode)
            update_any()

    
SETTINGS: Settings

@delete_init
class Trigger(QWidget if TYPE_CHECKING else object):
    def __init_subclass__(cls):

        old_changeEvent = cls.changeEvent
        def new_changeEvent(self: Self, event: QtCore.QEvent):
            if event.type() == QtCore.QEvent.Type.PaletteChange:
                self.onThemeChange()
            old_changeEvent(self, event)            
        cls.changeEvent = new_changeEvent

        old_init = cls.__init__
        def new_init(self: Self, *args: Any, **kwargs: Any):
            old_init(self, *args, **kwargs)
            self.onThemeChange()
        cls.__init__ = new_init

        return super().__init_subclass__()

    def onThemeChange(self):
        pass

def set_theme(theme: str, style: str | None = 'fusion') -> None:
    qt_themes.set_theme(THEMES.get(theme), style=style)

def get_curent_theme() -> qt_themes.Theme:
    theme = qt_themes.get_theme()
    if not theme: raise RuntimeError()
    return theme

def update_one_theme():
    if SETTINGS.twoMode: raise RuntimeError()
    set_theme(SETTINGS.one)

def update_two_theme():
    if not SETTINGS.twoMode: raise RuntimeError()
    theme = SETTINGS.dark if System.isDark() else SETTINGS.light
    set_theme(theme)

def update_any():
    update_two_theme() if SETTINGS.twoMode else update_one_theme()

class _System(QtCore.QObject):
    onChanged = QtCore.Signal(str)

    def isDark(self):
        return darkdetect.isDark()

    def isLight(self):
        return darkdetect.isLight()

    def start_listener(self):
        t = threading.Thread(target=darkdetect.listener, args=(self.onChanged.emit,), daemon=True) # type: ignore
        t.start()

System = _System()

def two_theme_wrapper(_: str):
    if not SETTINGS.twoMode: return
    set_theme(SETTINGS.dark if System.isDark() else SETTINGS.light)


def initThemes():
    System.start_listener()
    System.onChanged.connect(two_theme_wrapper)
    update_any()

    debug_first_theme() if DEBUG else None
    debug_theme_timer() if DEBUG else None
    debug_system_theme() if DEBUG else None
    debug_show_all_themes() if DEBUG else None


with Block("DEBUG"):
    def debug_show_all_themes():
        print(*(f"[THEMES]: {i}" for i in THEMES.keys()), sep="\n")

    def debug_first_theme():
        set_theme("catppuccin_mocha") if System.isDark() else set_theme("github_light")

    def debug_system_theme():
        def log(theme: str):
            print(f"System Theme Change To {theme}")
        System.onChanged.connect(log)

    def debug_theme_timer():
        import random
        from PySide6 import QtCore, QtWidgets
        timer = QtCore.QTimer(parent=QtWidgets.QApplication.instance())
        timer.setInterval(1000)

        @timer.timeout.connect
        def _():
            if not SETTINGS.twoMode: return SETTINGS.set_one_and_apply(random.choice(tuple(
                k for k, v in THEMES.items() if v.is_dark_theme() ^ System.isLight()
            ))) 

            SETTINGS.set_light_and_apply(random.choice(tuple(
                k for k, v in THEMES.items() if not v.is_dark_theme()
            ))) if System.isLight() else SETTINGS.set_dark_and_apply(random.choice(tuple(
                k for k, v in THEMES.items() if v.is_dark_theme()
            )))
            
        timer.start()