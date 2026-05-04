from pathlib import Path
from typing import Any
import random
import sys
import threading
import subprocess

from PySide6.QtGui import QGuiApplication
from PySide6 import QtCore
import qt_themes

from W.PySide6.Core.Settings import Field

class _System(QtCore.QObject):
    onChanged = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self._style_hints = None

    def isDark(self) -> bool:
        app = QGuiApplication.instance()
        if app:
            scheme = app.styleHints().colorScheme()
            if scheme == QtCore.Qt.ColorScheme.Dark: return True
            elif scheme == QtCore.Qt.ColorScheme.Light: return False
        
        if sys.platform.startswith("linux"): return self._is_linux_dark()
        else:
            try:
                import darkdetect
                return darkdetect.isDark()
            except ImportError: return False

    def isLight(self) -> bool: return not self.isDark()

    def _is_linux_dark(self) -> bool:
        try:
            res = subprocess.run(
                ['dbus-send', '--session', '--print-reply=literal', 
                 '--dest=org.freedesktop.portal.Desktop',
                 '/org/freedesktop/portal/desktop', 
                 'org.freedesktop.portal.Settings.Read',
                 'string:org.freedesktop.appearance', 'string:color-scheme'],
                capture_output=True, text=True, timeout=1
            )
            if 'uint32 1' in res.stdout: return True
            if 'uint32 0' in res.stdout or 'uint32 2' in res.stdout: return False
        except Exception: pass
            
        try:
            res = subprocess.run(
                ['gsettings', 'get', 'org.gnome.desktop.interface', 'color-scheme'],
                capture_output=True, text=True, timeout=1
            )
            if 'prefer-dark' in res.stdout: return True
        except Exception: pass
        return False

    def start_listener(self) -> None:
        app = QGuiApplication.instance()
        if app:
            self._style_hints = app.styleHints()
            self._style_hints.colorSchemeChanged.connect(self._handle_scheme_change)

        if sys.platform.startswith("linux"):
            t = threading.Thread(target=self._linux_dbus_listener, daemon=True)
            t.start()
        else:
            try:
                import darkdetect
                t = threading.Thread(target=self._darkdetect_listener, daemon=True)
                t.start()
            except ImportError: pass

    def _handle_scheme_change(self, scheme: QtCore.Qt.ColorScheme) -> None:
        if scheme == QtCore.Qt.ColorScheme.Dark: self.onChanged.emit('Dark')
        elif scheme == QtCore.Qt.ColorScheme.Light: self.onChanged.emit('Light')

    def _linux_dbus_listener(self):
        try:
            process = subprocess.Popen(
                ['dbus-monitor', "path='/org/freedesktop/portal/desktop',interface='org.freedesktop.portal.Settings',member='SettingChanged'"],
                stdout=subprocess.PIPE, text=True
            )
            if not process.stdout: raise RuntimeError()
            for line in iter(process.stdout.readline, ''):
                if 'color-scheme' in line or 'uint32 1' in line or 'uint32 0' in line:
                    self.onChanged.emit('Dark' if self._is_linux_dark() else 'Light')
        except Exception: pass

    def _darkdetect_listener(self):
        import darkdetect
        darkdetect.listener(lambda theme: self.onChanged.emit(theme))

System = _System()

class Themes:
    class Settings(QtCore.QObject):
        dark, dark_updated = Field(default="catppuccin_mocha", val_type=str, key="Theme/Dark")
        light, light_updated = Field(default="github_light", val_type=str, key="Theme/Light")
        one, one_updated = Field(default="catppuccin_mocha", val_type=str, key="Theme/One")
        twoMode, twoMode_updated = Field(default=True, val_type=bool, key="Theme/TwoMode")

        def __init__(self, this: 'Themes', SETTING_PATH: Path) -> None:
            super().__init__()
            self.this = this
            self._settings = QtCore.QSettings(str(SETTING_PATH), QtCore.QSettings.Format.IniFormat)
        
        @property
        def settings(self) -> QtCore.QSettings | None: return self._settings

    class API:
        def __init__(self, this: 'Themes') -> None:
            self.this = this

        @property
        def dark(self) -> str: return self.this.settings.dark
        @property
        def light(self) -> str: return self.this.settings.light
        @property
        def one(self) -> str: return self.this.settings.one
        @property
        def twoMode(self) -> bool: return self.this.settings.twoMode

        @property
        def current_active_theme(self) -> qt_themes.Theme:
            return self.this.wrapper.get_curent_theme()

        def set_one_and_apply(self, theme: str) -> None:
            if self.twoMode: raise RuntimeError("Disable twoMode first")
            self.this.settings.one = theme
            self.this.wrapper.update_one_theme()
            
        def set_dark_and_apply(self, theme: str) -> None:
            if not self.twoMode: raise RuntimeError("Enable twoMode first")
            self.this.settings.dark = theme
            self.this.wrapper.update_two_theme()

        def set_light_and_apply(self, theme: str) -> None:
            if not self.twoMode: raise RuntimeError("Enable twoMode first")
            self.this.settings.light = theme
            self.this.wrapper.update_two_theme()
        
        def change_mod(self, mode: bool) -> None:
            if mode ^ self.twoMode:
                self.this.settings.twoMode = mode
                self.this.wrapper.update_any()

    class Wrapper:
        def __init__(self, this: 'Themes') -> None:
            self.this = this

        @staticmethod
        def set_theme(theme: str, style: str | None = 'fusion') -> None:
            qt_themes.set_theme(qt_themes.get_themes().get(theme), style=style)
        
        @staticmethod
        def get_curent_theme() -> qt_themes.Theme:
            theme = qt_themes.get_theme()
            if not theme: raise RuntimeError()
            return theme

        def update_one_theme(self) -> None:
            if self.this.settings.twoMode: raise RuntimeError()
            self.set_theme(self.this.settings.one)

        def update_two_theme(self) -> None:
            if not self.this.settings.twoMode: raise RuntimeError()
            theme = self.this.settings.dark if System.isDark() else self.this.settings.light
            self.set_theme(theme)

        def update_any(self) -> None:
            self.update_two_theme() if self.this.settings.twoMode else self.update_one_theme()
            
        def on_system_theme_changed(self, system_theme_name: str) -> None:
            if not self.this.settings.twoMode: return
            self.update_two_theme()

    class Debug:
        def __init__(self, api: 'Themes.API') -> None:
            self.api = api
            self._timer: QtCore.QTimer | None = None

        def show_all_themes(self) -> None:
            themes = qt_themes.get_themes()
            print(*(f"[THEMES]: {i}" for i in themes.keys()), sep="\n")

        def force_initial_theme(self) -> None:
            theme_name = "catppuccin_mocha" if System.isDark() else "github_light"
            theme_obj = qt_themes.get_themes().get(theme_name)
            qt_themes.set_theme(theme_obj, style='fusion')

        def track_system_theme(self) -> None:
            @System.onChanged.connect
            def _(theme: Any): print(f"[DEBUG] System Theme Change Detected: {theme}")

        def start_random_timer(self) -> None:
            app_instance = QtCore.QCoreApplication.instance()
            self._timer = QtCore.QTimer(parent=app_instance)
            self._timer.setInterval(1000)

            @self._timer.timeout.connect
            def _() -> None:
                themes = qt_themes.get_themes()
                if not self.api.twoMode:
                    valid = [k for k, v in themes.items() if v.is_dark_theme() ^ System.isLight()]
                    if valid: self.api.set_one_and_apply(random.choice(valid))
                    return

                if System.isLight():
                    light_themes = [k for k, v in themes.items() if not v.is_dark_theme()]
                    if light_themes: self.api.set_light_and_apply(random.choice(light_themes))
                else:
                    dark_themes = [k for k, v in themes.items() if v.is_dark_theme()]
                    if dark_themes: self.api.set_dark_and_apply(random.choice(dark_themes))
            self._timer.start()

    def __init__(self, SETTING_PATH: Path) -> None:
        self.settings = self.Settings(self, SETTING_PATH=SETTING_PATH)
        self.api = self.API(self)
        self.wrapper = self.Wrapper(self)
        self.debug = self.Debug(self.api)
        
    def init(self, debug: bool = False) -> None:
        System.start_listener()
        System.onChanged.connect(self.wrapper.on_system_theme_changed)
        self.wrapper.update_any()

        if debug:
            self.debug.force_initial_theme()
            self.debug.track_system_theme()
            self.debug.show_all_themes()
            self.debug.start_random_timer()
