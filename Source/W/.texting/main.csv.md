Отличный план! Строгие правила (namespace modules, PascalCase, только `__init__.py`, тесты рядом) делают архитектуру монолита очень гибкой и предсказуемой. Верхнеуровневые папки (`Core`, `App`, `Data`, `Widgets`, `Graphics`) теперь выступают в роли пространств имен (namespace modules), и мы обращаемся напрямую к подмодулям.

Я полностью удалил старую структуру и сгенерировал новую в соответствии с твоими жесткими правилами.

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Core/Lifecycle/__init__.py" encoding="utf-8"
from typing import TYPE_CHECKING, Any
from PySide6 import QtWidgets, QtCore, QtGui # type: ignore

class ComponentLifecycle(QtWidgets.QWidget if TYPE_CHECKING else object):
    """
    Универсальный миксин, задающий строгий жизненный цикл виджета.
    Избавляет от монолитного __init__ и предотвращает SegFault.
    """
    
    def __init_state__(self):
        """1. Инициализация переменных, флагов, моделей данных."""
        pass

    def __init_graph__(self):
        """2. Построение визуального дерева."""
        pass

    def __init_signal__(self):
        """3. Подключение сигналов и событий."""
        pass

    def onLanguageChange(self):
        """4. Применение текстов и переводов."""
        pass

    def onThemeChange(self):
        """5. Применение стилей, цветов и иконок."""
        pass

    def onResize(self, event: QtGui.QResizeEvent):
        """7. Адаптивный дизайн (срабатывает при изменении размеров виджета)."""
        pass

    def onWindowStateChange(self, state: QtCore.Qt.WindowState):
        """8. Реакция на сворачивание/разворачивание окна."""
        pass

    def __init_ready__(self):
        """6. Пост-инициализация: загрузка данных, запуск таймеров, обновление UI."""
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # 0. Инициализация C++ объекта
        super().__init__(*args, **kwargs)
        
        # Запускаем строгий конвейер инициализации
        self.__init_state__()
        self.__init_graph__()
        self.__init_signal__()
        
        # Первичное применение темы и локали
        self.onLanguageChange()
        self.onThemeChange()
        
        self.__init_ready__()

    # ==========================================
    # ПЕРЕХВАТ НАИВНЫХ СОБЫТИЙ QT
    # ==========================================
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if hasattr(super(), 'resizeEvent'):
            super().resizeEvent(event)
        self.onResize(event)

    def changeEvent(self, event: QtCore.QEvent) -> None:
        if hasattr(super(), 'changeEvent'):
            super().changeEvent(event)
            
        t = event.type()
        
        # Ловим смену языка ОС/Приложения
        if t == QtCore.QEvent.Type.LanguageChange:
            self.onLanguageChange()
            
        # Ловим смену палитры/темы
        elif t == QtCore.QEvent.Type.PaletteChange:
            self.onThemeChange()
            
        # Ловим разворачивание/сворачивание окна
        elif t == QtCore.QEvent.Type.WindowStateChange:
            window = self.window()
            if window:
                self.onWindowStateChange(window.windowState())

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Core/Lifecycle/__test__/__init__.py" encoding="utf-8"
import sys
from PySide6.QtWidgets import QApplication, QLabel
from W.PySide6.Core.Lifecycle import ComponentLifecycle

class MyWidget(ComponentLifecycle, QLabel):
    def __init_state__(self):
        self.click_count = 0

    def __init_graph__(self):
        self.setText(f"Lifecycle Initialized! Clicks: {self.click_count}")

    def mousePressEvent(self, event):
        self.click_count += 1
        self.setText(f"Lifecycle Initialized! Clicks: {self.click_count}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyWidget()
    w.resize(300, 100)
    w.show()
    sys.exit(app.exec())

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Core/Builder/__init__.py" encoding="utf-8"
from contextlib import contextmanager
from typing import Generator, Any
from PySide6.QtWidgets import QWidget, QLayout

@contextmanager
def build_node[T](parent: QWidget | QLayout | Any, item: T) -> Generator[T, None, None]:
    """
    Декларативный билдер UI. 
    Поддерживает любую вложенность:
    - Widget -> Widget (QSplitter, QStackedWidget и т.д.)
    - Widget -> Layout (Устанавливает слой для виджета)
    - Layout -> Widget (Добавляет виджет в слой)
    - Layout -> Layout (Вкладывает слой в слой)
    """
    yield item 
    
    if isinstance(parent, QLayout):
        if isinstance(item, QWidget):
            parent.addWidget(item)
        elif isinstance(item, QLayout):
            parent.addLayout(item)
        else:
            raise TypeError(f"Cannot add {type(item)} to QLayout.")
            
    elif isinstance(parent, QWidget):
        if isinstance(item, QLayout):
            parent.setLayout(item)
        elif isinstance(item, QWidget):
            if hasattr(parent, 'addWidget'):
                parent.addWidget(item)
            else:
                item.setParent(parent)
        else:
            raise TypeError(f"Cannot add {type(item)} to QWidget.")
            
    else:
        raise TypeError(f"Invalid parent type: {type(parent)}. Must be QWidget or QLayout.")

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Core/Builder/__test__/__init__.py" encoding="utf-8"
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from W.PySide6.Core.Builder import build_node as b

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QWidget()
    
    with b(win, QVBoxLayout()) as layout:
        with b(layout, QPushButton("Button 1")) as btn1:
            btn1.setStyleSheet("color: red;")
        with b(layout, QPushButton("Button 2")) as btn2:
            pass

    win.show()
    sys.exit(app.exec())

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Core/Settings/__init__.py" encoding="utf-8"
import json
from enum import Enum
from typing import Any, Dict, List, Protocol, Type, overload, TypeVar, Generic, Tuple
from PySide6 import QtCore, QtGui # type: ignore

# ==========================================
# 1. АДАПТЕРЫ (СЕРИАЛИЗАТОРЫ)
# ==========================================
T = TypeVar('T')

class BaseAdapter(Generic[T]):
    def dump(self, value: T) -> Any: ...
    def load(self, raw_value: Any) -> T: ...

class EnumAdapter(BaseAdapter[Enum]):
    def __init__(self, enum_cls: Type[Enum], fallback: Enum):
        self.enum_cls = enum_cls
        self.fallback = fallback

    def dump(self, value: Enum) -> str:
        return value.name

    def load(self, raw_value: Any) -> Enum:
        try: return self.enum_cls[str(raw_value)]
        except (KeyError, ValueError): return self.fallback

class JsonAdapter(BaseAdapter[Dict[Any, Any] | List[Any]]):
    def dump(self, value: Dict[Any, Any] | List[Any]) -> str:
        return json.dumps(value, ensure_ascii=False)

    def load(self, raw_value: Any) -> Dict[Any, Any] | List[Any]:
        if not isinstance(raw_value, str): return {}
        try: return json.loads(raw_value)
        except json.JSONDecodeError: return {}

class LocalesAdapter(BaseAdapter[list[QtCore.QLocale]]):
    def dump(self, value: list[QtCore.QLocale]) -> list[str]:
        return [loc.name() for loc in value]

    def load(self, raw_value: Any) -> list[QtCore.QLocale]:
        if not raw_value: return []
        if isinstance(raw_value, str): raw_value = [raw_value]
        return [QtCore.QLocale(s) for s in raw_value]

class QtObjAdapter(BaseAdapter[T]):
    def __init__(self, obj_type: Type[T], fallback: T):
        self.obj_type = obj_type
        self.fallback = fallback

    def dump(self, value: T) -> Any: return value

    def load(self, raw_value: Any) -> T:
        if isinstance(raw_value, self.obj_type): return raw_value
        return self.fallback
    
class QColorAdapter(BaseAdapter[QtGui.QColor]):
    def __init__(self, fallback: QtGui.QColor):
        self.fallback = fallback

    def dump(self, value: QtGui.QColor) -> str: return value.name()

    def load(self, raw_value: Any) -> QtGui.QColor:
        if isinstance(raw_value, str) and QtGui.QColor.isValidColor(raw_value):
            return QtGui.QColor(raw_value)
        return self.fallback

# ==========================================
# 2. УНИВЕРСАЛЬНЫЙ ДЕСКРИПТОР
# ==========================================
class Settings(Protocol):
    @property
    def settings(self) -> QtCore.QSettings | None: ...

class SettingField[T]:
    def __init__(self, default: T, val_type: Type[T] | None = None, key: str | None = None, signal_ref: Any = None, adapter: BaseAdapter[T] | None = None) -> None:
        self.default = default
        self.val_type = val_type
        self.key = key
        self.adapter = adapter
        self._signal_ref = signal_ref
        self._signal_name: str | None = None

    def __set_name__(self, owner: type[Any], name: str) -> None:
        if self.key is None: self.key = name
        if self._signal_ref is not None:
            for attr_name, attr_value in vars(owner).items():
                if attr_value is self._signal_ref:
                    self._signal_name = attr_name
                    break

    def _get_settings(self, instance: Settings) -> QtCore.QSettings:
        st = getattr(instance, 'settings', None)
        return st if isinstance(st, QtCore.QSettings) else QtCore.QSettings()

    @overload
    def __get__(self, instance: None, owner: Type[Any]) -> 'SettingField[T]': ...
    @overload
    def __get__(self, instance: Settings, owner: Type[Any]) -> T: ...
    def __get__(self, instance: Settings | None, owner: Type[Any]) -> T | 'SettingField[T]':
        if instance is None: return self
        if self.key is None: raise RuntimeError("__set_name__ not colled")
        
        settings = self._get_settings(instance)
        raw_val = settings.value(self.key, None)

        if raw_val is None: return self.default
        if self.adapter: return self.adapter.load(raw_val)
        if self.val_type is not None: return settings.value(self.key, self.default, type=self.val_type) # type: ignore
        return raw_val

    def __set__(self, instance: Settings, value: T) -> None:
        current_value = self.__get__(instance, type(instance))
        if current_value == value: return

        settings = self._get_settings(instance)
        save_val = self.adapter.dump(value) if self.adapter else value
        settings.setValue(self.key, save_val) # type: ignore

        if self._signal_name and hasattr(instance, self._signal_name):
            getattr(instance, self._signal_name).emit(value)

# ==========================================
# 3. ФАБРИКА
# ==========================================
def Field[T](default: T, val_type: Type[T] | None = None, key: str | None = None, adapter: BaseAdapter[T] | None = None, sig_type: Type[Any] | None = None) -> Tuple[SettingField[T], Any]:
    if sig_type is None:
        sig_type = val_type if val_type is not None else type(default)
        if sig_type is list or sig_type is dict: sig_type = object

    sig = QtCore.Signal(sig_type)
    desc = SettingField[T](default=default, val_type=val_type, key=key, signal_ref=sig, adapter=adapter)
    return desc, sig

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Core/Settings/__test__/__init__.py" encoding="utf-8"
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

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/App/Themes/__init__.py" encoding="utf-8"
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

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/App/Themes/__test__/__init__.py" encoding="utf-8"
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from W.PySide6.App.Themes import Themes, System

if __name__ == "__main__":
    app = QApplication(sys.argv)
    t = Themes(Path("test_themes.ini"))
    print(f"Is system dark? {System.isDark()}")
    sys.exit()

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/App/Locales/__init__.py" encoding="utf-8"
import os
import inspect
import random
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QLocale, QObject, QTranslator
from PySide6 import QtCore

from W.PySide6.Core.Settings import Field, LocalesAdapter

class Locales:
    _active_instances: list['Locales'] = []

    class Settings(QObject):
        locales, locales_updated = Field(
            default=[QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)],
            adapter=LocalesAdapter(),
            key="Locales"
        )

        def __init__(self, this: 'Locales', SETTING_PATH: Path) -> None:
            super().__init__()
            self.this = this
            self._settings = QtCore.QSettings(str(SETTING_PATH), QtCore.QSettings.Format.IniFormat)
        
        @property
        def settings(self) -> QtCore.QSettings | None: return self._settings

    class API:
        def __init__(self, this: 'Locales') -> None:
            self.this = this
            self._supported_languages: list[QLocale] = []

        @property
        def current_locales(self) -> list[QLocale]: return self.this.settings.locales
        @property
        def supported_languages(self) -> list[QLocale]: return self._supported_languages

        def set_supported_languages(self, languages: list[QLocale]) -> None:
            self._supported_languages = languages

        def change_language(self, new_locales: list[QLocale]) -> None:
            self.this.settings.locales = new_locales

    class TranslateComponent:
        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            module = inspect.getmodule(cls)
            
            old_init = cls.__init__
            def new_init(self_obj: Any, *args: Any, **kw: Any) -> None:
                if not getattr(cls, "_translator_registered", False):
                    setattr(cls, "_translator_registered", True)
                    if module and module.__file__:
                        path = Path(module.__file__).parent / "__assets__" / "translations"
                        for instance in Locales._active_instances:
                            instance.wrapper.compile_jit(module.__file__)
                            instance.wrapper.register_translation_path(path)
                old_init(self_obj, *args, **kw)
            cls.__init__ = new_init

    class Wrapper:
        def __init__(self, this: 'Locales') -> None:
            self.this = this
            self.jit_enabled: bool = False
            self.translation_paths: set[Path] = set()
            self.active_translators: list[QTranslator] = []

        def register_translation_path(self, path: Path) -> None:
            if path not in self.translation_paths:
                self.translation_paths.add(path)
                self.reload_translators(self.this.api.current_locales)

        def compile_jit(self, module_file: str) -> None:
            if not self.jit_enabled: return
            path = Path(module_file).parent / "__script__" / "loc.comp.sh"
            if path.exists(): os.system(str(path))

        def reload_translators(self, locales: list[QLocale]) -> None:
            app = QApplication.instance()
            if not app: return

            for t in self.active_translators: app.removeTranslator(t)
            self.active_translators.clear()

            for path in self.translation_paths:
                for locale in locales:
                    translator = QTranslator()
                    if translator.load(locale, "this", "_", str(path), ".qm"):
                        app.installTranslator(translator)
                        self.active_translators.append(translator)

    class Debug:
        def __init__(self, api: 'Locales.API') -> None:
            self.api = api
            self._timer: QtCore.QTimer | None = None

        def start_random_timer(self) -> None:
            app_instance = QApplication.instance()
            if not app_instance: return
            self._timer = QtCore.QTimer(parent=app_instance)
            self._timer.setInterval(1000)

            @self._timer.timeout.connect
            def _() -> None:
                supported = self.api.supported_languages
                if not supported: return
                new_locales = [
                    QLocale(QLocale.Language.English, QLocale.Country.UnitedStates),
                    random.choice(supported)
                ]
                self.api.change_language(new_locales)
            self._timer.start()

    def __init__(self, SETTING_PATH: Path) -> None:
        self.settings = self.Settings(self, SETTING_PATH)
        self.api = self.API(self)
        self.wrapper = self.Wrapper(self)
        self.debug = self.Debug(self.api)
        Locales._active_instances.append(self)

    def init(self, supported_languages: list[QLocale], jit_compile: bool = False, debug: bool = False) -> None:
        self.api.set_supported_languages(supported_languages)
        self.wrapper.jit_enabled = jit_compile
        self.settings.locales_updated.connect(self.wrapper.reload_translators)
        self.wrapper.reload_translators(self.api.current_locales)
        if debug: self.debug.start_random_timer()

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/App/Locales/__test__/__init__.py" encoding="utf-8"
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from W.PySide6.App.Locales import Locales

if __name__ == "__main__":
    app = QApplication(sys.argv)
    l = Locales(Path("test_locales.ini"))
    print("Locales System Created")
    sys.exit()

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Data/Reactive/__init__.py" encoding="utf-8"
from collections.abc import Hashable, Iterable
from typing import Any
from PySide6.QtCore import QObject, Signal, Property

class ReactiveDict[K: Hashable, V](dict[K, V]):
    class Signals(QObject):
        itemSet = Signal(object, object, object)
        itemRemoved = Signal(object, object)
        cleared = Signal()
        updated = Signal(list)
        changed = Signal()

        def __init__(self, parent_dict: 'ReactiveDict[K, V]', parent: QObject | None = None):
            super().__init__(parent)
            self._dict = parent_dict

        @Property("QVariantMap", notify=changed)
        def data(self) -> dict[K, V]: return dict(self._dict)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.signals = self.Signals(self)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: K, value: V) -> None:
        is_existing = key in self
        old_value = self.get(key)
        if is_existing and old_value == value: return
        super().__setitem__(key, value)
        self.signals.itemSet.emit(key, value, old_value)
        self.signals.changed.emit()

    def __delitem__(self, key: K) -> None:
        old_value = super().pop(key) 
        self.signals.itemRemoved.emit(key, old_value)
        self.signals.changed.emit()

    def pop(self, key: K, *args: Any) -> Any:
        try:
            value = super().pop(key)
            self.signals.itemRemoved.emit(key, value)
            self.signals.changed.emit()
            return value
        except KeyError:
            if args: return args[0]
            raise 

    def clear(self) -> None:
        if not self: return
        super().clear()
        self.signals.cleared.emit()
        self.signals.changed.emit()

    def update(self, *args: Any, **kwargs: Any) -> None:
        new_data = dict(*args, **kwargs)
        if not new_data: return
        changed_keys: list[K] = []
        for k, v in new_data.items():
            if k not in self or self.get(k) != v:
                old_val = self.get(k)
                super().__setitem__(k, v)
                changed_keys.append(k)
                self.signals.itemSet.emit(k, v, old_val)
        if changed_keys:
            self.signals.updated.emit(changed_keys)
            self.signals.changed.emit()

def make_reactive_dict[K, V](d: dict[K, V]) -> ReactiveDict[K, V]:
    return d if type(d) is ReactiveDict else ReactiveDict(d)

class ReactiveList[T](list[T]):
    class Signals(QObject):
        itemAdded = Signal(int, object)
        itemRemoved = Signal(int, object)
        itemSet = Signal(int, object, object)
        cleared = Signal()
        updated = Signal()
        changed = Signal()

        def __init__(self, parent_list: 'ReactiveList[T]', parent: QObject | None = None):
            super().__init__(parent)
            self._list = parent_list

        @Property(list, notify=changed)
        def data(self) -> list[T]: return list(self._list)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.signals = self.Signals(self)
        super().__init__(*args, **kwargs)

    def append(self, item: T) -> None:
        super().append(item)
        self.signals.itemAdded.emit(len(self) - 1, item)
        self.signals.changed.emit()

    def remove(self, item: T) -> None:
        idx = self.index(item)
        super().remove(item)
        self.signals.itemRemoved.emit(idx, item)
        self.signals.changed.emit()

    def clear(self) -> None:
        if not self: return
        super().clear()
        self.signals.cleared.emit()
        self.signals.changed.emit()

    def extend(self, iterable: Iterable[T]) -> None:
        items = list(iterable)
        if not items: return
        super().extend(items)
        self.signals.updated.emit()
        self.signals.changed.emit()

def make_reactive_list[T](l: list[T]) -> ReactiveList[T]:
    return l if type(l) is ReactiveList else ReactiveList(l)

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Data/Reactive/__test__/__init__.py" encoding="utf-8"
from W.PySide6.Data.Reactive import ReactiveDict

if __name__ == "__main__":
    d = ReactiveDict({"a": 1})
    d.signals.itemSet.connect(lambda k, nv, ov: print(f"Changed {k}: {ov} -> {nv}"))
    d["a"] = 2
    d["b"] = 3

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Data/ModelSQLA/__init__.py" encoding="utf-8"
import logging
from typing import Any, List, Optional, Type
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import QStyledItemDelegate, QSpinBox, QDoubleSpinBox, QComboBox, QWidget
from sqlalchemy.orm import Session, DeclarativeBase
from sqlalchemy.types import Enum as SqlaEnum

logger = logging.getLogger(__name__)

def _get_python_type(column: Any) -> Type[Any]:
    try: return column.type.python_type
    except NotImplementedError: return str

class SqlAlchemyDelegate(QStyledItemDelegate):
    def __init__(self, sqla_model: Type[DeclarativeBase], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.sqla_model = sqla_model
        self.columns = list(sqla_model.__mapper__.columns)

    def createEditor(self, parent: QWidget, option: Any, index: QModelIndex) -> Optional[QWidget]:
        col = self.columns[index.column()]
        py_type = _get_python_type(col)

        if isinstance(col.type, SqlaEnum) and col.type.enum_class:
            editor = QComboBox(parent)
            editor.addItems([e.name for e in col.type.enum_class])
            return editor

        if py_type is bool: return None
        if py_type is int:
            editor = QSpinBox(parent)
            editor.setRange(-2147483648, 2147483647)
            return editor
        if py_type is float:
            editor = QDoubleSpinBox(parent)
            editor.setRange(-1e9, 1e9)
            editor.setDecimals(4)
            return editor
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        col = self.columns[index.column()]
        if isinstance(col.type, SqlaEnum) and isinstance(editor, QComboBox):
            value = index.model().data(index, Qt.ItemDataRole.EditRole)
            if value:
                idx = editor.findText(str(value))
                if idx >= 0: editor.setCurrentIndex(idx)
            return
        super().setEditorData(editor, index)

    def setModelData(self, editor: QWidget, model: QAbstractTableModel, index: QModelIndex) -> None:
        col = self.columns[index.column()]
        if isinstance(col.type, SqlaEnum) and isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
            return
        super().setModelData(editor, model, index)

class SqlAlchemyTableModel(QAbstractTableModel):
    def __init__(self, db_session: Session, sqla_model: Type[DeclarativeBase]):
        super().__init__()
        self.db = db_session
        self.sqla_model = sqla_model
        self.columns = list(sqla_model.__mapper__.columns)
        self.records: List[Any] = []
        self.refresh()

    def refresh(self) -> None:
        self.beginResetModel()
        try: self.records = self.db.query(self.sqla_model).all()
        except Exception as e:
            logger.error(f"Failed to fetch records: {e}")
            self.records = []
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int: return len(self.records)
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int: return len(self.columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.columns[section].name
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid(): return Qt.ItemFlag.NoItemFlags
        col_name = self.columns[index.column()].name
        py_type = _get_python_type(self.columns[index.column()])
        
        if self.columns[index.column()].primary_key or "id" in col_name.lower():
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
        if py_type is bool: flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid(): return None
        record = self.records[index.row()]
        col = self.columns[index.column()]
        val = getattr(record, col.name)
        py_type = _get_python_type(col)

        if role == Qt.ItemDataRole.CheckStateRole and py_type is bool:
            return Qt.CheckState.Checked if val else Qt.CheckState.Unchecked

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if py_type is bool: return None
            if val is None: return ""
            if isinstance(col.type, SqlaEnum): return val.name if hasattr(val, "name") else str(val)
            return str(val)
        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid(): return False
        record = self.records[index.row()]
        col = self.columns[index.column()]
        py_type = _get_python_type(col)

        try:
            if role == Qt.ItemDataRole.CheckStateRole and py_type is bool:
                new_val = (value == Qt.CheckState.Checked.value or value == 2)
                setattr(record, col.name, new_val)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
                return True
                
            if role == Qt.ItemDataRole.EditRole:
                if value == "": setattr(record, col.name, None)
                elif isinstance(col.type, SqlaEnum):
                    enum_member = col.type.enum_class[value]
                    setattr(record, col.name, enum_member)
                else: setattr(record, col.name, py_type(value))
                    
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
                return True
        except (ValueError, KeyError) as e:
            logger.warning(f"Validation error: {e}")
        return False

    def insertRows(self, row: int, count: int, parent: QModelIndex = QModelIndex()) -> bool:
        self.beginInsertRows(parent, row, row + count - 1)
        try:
            for _ in range(count):
                new_record = self.sqla_model()
                for col in self.columns:
                    if col.default and col.default.is_callable:
                        try: setattr(new_record, col.name, col.default.arg({})) 
                        except Exception: pass
                self.db.add(new_record)
                self.records.insert(row, new_record)
            self.endInsertRows()
            return True
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            self.db.rollback()
            self.endInsertRows()
            return False

    def removeRows(self, row: int, count: int, parent: QModelIndex = QModelIndex()) -> bool:
        self.beginRemoveRows(parent, row, row + count - 1)
        try:
            for i in range(count):
                self.db.delete(self.records[row + i])
            del self.records[row:row+count]
            self.endRemoveRows()
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            self.db.rollback()
            self.endRemoveRows()
            return False

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Data/ModelSQLA/__test__/__init__.py" encoding="utf-8"
print("Test mock for Data.ModelSQLA")

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Widgets/Frameless/__init__.py" encoding="utf-8"
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QGraphicsDropShadowEffect, QFrame, QDialog
)
from PySide6.QtGui import QPalette, QPixmap
from PySide6.QtCore import Qt, QEvent
from W.PySide6.Core.Builder import build_node as b
from typing import TYPE_CHECKING

GRIP_MARGIN = 6
GRIP_SIZE = 12

class WindowEdgeGrip(QWidget):
    def __init__(self, parent, edge: Qt.Edge, cursor: Qt.CursorShape):
        super().__init__(parent)
        self.edge = edge
        self.setCursor(cursor)
        self.setStyleSheet("background: transparent;")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.window().windowHandle():
                self.window().windowHandle().startSystemResize(self.edge)
                return
        super().mousePressEvent(event)

class TitleBarButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        self.setFixedSize(42, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)

class TitleBar(QWidget):
    def __init__(self, parent, icon_path: Path | None = None, fallback_icon: str = "🦇"):
        super().__init__(parent)
        self.setFixedHeight(36)
        
        with b(self, QHBoxLayout()) as self.layout:
            self.layout.setContentsMargins(12, 0, 0, 0)
            self.layout.setSpacing(8)
            
            with b(self.layout, QLabel()) as self.icon_label:
                self.icon_label.setFixedSize(20, 20)
                if icon_path and icon_path.exists():
                    pixmap = QPixmap(str(icon_path)).scaled(
                        20, 20, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                    )
                    self.icon_label.setPixmap(pixmap)
                else:
                    self.icon_label.setText(fallback_icon)
                    self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            with b(self.layout, QLabel(parent.windowTitle())) as self.title_label:
                font = self.title_label.font()
                font.setPointSize(10)
                font.setBold(True)
                self.title_label.setFont(font)
                
            with b(self.layout, QHBoxLayout()) as self.custom_layout:
                self.custom_layout.setContentsMargins(8, 0, 8, 0)
                self.custom_layout.setSpacing(8)
                self.custom_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
                
            self.layout.addStretch()
            
            if not isinstance(parent, QDialog):
                with b(self.layout, TitleBarButton("─")) as self.min_btn:
                    self.min_btn.clicked.connect(parent.showMinimized)
                with b(self.layout, TitleBarButton("◻")) as self.max_btn:
                    self.max_btn.clicked.connect(self._toggle_maximize)
                
            with b(self.layout, TitleBarButton("✕")) as self.close_btn:
                self.close_btn.clicked.connect(parent.close)

    def _toggle_maximize(self):
        win = self.window()
        if win.isMaximized(): win.showNormal()
        else: win.showMaximized()

    def update_title(self, title: str):
        self.title_label.setText(title)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.window().windowHandle():
                self.window().windowHandle().startSystemMove()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not isinstance(self.window(), QDialog):
            self._toggle_maximize()

class FramelessMixin(QWidget if TYPE_CHECKING else object):
    def init_frameless(self, icon_path: Path | None = None, fallback_icon: str = "🦇"):
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self._grips = []
        grip_data = [
            (Qt.Edge.TopEdge, Qt.CursorShape.SizeVerCursor),
            (Qt.Edge.BottomEdge, Qt.CursorShape.SizeVerCursor),
            (Qt.Edge.LeftEdge, Qt.CursorShape.SizeHorCursor),
            (Qt.Edge.RightEdge, Qt.CursorShape.SizeHorCursor),
            (Qt.Edge.TopEdge | Qt.Edge.LeftEdge, Qt.CursorShape.SizeFDiagCursor),
            (Qt.Edge.TopEdge | Qt.Edge.RightEdge, Qt.CursorShape.SizeBDiagCursor),
            (Qt.Edge.BottomEdge | Qt.Edge.LeftEdge, Qt.CursorShape.SizeBDiagCursor),
            (Qt.Edge.BottomEdge | Qt.Edge.RightEdge, Qt.CursorShape.SizeFDiagCursor),
        ]
        
        for edge, cursor in grip_data:
            grip = WindowEdgeGrip(self, edge, cursor)
            self._grips.append(grip)
            
        self._icon_path = icon_path
        self._fallback_icon = fallback_icon

    def build_frameless_ui(self) -> QVBoxLayout:
        with b(self, QWidget()) as self.bg_wrapper:
            if hasattr(self, 'setCentralWidget'):
                self.setCentralWidget(self.bg_wrapper)
            else:
                self.setLayout(QVBoxLayout())
                self.layout().setContentsMargins(0, 0, 0, 0)
                self.layout().addWidget(self.bg_wrapper)

            with b(self.bg_wrapper, QVBoxLayout()) as self.bg_layout:
                self.bg_layout.setContentsMargins(GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN)
                
                with b(self.bg_layout, QFrame()) as self.root:
                    self.root.setObjectName("FramelessRoot")
                    
                    self.shadow_effect = QGraphicsDropShadowEffect(self.root)
                    self.shadow_effect.setBlurRadius(GRIP_MARGIN * 5)
                    self.shadow_effect.setOffset(0, 0)
                    self.root.setGraphicsEffect(self.shadow_effect)
                    
                    with b(self.root, QVBoxLayout()) as self.root_v:
                        self.root_v.setContentsMargins(0, 0, 0, 0)
                        self.root_v.setSpacing(0)
                        
                        with b(self.root_v, TitleBar(self, self._icon_path, self._fallback_icon)) as self.title_bar: pass
                        with b(self.root_v, QWidget()) as self.content_widget:
                            with b(self.content_widget, QVBoxLayout()) as self.content_layout:
                                self.content_layout.setContentsMargins(4, 4, 4, 4)
                                
        self.update_frameless_theme()
        return self.content_layout

    def update_frameless_theme(self):
        if not hasattr(self, 'root'): return
        bg_color = self.palette().color(QPalette.ColorRole.Window)
        border_color = self.palette().color(QPalette.ColorRole.Shadow)
        self.shadow_effect.setColor(border_color)
        
        if self.isMaximized():
            self.bg_layout.setContentsMargins(0, 0, 0, 0)
            self.root.setStyleSheet(f"QFrame#FramelessRoot {{ background-color: {bg_color.name()}; border-radius: 0px; border: none; }}")
        else:
            self.bg_layout.setContentsMargins(GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN, GRIP_MARGIN)
            self.root.setStyleSheet(f"QFrame#FramelessRoot {{ background-color: {bg_color.name()}; border-radius: 8px; border: 1px solid {border_color.name()}; }}")

    def _update_grips(self):
        if not hasattr(self, '_grips'): return
        w, h = self.width(), self.height()
        s = GRIP_SIZE
        
        self._grips[0].setGeometry(s, 0, w - 2*s, s)
        self._grips[1].setGeometry(s, h - s, w - 2*s, s)
        self._grips[2].setGeometry(0, s, s, h - 2*s)
        self._grips[3].setGeometry(w - s, s, s, h - 2*s)
        self._grips[4].setGeometry(0, 0, s, s)
        self._grips[5].setGeometry(w - s, 0, s, s)
        self._grips[6].setGeometry(0, h - s, s, s)
        self._grips[7].setGeometry(w - s, h - s, s, s)
        
        for grip in self._grips: grip.raise_()

    def resizeEvent(self, event):
        if hasattr(super(), 'resizeEvent'): super().resizeEvent(event)
        self._update_grips()

    def changeEvent(self, event):
        if hasattr(super(), 'changeEvent'): super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            if hasattr(self, '_grips'):
                is_max = self.isMaximized()
                for grip in self._grips: grip.setVisible(not is_max)
            if hasattr(self, 'title_bar') and hasattr(self.title_bar, 'max_btn'):
                self.title_bar.max_btn.setText("🗗" if self.isMaximized() else "◻")
            self.update_frameless_theme()

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Widgets/Frameless/__test__/__init__.py" encoding="utf-8"
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
from W.PySide6.Widgets.Frameless import FramelessMixin

class MyWin(FramelessMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_frameless()
        layout = self.build_frameless_ui()
        layout.addWidget(QLabel("Frameless Window Magic!"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyWin()
    w.resize(400, 300)
    w.show()
    sys.exit(app.exec())

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Widgets/Selector/__init__.py" encoding="utf-8"
from typing import Any
from PySide6 import QtCore, QtWidgets
from W.PySide6.Data.Reactive import ReactiveDict

class Selector(QtWidgets.QGroupBox):
    class Signals(QtCore.QObject):
        keySelected = QtCore.Signal(str)
        activeValueChanged = QtCore.Signal(object)

    def __init__(self, title: str = "Select:", parent: QtWidgets.QWidget | None = None):
        super().__init__(title, parent)
        self.signals = self.Signals(self)
        self._rdict: ReactiveDict[str, Any] | None = None
        self._current_key: str | None = None

        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(4, 8, 4, 4)

        self.label = QtWidgets.QLabel(self.tr("Active Key:"))
        self.main_layout.addWidget(self.label)

        self.combo = QtWidgets.QComboBox(self)
        self.combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.main_layout.addWidget(self.combo)

        self.combo.currentIndexChanged.connect(self._on_combo_index_changed)

    def set_dictionary(self, rdict: ReactiveDict[str, Any]) -> None:
        if self._rdict is not None:
            self._rdict.signals.itemSet.disconnect(self._on_dict_item_set)
            self._rdict.signals.itemRemoved.disconnect(self._on_dict_item_removed)
            self._rdict.signals.cleared.disconnect(self._on_dict_cleared)

        self._rdict = rdict
        self._rdict.signals.itemSet.connect(self._on_dict_item_set)
        self._rdict.signals.itemRemoved.connect(self._on_dict_item_removed)
        self._rdict.signals.cleared.connect(self._on_dict_cleared)
        self._rebuild_ui()

    def active_key(self) -> str | None: return self._current_key

    def active_value(self) -> Any:
        if self._rdict is not None and self._current_key is not None:
            return self._rdict.get(self._current_key)
        return None

    def _rebuild_ui(self) -> None:
        with QtCore.QSignalBlocker(self.combo):
            self.combo.clear()
            if self._rdict: self.combo.addItems(list(self._rdict.keys()))
        self._sync_current_selection()

    def _on_dict_item_set(self, key: str, new_value: Any, old_value: Any) -> None:
        idx = self.combo.findText(key)
        if idx == -1:
            with QtCore.QSignalBlocker(self.combo): self.combo.addItem(key)
            self._sync_current_selection()
        elif key == self._current_key:
            self.signals.activeValueChanged.emit(new_value)

    def _on_dict_item_removed(self, key: str, old_value: Any) -> None:
        idx = self.combo.findText(key)
        if idx >= 0:
            with QtCore.QSignalBlocker(self.combo): self.combo.removeItem(idx)
            self._sync_current_selection()

    def _on_dict_cleared(self) -> None:
        with QtCore.QSignalBlocker(self.combo): self.combo.clear()
        self._sync_current_selection()

    def _on_combo_index_changed(self, index: int) -> None:
        self._sync_current_selection()

    def _sync_current_selection(self) -> None:
        new_key = self.combo.currentText() if self.combo.count() > 0 else None
        if new_key == self._current_key: return
        self._current_key = new_key
        
        if self._current_key is not None:
            self.signals.keySelected.emit(self._current_key)
            self.signals.activeValueChanged.emit(self.active_value())
        else:
            self.signals.keySelected.emit("")
            self.signals.activeValueChanged.emit(None)

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Widgets/Selector/__test__/__init__.py" encoding="utf-8"
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from W.PySide6.Widgets.Selector import Selector
from W.PySide6.Data.Reactive import ReactiveDict

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QWidget()
    win.resize(400, 200)
    layout = QVBoxLayout(win)

    db = ReactiveDict({"Key 1": "Hello", "Key 2": "World"})
    selector = Selector("Test Selector")
    selector.set_dictionary(db)
    layout.addWidget(selector)

    lbl = QLabel("Value: ")
    layout.addWidget(lbl)

    selector.signals.activeValueChanged.connect(lambda v: lbl.setText(f"Value: {v}"))
    
    btn = QPushButton("Add 'Key 3'")
    btn.clicked.connect(lambda: db.update({"Key 3": "PySide6 Magic!"}))
    layout.addWidget(btn)

    win.show()
    sys.exit(app.exec())

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Widgets/TabInstance/__init__.py" encoding="utf-8"
from typing import Self
from PySide6.QtWidgets import QWidget, QTabWidget

class TabInstance(QWidget):
    tab_name: str | None = None
    tab_widget: QTabWidget

    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.tab_widget = tab_widget
        self.setContentsMargins(0, 0, 0, 0)
        if self.tab_widget:
            self.tab_widget.addTab(self, self.tab_name)

    @property
    def tabIndex(self: Self) -> int:
        return self.tab_widget.indexOf(self)

    def setTabName(self: Self, name: str | None) -> Self:
        self.tab_name = name
        if self.tab_widget:
            self.tab_widget.setTabText(self.tabIndex, name)
        return self
    
    def setTabToolTip(self: Self, tooltip: str | None) -> Self:
        if self.tab_widget:
            self.tab_widget.setTabToolTip(self.tabIndex, tooltip)
        return self

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Widgets/TabInstance/__test__/__init__.py" encoding="utf-8"
import sys
from PySide6.QtWidgets import QApplication, QTabWidget
from W.PySide6.Widgets.TabInstance import TabInstance

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tabs = QTabWidget()
    
    class MyTab(TabInstance):
        tab_name = "Dynamic Tab"

    t = MyTab(tab_widget=tabs)
    tabs.show()
    sys.exit(app.exec())

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Graphics/FastGraph/__init__.py" encoding="utf-8"
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from dataclasses import dataclass

def _chaikin_smooth(pts: np.ndarray, iterations: int = 3) -> np.ndarray:
    if len(pts) < 3: return pts
    for _ in range(iterations):
        p0, p1 = pts[:-1], pts[1:]
        q1 = 0.75 * p0 + 0.25 * p1
        q2 = 0.25 * p0 + 0.75 * p1
        new_pts = np.empty((len(pts) * 2 - 2, 2), dtype=np.float32)
        new_pts[0::2], new_pts[1::2] = q1, q2
        new_pts[0], new_pts[-1] = pts[0], pts[-1]
        pts = new_pts
    return pts

class EightHandleROI(pg.ROI):
    def __init__(self, pos, size, pen, hoverPen=None, handlePen=None):
        super().__init__(pos, size, movable=True, resizable=True, rotatable=False, pen=pen, hoverPen=hoverPen)
        for x, y in [(0,0), (1,1), (0,1), (1,0), (0.5,0), (0.5,1), (0,0.5), (1,0.5)]:
            self.addScaleHandle([x, y], [1-x, 1-y], pen=handlePen)

class DraggablePointROI(pg.ROI):
    def __init__(self, pos, pen, brush):
        super().__init__(pos, [0, 0], movable=True, resizable=False, rotatable=False)
        self.pen = pen
        self.brush = brush

    def paint(self, p, opt, widget):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(self.pen)
        p.setBrush(self.brush)
        px, py = self.pixelSize()
        rx, ry = 6 * px, 6 * py
        p.drawEllipse(QtCore.QRectF(-rx, -ry, rx * 2, ry * 2))

    def boundingRect(self):
        px, py = self.pixelSize()
        rx, ry = 8 * px, 8 * py
        return QtCore.QRectF(-rx, -ry, rx * 2, ry * 2)

class SmoothPolyLineROI(pg.PolyLineROI):
    def __init__(self, positions, pen):
        super().__init__(positions, closed=False, pen=pg.mkPen(None), hoverPen=None, handlePen=pg.mkPen('w', width=2))
        self.handleSize = 10 
        for h in self.getHandles():
            h.radius = self.handleSize
            h.buildPath()              
            h._shape = None            
            
        self.smooth_path_item = QtWidgets.QGraphicsPathItem(self)
        self.smooth_path_item.setPen(pen)
        self.smooth_path_item.setBrush(QtCore.Qt.BrushStyle.NoBrush) 
        self.smooth_path_item.setZValue(-1)
        self.sigRegionChanged.connect(self.update_smooth_path)
        self.update_smooth_path()

    def update_smooth_path(self):
        handles = self.getHandles()
        if len(handles) < 2: return
        QtCore.QTimer.singleShot(0, self._force_handles_to_top)
        pts = np.array([[h.pos().x(), h.pos().y()] for h in handles])
        smooth_pts = _chaikin_smooth(pts, iterations=3)
        path = pg.arrayToQPath(smooth_pts[:, 0], smooth_pts[:, 1], connect='finite')
        self.smooth_path_item.setPath(path)

    def _force_handles_to_top(self):
        for h in self.getHandles():
            h.setZValue(99999) 
            if h.radius != self.handleSize:
                h.radius = self.handleSize
                h.buildPath()
                h._shape = None
                h.prepareGeometryChange()

    def get_raw_points(self):
        return [[float(self.mapToParent(h.pos()).x()), float(self.mapToParent(h.pos()).y())] for h in self.getHandles()]

@dataclass
class FastElement:
    id: str
    group_id: str
    geom: list | np.ndarray

class AbstractFastLayer(pg.GraphicsObject):
    def __init__(self, bucket_width: float = 1000.0):
        super().__init__()
        self.bucket_width = bucket_width
        self.elements: dict[str, FastElement] = {}
        self.group_colors: dict[str, tuple[QtGui.QColor, QtGui.QColor]] = {}
        self.hidden_ids: set[str] = set()
        self._path_cache: dict[str, QtGui.QPainterPath] = {} 
        self._buckets: dict[int, list[str]] = {}

    def set_colors(self, group_id: str, pen_color: QtGui.QColor, brush_color: QtGui.QColor | None = None):
        self.group_colors[group_id] = (pen_color, brush_color)

    def add_element(self, el: FastElement):
        self.elements[el.id] = el
        self._add_to_buckets(el)

    def remove_element(self, el_id: str):
        if el_id in self.elements:
            self._remove_from_buckets(self.elements[el_id])
            del self.elements[el_id]

    def hide_element(self, el_id: str): self.hidden_ids.add(el_id)
    def show_element(self, el_id: str): self.hidden_ids.discard(el_id)

    def _get_x_bounds(self, el: FastElement) -> tuple[float, float]: raise NotImplementedError()

    def _add_to_buckets(self, el: FastElement):
        x_min, x_max = self._get_x_bounds(el)
        b0, b1 = int(x_min // self.bucket_width), int(x_max // self.bucket_width)
        for b in range(b0, b1 + 1): self._buckets.setdefault(b, []).append(el.id)

    def _remove_from_buckets(self, el: FastElement):
        x_min, x_max = self._get_x_bounds(el)
        b0, b1 = int(x_min // self.bucket_width), int(x_max // self.bucket_width)
        for b in range(b0, b1 + 1):
            if b in self._buckets and el.id in self._buckets[b]:
                self._buckets[b].remove(el.id)

    def hit_test(self, px: float, py: float, threshold: float = 0) -> FastElement | None:
        b = int(px // self.bucket_width)
        for el_id in reversed(self._buckets.get(b, [])):
            if el_id in self.hidden_ids: continue
            el = self.elements[el_id]
            if self._check_hit(el, px, py, threshold): return el
        return None

    def _check_hit(self, el: FastElement, px: float, py: float, threshold: float) -> bool: raise NotImplementedError()

    def rebuild(self):
        self._path_cache.clear()
        self.update()

    def boundingRect(self):
        return QtCore.QRectF(-1e9, -1e9, 2e9, 2e9)

class FastRectLayer(AbstractFastLayer):
    def _get_x_bounds(self, el: FastElement): return el.geom[0], el.geom[0] + el.geom[2]

    def _check_hit(self, el: FastElement, px: float, py: float, threshold: float):
        x, y, w, h = el.geom
        return (x - threshold <= px <= x + w + threshold) and (y - threshold <= py <= y + h + threshold)

    def rebuild(self):
        self._path_cache.clear()
        for el in self.elements.values():
            if el.id in self.hidden_ids: continue
            if el.group_id not in self._path_cache:
                self._path_cache[el.group_id] = QtGui.QPainterPath()
            self._path_cache[el.group_id].addRect(QtCore.QRectF(el.geom[0], el.geom[1], el.geom[2], el.geom[3]))
        self.update()

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        for gid, path in self._path_cache.items():
            if path.isEmpty() or gid not in self.group_colors: continue
            pen_c, brush_c = self.group_colors[gid]
            p.setPen(pg.mkPen(pen_c, width=1, cosmetic=True))
            if brush_c: p.setBrush(pg.mkBrush(brush_c))
            p.drawPath(path)

class FastCurveLayer(AbstractFastLayer):
    def _get_x_bounds(self, el: FastElement):
        pts = np.asarray(el.geom)
        return float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))

    def _check_hit(self, el: FastElement, px: float, py: float, threshold: float):
        pts = np.asarray(el.geom)
        p0, p1 = pts[:-1], pts[1:]
        dx, dy = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
        l2 = dx*dx + dy*dy
        l2[l2 == 0] = 1e-6
        t = np.clip(((px - p0[:, 0]) * dx + (py - p0[:, 1]) * dy) / l2, 0, 1)
        dist_sq = (px - (p0[:, 0] + t * dx))**2 + (py - (p0[:, 1] + t * dy))**2
        return np.min(dist_sq) <= threshold**2

    def rebuild(self):
        self._path_cache.clear()
        nan_arr = np.array([np.nan], dtype=np.float32)
        group_pts = {}
        for el in self.elements.values():
            if el.id in self.hidden_ids: continue
            pts = np.asarray(el.geom)
            if len(pts) < 2: continue
            s_pts = _chaikin_smooth(pts, iterations=2)
            group_pts.setdefault(el.group_id, [[], []])
            group_pts[el.group_id][0].extend([s_pts[:, 0], nan_arr])
            group_pts[el.group_id][1].extend([s_pts[:, 1], nan_arr])

        for gid, (xs, ys) in group_pts.items():
            self._path_cache[gid] = pg.arrayToQPath(np.concatenate(xs), np.concatenate(ys), connect='finite')
        self.update()

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for gid, path in self._path_cache.items():
            if path.isEmpty() or gid not in self.group_colors: continue
            pen_c, _ = self.group_colors[gid]
            p.setPen(pg.mkPen(pen_c, width=2, cosmetic=True))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawPath(path)

class FastPointLayer(AbstractFastLayer):
    def __init__(self, point_radius: int = 4):
        super().__init__()
        self.radius = point_radius
        self._poly_cache: dict[str, QtGui.QPolygonF] = {}

    def _get_x_bounds(self, el: FastElement): return el.geom[0], el.geom[0]

    def _check_hit(self, el: FastElement, px: float, py: float, threshold: float):
        return (el.geom[0] - px)**2 + (el.geom[1] - py)**2 <= threshold**2

    def rebuild(self):
        self._poly_cache.clear()
        for el in self.elements.values():
            if el.id in self.hidden_ids: continue
            if el.group_id not in self._poly_cache:
                self._poly_cache[el.group_id] = QtGui.QPolygonF()
            self._poly_cache[el.group_id].append(QtCore.QPointF(el.geom[0], el.geom[1]))
        self.update()

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for gid, poly in self._poly_cache.items():
            if poly.isEmpty() or gid not in self.group_colors: continue
            pen_c, brush_c = self.group_colors[gid]
            p.setPen(pg.mkPen(pen_c, width=self.radius*2, cosmetic=True, cap=QtCore.Qt.PenCapStyle.RoundCap))
            p.drawPoints(poly)

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Graphics/FastGraph/__test__/__init__.py" encoding="utf-8"
import sys
import random
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget
from PySide6.QtGui import QColor

from W.PySide6.Graphics.FastGraph import FastElement, FastRectLayer, FastCurveLayer, FastPointLayer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.resize(1000, 600)
    central = QWidget()
    win.setCentralWidget(central)
    layout = QVBoxLayout(central)

    plot_widget = pg.PlotWidget(background='#111')
    layout.addWidget(plot_widget)

    rect_layer = FastRectLayer()
    rect_layer.set_colors("group_A", QColor(0, 255, 0), QColor(0, 255, 0, 50))
    plot_widget.addItem(rect_layer)

    for i in range(100):
        x, y = random.uniform(0, 500), random.uniform(0, 100)
        w, h = random.uniform(5, 20), random.uniform(5, 15)
        rect_layer.add_element(FastElement(f"r_{i}", "group_A", [x, y, w, h]))

    rect_layer.rebuild()
    win.show()
    sys.exit(app.exec())

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Graphics/AdaptiveImage/__init__.py" encoding="utf-8"
import logging
from threading import Thread
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QTransform

class AdaptiveImageItem(pg.ImageItem):
    MIN_COLS = 64
    _sigMipmapReady = Signal()

    def __init__(self):
        super().__init__()
        self.setOpts(axisOrder='row-major')
        cmap = pg.colormap.get('viridis')
        self.setLookupTable(cmap.getLookupTable())

        self._mipmap: list[np.ndarray] = []
        self._x_range = (0.0, 1.0)
        self._y_range = (0.0, 1.0)
        self._levels = (0.0, 1.0)
        self._plot_item: pg.PlotItem | None = None
        self._updating = False
        self._sigMipmapReady.connect(self._doUpdate)

    def attachTo(self, plot_item: pg.PlotItem) -> None:
        self._plot_item = plot_item
        plot_item.getViewBox().sigRangeChanged.connect(self._doUpdate)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if ax == 0: return self._x_range
        elif ax == 1: return self._y_range
        return super().dataBounds(ax, frac, orthoRange)

    def setFullData(self, arr: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
        self._x_range = x_range
        self._y_range = y_range
        data_min = float(np.min(arr))
        data_max = float(np.max(arr))
        self._levels = (data_min, data_max if data_max > data_min else data_min + 1.0)
        self._mipmap = []
        Thread(target=self._buildMipmap, args=(arr,), daemon=True).start()

    def _buildMipmap(self, arr: np.ndarray) -> None:
        data = np.ascontiguousarray(arr, dtype=np.float32)
        levels = [data]
        cur = data
        while cur.shape[1] // 2 >= self.MIN_COLS:
            nf, nt = cur.shape
            nt2 = nt // 2
            down = cur[:, :nt2 * 2].reshape(nf, nt2, 2).mean(axis=2).astype(np.float32)
            levels.append(down)
            cur = down
            
        self._mipmap = levels
        self._sigMipmapReady.emit()

    @Slot()
    def _doUpdate(self) -> None:
        if not self._mipmap or self._plot_item is None or self._updating: return
        self._updating = True
        try: self._renderTile()
        except Exception: logging.exception("AdaptiveImageItem._renderTile error")
        finally: self._updating = False

    def _renderTile(self) -> None:
        vb = self._plot_item.getViewBox()
        [[x0, x1], [y0, y1]] = vb.viewRange()

        t_min, t_max = self._x_range
        f_min, f_max = self._y_range
        nfreq, ntime = self._mipmap[0].shape

        if ntime == 0 or nfreq == 0: return

        t_per_col = (t_max - t_min) / ntime
        f_per_row = (f_max - f_min) / nfreq

        c0 = max(0, int(np.floor((x0 - t_min) / t_per_col)))
        c1 = min(ntime, int(np.ceil((x1 - t_min) / t_per_col)) + 1)
        r0 = max(0, int(np.floor((y0 - f_min) / f_per_row)))
        r1 = min(nfreq, int(np.ceil((y1 - f_min) / f_per_row)) + 1)

        if c0 >= c1 or r0 >= r1: return

        vis_rows, vis_cols = r1 - r0, c1 - c0
        geom = vb.screenGeometry()
        scr_h = geom.height() if (geom and geom.height() > 0) else 600
        scr_w = geom.width() if (geom and geom.width() > 0) else 1200

        tile_h = max(1, min(vis_rows, scr_h))
        tile_w = max(1, min(vis_cols, scr_w))

        col_step = vis_cols / tile_w
        k = min(max(0, int(np.floor(np.log2(max(1.0, col_step))))), len(self._mipmap) - 1)
        s = 1 << k
        level = self._mipmap[k]
        lw = level.shape[1]

        lc0 = c0 // s
        lc1 = min(lw, (c1 + s - 1) // s + 1)
        c0_a = lc0 * s

        crop = level[r0:r1, lc0:lc1]
        ch, cw = crop.shape
        bh, bw = max(1, ch // tile_h), max(1, cw // tile_w)

        if bh > 1 or bw > 1:
            out_h, out_w = ch // bh, cw // bw
            sub = crop[:bh * out_h, :bw * out_w].reshape(out_h, bh, out_w, bw).mean(axis=(1, 3)).astype(np.float32)
        else:
            sub = crop

        self.setImage(sub, autoLevels=False, levels=self._levels)

        tr = QTransform()
        tr.translate(t_min + c0_a * t_per_col, f_min + r0 * f_per_row)
        tr.scale(bw * s * t_per_col, bh * f_per_row)
        self.setTransform(tr)

``````

``````py path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/W/PySide6/Graphics/AdaptiveImage/__test__/__init__.py" encoding="utf-8"
import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow
from W.PySide6.Graphics.AdaptiveImage import AdaptiveImageItem

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.resize(800, 600)

    plot_widget = pg.PlotWidget()
    win.setCentralWidget(plot_widget)
    plot_item = plot_widget.getPlotItem()

    img_item = AdaptiveImageItem()
    plot_item.addItem(img_item)
    img_item.attachTo(plot_item)

    data = np.random.randn(100, 1000).astype(np.float32)
    img_item.setFullData(data, x_range=(0.0, 10.0), y_range=(0.0, 100.0))

    win.show()
    sys.exit(app.exec())

``````