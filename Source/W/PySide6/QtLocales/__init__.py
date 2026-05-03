import os
import inspect
import random
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QLocale, QObject, QTranslator
from PySide6 import QtCore # type: ignore

# Подключаем наш новый мощный фреймворк настроек
from W.PySide6.QtSettings import Settings as SettingsProtocol, Field, LocalesAdapter

class Locales:
    _active_instances: list['Locales'] = []

    # Наследуемся от QObject, чтобы магия создания сигналов в Field сработала!
    class Settings(QObject, SettingsProtocol):
        # Вся логика, адаптеры и сигналы теперь в одной строке!
        locales, locales_updated = Field(
            default=[QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)],
            adapter=LocalesAdapter(),
            key="Locales"
        )

        def __init__(self, this: 'Locales', SETTING_PATH: Path) -> None:
            super().__init__()
            self.this = this
            self._settings = QtCore.QSettings(
                str(SETTING_PATH), 
                QtCore.QSettings.Format.IniFormat
            )
        
        @property
        def settings(self) -> QtCore.QSettings | None:
            return self._settings

    class API:
        def __init__(self, this: 'Locales') -> None:
            self.this = this
            self._supported_languages: list[QLocale] = []

        @property
        def current_locales(self) -> list[QLocale]:
            return self.this.settings.locales

        @property
        def supported_languages(self) -> list[QLocale]:
            return self._supported_languages

        def set_supported_languages(self, languages: list[QLocale]) -> None:
            self._supported_languages = languages

        def change_language(self, new_locales: list[QLocale]) -> None:
            # Дескриптор сам отправит сигнал locales_updated
            self.this.settings.locales = new_locales

    class TranslateComponent:
        """
        Статичный класс для тайпчекеров.
        Перехватывает инициализацию и регистрирует JIT/пути переводов 
        во всех активных инстансах Locales.
        """
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
            
            cls.__init__ = new_init # type: ignore

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
            if path.exists():
                print(f"[JIT Locales] Compiling {path}")
                os.system(str(path))
            else:
                print(f"[JIT Locales] No File {path}")

        def reload_translators(self, locales: list[QLocale]) -> None:
            app = QApplication.instance()
            if not app: return

            for t in self.active_translators:
                app.removeTranslator(t)
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
                if not supported:
                    return

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
        
        # Подписываемся напрямую на сигнал дескриптора
        self.settings.locales_updated.connect(self.wrapper.reload_translators)
        
        # Первичная загрузка
        self.wrapper.reload_translators(self.api.current_locales)

        if debug:
            self.debug.start_random_timer()
