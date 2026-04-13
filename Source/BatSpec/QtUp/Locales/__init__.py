import inspect

from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import QEvent, QLocale, QObject, QTranslator, Signal
from PySide6 import QtCore
from typing import TYPE_CHECKING, Self, Any, List
from BatSpec.Python import delete_init
from BatSpec.QtUp.Settings import Settings as SettingsProtocol, LocalesField
from pathlib import Path
from typing import List
from contextlib import nullcontext as Block

DEBUG = False
JIT_TRANSLATE_COMPILATION = False
SUPPORTED_LANGUAGES: List[QLocale] = []

class Settings(QtCore.QObject, SettingsProtocol if TYPE_CHECKING else object):
    SETTING_PATH: str

    locales = LocalesField(
        default=[QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)],
        key="Locales"
    )
    locales_update = QtCore.Signal(list)

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self, "SETTING_PATH"): raise NotImplementedError()

        self.settings = QtCore.QSettings(
            self.SETTING_PATH, 
            QtCore.QSettings.Format.IniFormat
        )

    def change_language(self, new_locales: List[QLocale]):
        self.locales = new_locales
        self.locales_update.emit(new_locales)
    

SETTINGS: Settings

class TranslatorModule: # Возможно стоит сделать QObject чтобы parent удалял его когда нужно
    translator: QTranslator

    def __init__(self, qmfolder: Path):
        SETTINGS.locales_update.connect(self.load_translators)
        self.path = qmfolder
        self.translators = []
        self.load_translators(SETTINGS.locales)

    def load_translators(self, locales: List[QLocale]):
        for i in self.translators: 
            QApplication.removeTranslator(i)

        self.translators = []
        for locale in locales:
            translator = QTranslator()
            translator.load(locale, "this", "_", str(self.path), ".qm")
            QApplication.installTranslator(translator)
            self.translators.append(translator)


@delete_init
class Trigger(QWidget if TYPE_CHECKING else object):
    def __init_subclass__(cls):

        old_changeEvent = cls.changeEvent
        def new_changeEvent(self: Self, event: QtCore.QEvent):
            if event.type() == QEvent.Type.LanguageChange:
                self.onLanguageChange()
            old_changeEvent(self, event)
        cls.changeEvent = new_changeEvent

        old_init = cls.__init__
        def new_init(self: Self, *args: Any, **kwargs: Any):
            old_init(self, *args, **kwargs)
            self.onLanguageChange()
        cls.__init__ = new_init

        return super().__init_subclass__()

    def onLanguageChange(self):
        pass

class TranslateComponent:
    translatorModule: TranslatorModule | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        module = inspect.getmodule(cls)
        

        if JIT_TRANSLATE_COMPILATION:
            if not module or not module.__file__: raise RuntimeError()
            path = Path(module.__file__).parent / "__script__" / "loc.comp.sh"
            print(f"[JIT Locales] {path}"); import os; os.system(str(path))
    

        old_init = cls.__init__
        def new_init(self, *args, **kwargs):
            if not cls.translatorModule:
                if not module or not module.__file__: raise RuntimeError()
                cls.translatorModule = TranslatorModule(
                    Path(module.__file__).parent / "__assets__" / "translations"
                )
            old_init(self, *args, **kwargs)
        cls.__init__ = new_init


def initLocales():
    SETTINGS.locales_update.emit(SETTINGS.locales)
    debug_locales_timer() if DEBUG else None

with Block("DEBUG"):
    def debug_locales_timer():
        import random
        from PySide6 import QtCore, QtWidgets
        timer = QtCore.QTimer(parent=QtWidgets.QApplication.instance())
        timer.setInterval(1000)

        @timer.timeout.connect
        def _():
            SETTINGS.change_language([
                QLocale(QLocale.Language.English, QLocale.Country.UnitedStates),
                random.choice(SUPPORTED_LANGUAGES)
            ])
        
        timer.start()
