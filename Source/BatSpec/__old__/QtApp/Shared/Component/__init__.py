
from ..中LocalesSettings.Logic import Locales
from ..中ThemeSettings.Logic import Themes
from BatSpec.QtUp.Settings import Settings as SettingsProtocol

from typing import TYPE_CHECKING
from PySide6 import QtCore
from pathlib import Path


class ComponentSettings(QtCore.QObject, SettingsProtocol if TYPE_CHECKING else object):
    COMPONENT_DIR: Path

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self, "COMPONENT_DIR"): raise NotImplementedError()

        self.settings = QtCore.QSettings(
            str(self.COMPONENT_DIR / "__assets__" / "settings.ini"), 
            QtCore.QSettings.Format.IniFormat
        )


Component = Locales.TranslateComponent, Locales.Trigger, Themes.Trigger

GLOBAL_MARGINS = (4, 0, 4, 0)