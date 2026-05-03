from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from BatSpec.App.中Menu.中LocalesSettings.Widget import LocalesSettingsPanel
from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtUp.Locales import Locales
from BatSpec.QtUp.Themes import Themes
from BatSpec.QtUp.Builder import build_node as b

class LocalesSettingsTab(Locales.TranslateComponent, Locales.Trigger, Themes.Trigger, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)
        self.setContentsMargins(0, 0, 0, 0)
        
        with b(self, QVBoxLayout()) as self.v:
            with b(self.v, LocalesSettingsPanel()) as self.panel: pass
    
    def onLanguageChange(self):
        if self.tab_widget:
            self.setTabName(self.tr("Locales"))
        super().onLanguageChange()
