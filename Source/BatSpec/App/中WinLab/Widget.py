from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PySide6.QtCore import Qt

from BatSpec.QtUp.TabInst import TabInstance
from BatSpec.QtUp.Locales import Locales
from BatSpec.QtUp.Themes import Themes
from BatSpec.QtUp.Builder import build_node as b

class WindowLabTab(Locales.TranslateComponent, Locales.Trigger, Themes.Trigger, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)

        with b(self, QVBoxLayout()) as self.v:
            self.v.setContentsMargins(4, 0, 4, 0)
            
            with b(self.v, QLabel()) as self.vL:
                self.vL.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def onLanguageChange(self):
        if self.tab_widget: 
            self.setTabName(self.tr("Window Lab"))
        self.vL.setText(self.tr("Window Lab not implemented"))
        super().onLanguageChange()

    def onThemeChange(self):
        super().onThemeChange()
