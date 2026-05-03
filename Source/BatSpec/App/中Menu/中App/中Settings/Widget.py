from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QDialog

from BatSpec.QtUp.Locales import Locales
from BatSpec.QtUp.Themes import Themes
from BatSpec.QtUp.Builder import build_node as b

from BatSpec.App.中Menu.中App.中Settings.中LocalesTab.Widget import LocalesSettingsTab
from BatSpec.App.中Menu.中App.中Settings.中ThemesTab.Widget import ThemeSettingsTab

class SettingsDialog(Locales.TranslateComponent, Locales.Trigger, Themes.Trigger, QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setModal(True)
        self.resize(500, 400)
        
        with b(self, QVBoxLayout()) as self.l:
            with b(self.l, QTabWidget()) as self.lT:
                self.lT.setUsesScrollButtons(True)
                
                self.lT1 = ThemeSettingsTab(tab_widget=self.lT)
                self.lT2 = LocalesSettingsTab(tab_widget=self.lT)

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("Settings"))
        super().onLanguageChange()

    def onThemeChange(self):
        super().onThemeChange()
