from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QDialog

from W.PySide6.App.Locales import Locales
from W.PySide6.Core.Lifecycle import ComponentLifecycle
from W.PySide6.Core.Builder import build_node as b
from W.PySide6.Widgets.Frameless import FramelessMixin

from BatSpec.App.中Menu.中App.中Settings.中LocalesTab.Widget import LocalesSettingsTab
from BatSpec.App.中Menu.中App.中Settings.中ThemesTab.Widget import ThemeSettingsTab

class SettingsDialog(Locales.TranslateComponent, ComponentLifecycle, FramelessMixin, QDialog):
    def __init_state__(self):
        self.setModal(True)
        self.resize(550, 450)
        self.init_frameless(fallback_icon="⚙️")
        
    def __init_graph__(self):
        content_layout = self.build_frameless_ui()
        
        with b(content_layout, QTabWidget()) as self.lT:
            self.lT.setUsesScrollButtons(True)
            self.lT1 = ThemeSettingsTab(tab_widget=self.lT)
            self.lT2 = LocalesSettingsTab(tab_widget=self.lT)

    def setWindowTitle(self, title: str):
        super().setWindowTitle(title)
        if hasattr(self, 'title_bar'):
            self.title_bar.update_title(title)

    def onThemeChange(self):
        self.update_frameless_theme()
        super().onThemeChange()

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("Settings"))
        super().onLanguageChange()
