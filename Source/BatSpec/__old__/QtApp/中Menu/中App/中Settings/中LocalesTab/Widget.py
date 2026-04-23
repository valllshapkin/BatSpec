from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from BatSpec.QtApp.Shared.中LocalesSettings.Widget import LocalesSettingsPanel
from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component


class LocalesSettingsTab(*Component, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)
        self.setContentsMargins(0, 0, 0, 0)
        self.v = QVBoxLayout()
        self.setLayout(self.v)
        self.panel = LocalesSettingsPanel()
        self.v.addWidget(self.panel)
    
    def onLanguageChange(self):
        self.setTabName(self.tr("Locales"))

