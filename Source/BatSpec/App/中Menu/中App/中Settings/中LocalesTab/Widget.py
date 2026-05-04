from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from BatSpec.App.中Menu.中LocalesSettings.Widget import LocalesSettingsPanel
from W.PySide6.Widgets.TabInstance import TabInstance

from W.PySide6.App.Locales import Locales
from W.PySide6.Core.Lifecycle import ComponentLifecycle
from W.PySide6.Core.Builder import build_node as b

class LocalesSettingsTab(Locales.TranslateComponent, ComponentLifecycle, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        super().__init__(parent, tab_widget)
        
    def __init_graph__(self):
        self.setContentsMargins(0, 0, 0, 0)
        with b(self, QVBoxLayout()) as self.v:
            with b(self.v, LocalesSettingsPanel()) as self.panel: pass
    
    def onLanguageChange(self):
        if self.tab_widget:
            self.setTabName(self.tr("Locales"))
        super().onLanguageChange()
