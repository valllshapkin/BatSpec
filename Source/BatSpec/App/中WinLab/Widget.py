from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PySide6.QtCore import Qt

from W.PySide6.App.Locales import Locales
from W.PySide6.Core.Lifecycle import ComponentLifecycle
from W.PySide6.Widgets.TabInstance import TabInstance
from W.PySide6.Core.Builder import build_node as b

class WindowLabTab(Locales.TranslateComponent, ComponentLifecycle, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        super().__init__(parent, tab_widget)

    def __init_graph__(self):
        with b(self, QVBoxLayout()) as self.v:
            self.v.setContentsMargins(4, 0, 4, 0)
            
            with b(self.v, QLabel()) as self.vL:
                self.vL.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def onLanguageChange(self):
        if self.tab_widget: 
            self.setTabName(self.tr("Window Lab"))
        self.vL.setText(self.tr("Window Lab not implemented"))
        super().onLanguageChange()
