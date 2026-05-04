from pathlib import Path
from PySide6.QtWidgets import QTabWidget

from BatSpec.App.中Menu.Widget import AddMenu
from W.PySide6.Core.Builder import build_node as b
from W.PySide6.Widgets.Frameless import FramelessMixin

from BatSpec.App.中Project.Widget import ProjectTab
from BatSpec.App.中WinLab.Widget import WindowLabTab
from BatSpec.App.中Annotation.Widget import AnnotationTab
from BatSpec.App.中Annotation.中Lookups.Widget import LookupsTab

ScriptDir = Path(__file__).parent
ICON_PATH = ScriptDir / "__assets__" / "bat.ico"

class MainWindow(FramelessMixin, AddMenu):
    def __init_state__(self):
        if hasattr(super(), '__init_state__'):
            super().__init_state__()
            
        self.init_frameless(icon_path=ICON_PATH, fallback_icon="🦇")

    def __init_graph__(self):
        if hasattr(super(), '__init_graph__'):
            super().__init_graph__()
            
        content_layout = self.build_frameless_ui()
        self.menuBar().setStyleSheet("background: transparent;")
        self.title_bar.custom_layout.addWidget(self.menuBar())
        
        with b(content_layout, QTabWidget()) as self.main_tabs:
            self.project = ProjectTab(tab_widget=self.main_tabs)
            self.window_lab = WindowLabTab(tab_widget=self.main_tabs)
            self.annotation_tab = AnnotationTab(tab_widget=self.main_tabs)
            self.lookups_tab = LookupsTab(tab_widget=self.main_tabs)

    def setWindowTitle(self, title: str):
        super().setWindowTitle(title)
        if hasattr(self, 'title_bar'):
            self.title_bar.update_title(title)

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("BatSpec"))
        super().onLanguageChange()

    def onThemeChange(self):
        self.update_frameless_theme()
        super().onThemeChange()
