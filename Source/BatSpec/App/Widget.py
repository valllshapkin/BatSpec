from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

from BatSpec.App.中Menu.Widget import AddMenu
from W.PySide6.QtBuilder import build_node as b
from W.PySide6.QtFrameless import FramelessMixin

from BatSpec.App.中Project.Widget import ProjectTab
from BatSpec.App.中WinLab.Widget import WindowLabTab

ScriptDir = Path(__file__).parent
ICON_PATH = ScriptDir / "__assets__" / "bat.ico"

class MainWindow(FramelessMixin, AddMenu):
    def __init_state__(self):
        if hasattr(super(), '__init_state__'):
            super().__init_state__()
            
        # Инициализируем безрамочное окно и передаем путь к иконке
        self.init_frameless(icon_path=ICON_PATH, fallback_icon="🦇")

    def __init_graph__(self):
        if hasattr(super(), '__init_graph__'):
            super().__init_graph__()
            
        # МАГИЯ: Одна строчка строит тень, скругления, фон и шапку!
        # Возвращает готовый слой для контента.
        content_layout = self.build_frameless_ui()
        
        # Делаем фон QMenuBar прозрачным, чтобы он красиво лег в шапку
        self.menuBar().setStyleSheet("background: transparent;")
        
        # Встраиваем меню в шапку (в спец. выровненный контейнер)
        self.title_bar.custom_layout.addWidget(self.menuBar())
        
        # Наполняем основной контент
        with b(content_layout, QTabWidget()) as self.main_tabs:
            self.project = ProjectTab(tab_widget=self.main_tabs)
            self.window_lab = WindowLabTab(tab_widget=self.main_tabs)

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
