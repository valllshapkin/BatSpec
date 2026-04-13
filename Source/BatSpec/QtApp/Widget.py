from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from BatSpec.QtApp.中Menu.Widget import AddMenu

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component, GLOBAL_MARGINS
from .中Project.Widget import ProjectTab
from .中WinLab.Widget import WindowLabTab
from .中Analize.Widget import RecordAnalisys

class MainWindow(AddMenu, *Component, QMainWindow):
    def __init__(self) -> None:
        AddMenu.__init__(self)
    
        self.c = QWidget()
        self.setCentralWidget(self.c)
        self.cV = QVBoxLayout(self.c)
        self.cV.setContentsMargins(*GLOBAL_MARGINS)

        

        # Создаем главный контейнер для модулей-страниц
        self.main_tabs = QTabWidget()
        self.cV.addWidget(self.main_tabs)

        # Подключаем модули
        self.project = ProjectTab(tab_widget=self.main_tabs)
        self.window_lab = WindowLabTab(tab_widget=self.main_tabs)
        # self.analysis_space = AnalysisSpaceTab(tab_widget=self.main_tabs)
        self.record_analisis = RecordAnalisys(tab_widget=self.main_tabs)


    def onLanguageChange(self):
        self.setWindowTitle(self.tr("BatSpec"))

    def onThemeChange(self):
        pass



    



