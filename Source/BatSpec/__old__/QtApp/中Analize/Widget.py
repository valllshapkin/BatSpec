import os
import scipy.signal as sig
import pyqtgraph as pg
from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                               QSpinBox, QPushButton, QFormLayout, QSplitter, 
                               QTextBrowser, QTabWidget, QLabel)
from PySide6.QtCore import Qt

from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component, GLOBAL_MARGINS
from .中Record.Widget import Record
from .中STFT.Widget import StftModelTab
from .中Detection.Widget import AutoDetectWidget
from .中SpecView.Widget import SpecViewWidget

from BatSpec.QtUp.TabInst import TabInstance

class SpecViewTab(TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        super().__init__(parent, tab_widget)
        self.l = QVBoxLayout(self)
        self.l.addWidget(SpecViewWidget())

class AutoDetectWidgetTab(TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        super().__init__(parent, tab_widget)
        self.l = QVBoxLayout(self)
        self.l.addWidget(AutoDetectWidget())

class RecordAnalisys(*Component, TabInstance):

    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)

        self.l = QVBoxLayout(self)
        self.l.setContentsMargins(*GLOBAL_MARGINS)


        self.lT = QTabWidget()
        # self.lT.setContentsMargins(0, 0, 0, 0)
        self.l.addWidget(self.lT)

        self.lTR = Record(self, self.lT)
        self.lRM = StftModelTab(self, self.lT)
        self.stft_graph = SpecViewTab(self, self.lT)
        self.lProc = AutoDetectWidgetTab(self, self.lT)  



    def onLanguageChange(self):
        self.setTabName(self.tr("Record Analisys"))
        self.setTabToolTip(self.tr("..."))

    def onThemeChange(self):
        pass
