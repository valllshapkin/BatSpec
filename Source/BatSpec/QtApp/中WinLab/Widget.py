import os
import scipy.signal as sig
import pyqtgraph as pg
from pathlib import Path
from PySide6.QtWidgets import (QFileDialog, QFileSystemModel, QTreeView, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                               QSpinBox, QPushButton, QFormLayout, QSplitter, 
                               QTextBrowser, QTabWidget, QLabel, QStackedWidget)
from PySide6.QtCore import QDir, Qt

from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component, GLOBAL_MARGINS

class WindowLabTab(*Component, TabInstance):

    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)

        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(*GLOBAL_MARGINS)

        self.vL = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)

    def onLanguageChange(self):
        self.setTabName(self.tr("Window Lab")) if self.tab_widget else None
        self.vL.setText("Window Lab no impplemented")

    def onThemeChange(self):
        pass