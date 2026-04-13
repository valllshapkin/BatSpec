import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                               QSplitter, 
                               QTabWidget, QLabel, QGroupBox)
from PySide6.QtCore import Qt


from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component, GLOBAL_MARGINS




from .中Import.Widget import Import
from .中Load.Widget import Load
from .中MetaData.Widget import MetaData
from .中Graph.Widget import RawGraph
from .Logic import RECORD_STATE
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox


class Record(*Component, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)

        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(*GLOBAL_MARGINS)

        # главный вертикальный сплиттер
        self.vV = QSplitter(Qt.Orientation.Vertical)
        self.v.addWidget(self.vV)

        # верхняя горизонтальная часть (вкладки + метаданные)
        self.vVH = QSplitter(Qt.Orientation.Horizontal)
        self.vV.addWidget(self.vVH)

        # вкладки импорта / загрузки
        self.vVHT = QTabWidget()
        self.vVH.addWidget(self.vVHT)

        self.vVHTI = Import(tab_widget=self.vVHT)

        self.vVHTL = Load(tab_widget=self.vVHT)

        # блок метаданных справа
        self.vVHM = MetaData(self)
        self.vVH.addWidget(self.vVHM)

        self.vVH.setSizes([300, 300])

        # нижняя часть — график
        self.vVG = RawGraph(self)
        self.vV.addWidget(self.vVG)

        self.vV.setSizes([400, 600])

    def onLanguageChange(self):
        self.setTabName(self.tr("Record"))
        self.setTabToolTip(self.tr("Record management"))

    def onThemeChange(self):
        pass