import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QGroupBox)
from PySide6.QtCore import Qt

from BatSpec.QtApp.Shared.Component import Component, GLOBAL_MARGINS
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales

from ..Logic import RECORD_STATE
from .Logic import GRAPH_SETTINGS
from qt_pg_theme import ThemedPlotWidget


class ModeSelector(Locales.Trigger, RECORD_STATE.Trigger, QtWidgets.QComboBox):
    """
    UI Hierarchy:
    ModeSelector (QComboBox)
    """
    def __init__(self):
        QtWidgets.QComboBox.__init__(self)
        self.addItem(self.tr("Raw (original)"))
        self.addItem(self.tr("Centered (median = 0)"))
        self.setCurrentIndex(GRAPH_SETTINGS.mode)

        self.currentIndexChanged.connect(self.currentIndexChangedToSettings)
        
    def onLanguageChange(self):
        self.setItemText(0, self.tr("Raw (original)"))
        self.setItemText(1, self.tr("Centered (median = 0)"))

    def currentIndexChangedToSettings(self):
        GRAPH_SETTINGS.mode = self.currentIndex()


class RawGraph(*Component, RECORD_STATE.Trigger, QGroupBox):
    """
    UI Hierarchy:
    RawGraph (QGroupBox)
    └── self.main_layout : QVBoxLayout
        ├── self.controls_layout : QHBoxLayout
        │   ├── self.display_label : QLabel
        │   ├── self.mode_selector : ModeSelector
        │   └── [Stretch]
        │
        └── self.plot_widget : ThemedPlotWidget
            └── self.plot_curve : pg.PlotDataItem (Actual line drawn on the plot)
    """

    def __init__(self, parent: QWidget | None = None):
        QGroupBox.__init__(self, parent)

        # Main vertical layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(*GLOBAL_MARGINS)

        # Horizontal layout for top controls
        self.controls_layout = QHBoxLayout()
        self.controls_layout.setContentsMargins(0, 0, 0, 4)

        self.display_label = QLabel(self.tr("Display:"))
        self.controls_layout.addWidget(self.display_label)

        self.mode_selector = ModeSelector()
        self.controls_layout.addWidget(self.mode_selector)

        self.controls_layout.addStretch()
        self.main_layout.addLayout(self.controls_layout)

        # Plot widget setup
        self.plot_widget = ThemedPlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.main_layout.addWidget(self.plot_widget)

        self.plot_curve = self.plot_widget.plot()
        self.plot_curve.setDownsampling(ds=True, auto=True, method='peak')
        self.plot_curve.setClipToView(True)

        self.mode_selector.currentIndexChanged.connect(self.updatePlotAll)
        self.updatePlotAll()

    def clearPlot(self):
        self.plot_curve.setData([], [])
        self.plot_widget.setTitle(self.tr("No data"))

    def updatePlotAll(self):
        self.updatePlotData()
        self.updatePlotText()

    def updatePlotText(self):
        mode = self.mode_selector.currentIndex()

        if mode == 0:
            s = self.tr("(original)")
        elif mode == 1:
            s = self.tr("(DC offset removed)")
        else:
            raise RuntimeError()
        
        self.plot_widget.setTitle(self.tr("Oscillogram") + " " + s)
        self.plot_widget.update()

    def updatePlotData(self):
        if not RECORD_STATE.record: 
            return self.clearPlot()

        mode = self.mode_selector.currentIndex()

        if mode == 0:
            f = RECORD_STATE.record.row
            s = self.tr("(original)")
        elif mode == 1:
            f = RECORD_STATE.record.corected
            s = self.tr("(DC offset removed)")
        else:
            raise RuntimeError()

        self.plot_curve.setData(x=f.time_axis, y=f.data)

        self.plot_widget.setXRange(0, f.duration, padding=0.01)

        ymin = float(f.data.min())
        ymax = float(f.data.max())
        yrange = ymax - ymin
        pad = max(0.08, yrange * 0.12)
        self.plot_widget.setYRange(ymin - pad, ymax + pad)

        self.plot_widget.update()

    def onRecordChange(self):
        self.updatePlotData()

    def onLanguageChange(self):
        self.display_label.setText(self.tr("Display:"))

        pi = self.plot_widget.getPlotItem()
        pi.setLabel('left',  self.tr("Amplitude"), units='FS')
        pi.setLabel('bottom', self.tr("Time"),    units='s')

        self.updatePlotText()

    def onThemeChange(self):
        self.plot_curve.setPen(pg.mkPen(Themes.get_curent_theme().primary, width=1))