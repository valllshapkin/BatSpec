from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, 
                               QTableWidgetItem, QGroupBox, QHeaderView,
                               QTabWidget)
from PySide6.QtCore import Slot
import logging

from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtApp.Shared.Component import Component, GLOBAL_MARGINS
from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes

from ...Logic import RECORD_STATE

logger = logging.getLogger(__name__)

from guano import GuanoFile



class Guano(*Component, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)
        
        self.l = QVBoxLayout(self)
        self.l.setContentsMargins(*GLOBAL_MARGINS)
        
        self.group = QGroupBox()
        self.layout = QVBoxLayout(self.group)
        
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        
        self.layout.addWidget(self.table)
        self.l.addWidget(self.group)
        self.l.addStretch()
        
        RECORD_STATE.signals.recordChanged.connect(self.update_metadata)
        
        self.update_metadata()
    
    @Slot()
    def update_metadata(self):            
        record = RECORD_STATE.record
        
        if not record:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(self.tr("Info")))
            self.table.setItem(0, 1, QTableWidgetItem(self.tr("No recording or file path available")))
            return
        
        try:
            g = GuanoFile(str(record.path))
            items = list(g.items())
            
            if not items:
                self.table.setRowCount(1)
                self.table.setItem(0, 0, QTableWidgetItem(self.tr("Info")))
                self.table.setItem(0, 1, QTableWidgetItem(self.tr("No GUANO metadata found")))
                return
            
            self.table.setRowCount(len(items))
            
            for i, (key, value) in enumerate(items):
                self.table.setItem(i, 0, QTableWidgetItem(str(key)))
                self.table.setItem(i, 1, QTableWidgetItem(str(value)))
            
            self.table.resizeColumnsToContents()
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            
        except Exception as e:
            logger.exception("Error reading GUANO metadata")
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(self.tr("Error")))
            self.table.setItem(0, 1, QTableWidgetItem(str(e)))
    
    def onLanguageChange(self):
        self.setTabName(self.tr("GUANO"))
        self.setTabToolTip(self.tr("GUANO metadata"))
        
        self.group.setTitle(self.tr("GUANO Metadata"))
        self.table.setHorizontalHeaderLabels([self.tr("Key"), self.tr("Value")])
        
        self.update_metadata()
    
    def onThemeChange(self):
        pass