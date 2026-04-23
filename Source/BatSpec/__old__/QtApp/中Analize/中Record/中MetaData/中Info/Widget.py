import os
import numpy as np
import scipy.signal as sig
import pyqtgraph as pg
from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                               QSpinBox, QPushButton, QFormLayout, QSplitter, 
                               QTextBrowser, QTabWidget, QLabel, QGroupBox,
                               QGridLayout)
from PySide6.QtCore import Qt, Slot

from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component, GLOBAL_MARGINS

from ...Logic import RECORD_STATE


class Info(*Component, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)
        
        self.l = QVBoxLayout(self)
        self.l.setContentsMargins(*GLOBAL_MARGINS)
        
        # --- Группа информации ---
        self.group = QGroupBox()
        self.layout = QGridLayout(self.group)
        
        self.lbl_status_title = QLabel()
        self.lbl_status = QLabel()
        
        self.lbl_samplerate_title = QLabel()
        self.lbl_samplerate = QLabel()
        
        self.lbl_duration_title = QLabel()
        self.lbl_duration = QLabel()
        
        self.lbl_samples_title = QLabel()
        self.lbl_samples = QLabel()
        
        self.lbl_channels_title = QLabel()
        self.lbl_channels = QLabel()
        
        self.lbl_range_title = QLabel()
        self.lbl_range = QLabel()
        
        # Новые аналитические данные
        self.lbl_energy_title = QLabel()
        self.lbl_energy = QLabel()
        
        self.lbl_median_title = QLabel()
        self.lbl_median = QLabel()
        
        self.lbl_std_title = QLabel()
        self.lbl_std = QLabel()
        
        self.layout.addWidget(self.lbl_status_title, 0, 0)
        self.layout.addWidget(self.lbl_status, 0, 1)
        self.layout.addWidget(self.lbl_samplerate_title, 1, 0)
        self.layout.addWidget(self.lbl_samplerate, 1, 1)
        self.layout.addWidget(self.lbl_duration_title, 2, 0)
        self.layout.addWidget(self.lbl_duration, 2, 1)
        self.layout.addWidget(self.lbl_samples_title, 3, 0)
        self.layout.addWidget(self.lbl_samples, 3, 1)
        self.layout.addWidget(self.lbl_channels_title, 4, 0)
        self.layout.addWidget(self.lbl_channels, 4, 1)
        self.layout.addWidget(self.lbl_range_title, 5, 0)
        self.layout.addWidget(self.lbl_range, 5, 1)
        
        # Добавляем новые поля в сетку
        self.layout.addWidget(self.lbl_energy_title, 6, 0)
        self.layout.addWidget(self.lbl_energy, 6, 1)
        self.layout.addWidget(self.lbl_median_title, 7, 0)
        self.layout.addWidget(self.lbl_median, 7, 1)
        self.layout.addWidget(self.lbl_std_title, 8, 0)
        self.layout.addWidget(self.lbl_std, 8, 1)
        
        self.layout.setColumnStretch(1, 1)
        self.l.addWidget(self.group)
        
        # --- Кнопка выгрузки записи ---
        self.btn_unload = QPushButton()
        self.btn_unload.clicked.connect(self.on_unload_clicked)
        self.l.addWidget(self.btn_unload)
        
        self.l.addStretch()
        
        RECORD_STATE.signals.recordChanged.connect(self.update_info)
        
        self.update_info()
    
    @Slot()
    def update_info(self):
        """Обновляет информацию при изменении записи"""
        record = RECORD_STATE.record
        
        if record is not None:
            data = record.corected.data
            sr = record.samplerate
            
            self.lbl_status.setText(self.tr("Loaded"))
            self.lbl_samplerate.setText(f"{sr} Hz")
            self.lbl_duration.setText(f"{record.duration:.2f} {self.tr('sec')}")
            self.lbl_samples.setText(f"{len(data)}")
            
            # Берём оригинальное количество каналов (до сведения в моно)
            channels = getattr(record, 'original_channels', 1)
            self.lbl_channels.setText(f"{channels}")
            
            self.lbl_range.setText(f"[{data.min():.4f}, {data.max():.4f}]")
            
            # --- Расчет аналитики ---
            # Энергия: Интеграл квадрата сигнала (Сумма(x^2) * dt)
            energy = np.sum(data**2) / sr
            median = np.median(data)
            std = np.std(data)
            
            self.lbl_energy.setText(f"{energy:.4f}")
            self.lbl_median.setText(f"{median:.6f}")
            self.lbl_std.setText(f"{std:.6f}")
            
            self.btn_unload.setEnabled(True)
        else:
            self.lbl_status.setText(self.tr("No data"))
            self.lbl_samplerate.setText("-")
            self.lbl_duration.setText("-")
            self.lbl_samples.setText("-")
            self.lbl_channels.setText("-")
            self.lbl_range.setText("-")
            
            self.lbl_energy.setText("-")
            self.lbl_median.setText("-")
            self.lbl_std.setText("-")
            
            self.btn_unload.setEnabled(False)
            
    @Slot()
    def on_unload_clicked(self):
        RECORD_STATE.setRecordFile(None)
    
    def onLanguageChange(self):
        self.setTabName(self.tr("Info"))
        self.setTabToolTip(self.tr("Information about loaded recording"))
        
        self.group.setTitle(self.tr("Recording Information"))
        self.lbl_status_title.setText(self.tr("Status:"))
        self.lbl_samplerate_title.setText(self.tr("Sample rate:"))
        self.lbl_duration_title.setText(self.tr("Duration:"))
        self.lbl_samples_title.setText(self.tr("Samples:"))
        self.lbl_channels_title.setText(self.tr("Original channels:"))
        self.lbl_range_title.setText(self.tr("Value range:"))
        
        self.lbl_energy_title.setText(self.tr("Energy (∫x² dt):"))
        self.lbl_median_title.setText(self.tr("Median:"))
        self.lbl_std_title.setText(self.tr("Standard Deviation:"))
        
        self.btn_unload.setText(self.tr("Unload recording"))
        
        self.update_info()
    
    def onThemeChange(self):
        # Если необходимо стилизовать кнопку выгрузки (например, в красный цвет)
        pass