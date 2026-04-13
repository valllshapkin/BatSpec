from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
                             QPlainTextEdit, QPushButton, QSpinBox, QLabel, QLineEdit)
from PySide6.QtCore import Qt
import pyqtgraph as pg
import numpy as np
from .Logic import CORP_STATE

class CropVisualizer(pg.ImageView, CORP_STATE.Trigger):
    """Отображает только вырезанный кусок (Raw Crop)"""
    def __init__(self):
        super().__init__()
        self.ui.histogram.hide()
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()

    def onResultUpdate(self):
        if CORP_STATE.cropped_matrix is not None:
            self.setImage(CORP_STATE.cropped_matrix.T)

class ResultVisualizer(pg.PlotWidget, CORP_STATE.Trigger):
    """Отображает кусок + наложенную кривую аппроксимации"""
    def __init__(self):
        super().__init__()
        self.img_item = pg.ImageItem()
        self.addItem(self.img_item)
        self.curve_item = pg.PlotDataItem(pen=pg.mkPen('r', width=2), symbol='o', symbolSize=3)
        self.addItem(self.curve_item)

    def onResultUpdate(self):
        if CORP_STATE.cropped_matrix is not None:
            self.img_item.setImage(CORP_STATE.cropped_matrix.T)
            if CORP_STATE.result_data:
                # result_data - список кортежей (x, y)
                pts = np.array(CORP_STATE.result_data)
                self.curve_item.setData(pts[:, 0], pts[:, 1])
            else:
                self.curve_item.setData([], [])

class ControlPanel(QWidget, CORP_STATE.Trigger):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # ROI Debug Info
        self.roi_info = QLineEdit()
        self.roi_info.setReadOnly(True)
        layout.addWidget(QLabel("Current ROI:"))
        layout.addWidget(self.roi_info)
        
        # Padding
        layout.addWidget(QLabel("Padding (bins):"))
        self.pad_spin = QSpinBox()
        self.pad_spin.setRange(0, 100)
        self.pad_spin.setValue(CORP_STATE.padding)
        self.pad_spin.valueChanged.connect(CORP_STATE.setPadding)
        layout.addWidget(self.pad_spin)
        
        # Code Editor
        layout.addWidget(QLabel("Processing Script (Python):"))
        self.code_edit = QPlainTextEdit()
        self.code_edit.setPlainText(CORP_STATE.user_code)
        layout.addWidget(self.code_edit)
        
        # Run Button
        self.run_btn = QPushButton("🚀 Run Process")
        self.run_btn.clicked.connect(self._on_run)
        layout.addWidget(self.run_btn)
        
    def _on_run(self):
        CORP_STATE.user_code = self.code_edit.toPlainText()
        CORP_STATE.run_process()

    def onDataUpdate(self):
        if CORP_STATE.roi:
            r = CORP_STATE.roi
            self.roi_info.setText(f"X:{r.x:.1f} Y:{r.y:.1f} W:{r.w:.1f} H:{r.h:.1f}")

class CorpProcessMainWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая часть - управление
        self.controls = ControlPanel()
        splitter.addWidget(self.controls)
        
        # Правая часть - визуализация (вертикальный сплиттер)
        viz_splitter = QSplitter(Qt.Vertical)
        self.crop_view = CropVisualizer()
        self.res_view = ResultVisualizer()
        
        viz_splitter.addWidget(QLabel("1. Raw Crop with Padding"))
        viz_splitter.addWidget(self.crop_view)
        viz_splitter.addWidget(QLabel("2. Processed Result (Curve)"))
        viz_splitter.addWidget(self.res_view)
        
        splitter.addWidget(viz_splitter)
        layout.addWidget(splitter)