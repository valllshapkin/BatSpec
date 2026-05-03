from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame
)
from PySide6.QtCore import Qt

from BatSpec.QtUp.Locales import Locales
from BatSpec.QtUp.Themes import Themes
from BatSpec.QtUp.Builder import build_node as b

class HelpDialog(Locales.TranslateComponent, Locales.Trigger, Themes.Trigger, QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setModal(True)
        self.setFixedSize(400, 250)
        
        with b(self, QVBoxLayout()) as self.layout:
            self.layout.setSpacing(10)

            with b(self.layout, QHBoxLayout()) as header_layout:
                with b(header_layout, QLabel()) as self.icon_label:
                    self.icon_label.setFixedSize(64, 64)
                    self.icon_label.setStyleSheet("background-color: #3498DB; border-radius: 10px;") 
                
                with b(header_layout, QVBoxLayout()) as title_v_box:
                    with b(title_v_box, QLabel()) as self.lbl_app_name:
                        self.lbl_app_name.setStyleSheet("font-size: 18pt; font-weight: bold;")
                    with b(title_v_box, QLabel()) as self.lbl_version:
                        self.lbl_version.setStyleSheet("color: gray;")
                    title_v_box.addStretch()
                
                header_layout.addStretch()

            with b(self.layout, QFrame()) as line:
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFrameShadow(QFrame.Shadow.Sunken)

            with b(self.layout, QLabel()) as self.lbl_publisher: pass
            
            with b(self.layout, QLabel()) as self.lbl_description:
                self.lbl_description.setWordWrap(True)
                
            self.layout.addStretch()
            
            with b(self.layout, QLabel()) as self.lbl_copyright:
                self.lbl_copyright.setStyleSheet("font-size: 9pt; color: gray;")

            with b(self.layout, QPushButton()) as self.btn_close:
                self.btn_close.setFixedWidth(100)
                self.btn_close.clicked.connect(self.accept)
                self.layout.setAlignment(self.btn_close, Qt.AlignmentFlag.AlignRight)

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("Help & About"))
        
        self.lbl_app_name.setText("BatSpec QtApp")
        self.lbl_version.setText(self.tr("Version: %1").replace("%1", "1.2.5 (Stable)"))
        
        self.lbl_publisher.setText(self.tr("Publisher: %1").replace("%1", "BatSpec Industries"))
        
        self.lbl_description.setText(
            self.tr("This application is designed for professional theme and localization management.")
        )
        
        self.lbl_copyright.setText(self.tr("© 2024-2025 BatSpec. All rights reserved."))
        self.btn_close.setText(self.tr("Close"))
        
        super().onLanguageChange()

    def onThemeChange(self):
        super().onThemeChange()
