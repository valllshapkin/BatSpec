from pathlib import Path
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PySide6.QtCore import Qt

from W.PySide6.QtLocales import Locales
from W.PySide6.QtSсheme import ComponentLifecycle
from W.PySide6.QtBuilder import build_node as b
from W.PySide6.QtFrameless import FramelessMixin

ScriptDir = Path(__file__).parent
ICON_PATH = ScriptDir.parent.parent.parent.parent / "__assets__" / "icon.png"

class HelpDialog(Locales.TranslateComponent, ComponentLifecycle, FramelessMixin, QDialog):
    def __init_state__(self):
        self.setModal(True)
        self.resize(450, 300)
        # Инициализируем кастомную шапку для диалога!
        self.init_frameless(icon_path=ICON_PATH, fallback_icon="ℹ️")
        
    def __init_graph__(self):
        # Получаем слой контента внутри кастомного окна
        content_layout = self.build_frameless_ui()
        
        with b(content_layout, QVBoxLayout()) as self.layout:
            self.layout.setSpacing(10)
            self.layout.setContentsMargins(16, 16, 16, 16)

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
                self.layout.setAlignment(self.btn_close, Qt.AlignmentFlag.AlignRight)

    def __init_signal__(self):
        self.btn_close.clicked.connect(self.accept)

    def setWindowTitle(self, title: str):
        super().setWindowTitle(title)
        if hasattr(self, 'title_bar'):
            self.title_bar.update_title(title)

    def onThemeChange(self):
        self.update_frameless_theme()
        super().onThemeChange()

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("Help & About"))
        self.lbl_app_name.setText("BatSpec QtApp")
        self.lbl_version.setText(self.tr("Version: %1").replace("%1", "1.2.5 (Stable)"))
        self.lbl_publisher.setText(self.tr("Publisher: %1").replace("%1", "BatSpec Industries"))
        self.lbl_description.setText(self.tr("This application is designed for professional theme and localization management."))
        self.lbl_copyright.setText(self.tr("© 2024-2025 BatSpec. All rights reserved."))
        self.btn_close.setText(self.tr("Close"))
        super().onLanguageChange()
