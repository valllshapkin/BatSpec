from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont


from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component

class HelpDialog(*Component, QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setModal(True)
        self.setFixedSize(400, 250) # Фиксированный размер для аккуратного вида
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)

        # --- Верхняя часть: Иконка и Название ---
        header_layout = QHBoxLayout()
        
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(64, 64)
        # self.icon_label.setPixmap(QPixmap("path/to/logo.png").scaled(64, 64, Qt.KeepAspectRatio))
        # Заглушка вместо иконки
        self.icon_label.setStyleSheet("background-color: #3498DB; border-radius: 10px;") 
        
        title_v_box = QVBoxLayout()
        self.lbl_app_name = QLabel()
        self.lbl_app_name.setStyleSheet("font-size: 18pt; font-weight: bold;")
        
        self.lbl_version = QLabel()
        self.lbl_version.setStyleSheet("color: gray;")
        
        title_v_box.addWidget(self.lbl_app_name)
        title_v_box.addWidget(self.lbl_version)
        title_v_box.addStretch()
        
        header_layout.addWidget(self.icon_label)
        header_layout.addLayout(title_v_box)
        header_layout.addStretch()
        self.layout.addLayout(header_layout)

        # Разделительная линия
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(line)

        # --- Информационный блок ---
        self.lbl_publisher = QLabel()
        self.lbl_description = QLabel()
        self.lbl_description.setWordWrap(True)
        self.lbl_copyright = QLabel()
        self.lbl_copyright.setStyleSheet("font-size: 9pt; color: gray;")

        self.layout.addWidget(self.lbl_publisher)
        self.layout.addWidget(self.lbl_description)
        self.layout.addStretch()
        self.layout.addWidget(self.lbl_copyright)

        # --- Кнопка закрытия ---
        self.btn_close = QPushButton()
        self.btn_close.setFixedWidth(100)
        self.btn_close.clicked.connect(self.accept)
        
        self.layout.addWidget(self.btn_close, alignment=Qt.AlignmentFlag.AlignRight)

        # Инициализация текстов
        self.onLanguageChange()
        self.onThemeChange()

    def onLanguageChange(self):
        # Перевод заголовка и статических данных
        self.setWindowTitle(self.tr("Help & About"))
        
        self.lbl_app_name.setText("BatSpec QtApp")
        self.lbl_version.setText(self.tr("Version: %1").replace("%1", "1.2.5 (Stable)"))
        
        self.lbl_publisher.setText(self.tr("Publisher: %1").replace("%1", "BatSpec Industries"))
        
        self.lbl_description.setText(
            self.tr("This application is designed for professional theme and localization management.")
        )
        
        self.lbl_copyright.setText(self.tr("© 2024-2025 BatSpec. All rights reserved."))
        self.btn_close.setText(self.tr("Close"))
        
        return super().onLanguageChange()

    def onThemeChange(self):
        # Здесь можно менять стили динамически при смене темы
        return super().onThemeChange()