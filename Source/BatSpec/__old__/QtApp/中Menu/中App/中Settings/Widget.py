
from PySide6.QtWidgets import (
    QPushButton, QTabWidget, 
    QVBoxLayout, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component

from BatSpec.QtApp.中Menu.中App.中Settings.中LocalesTab.Widget import LocalesSettingsTab
from BatSpec.QtApp.中Menu.中App.中Settings.中ThemesTab.Widget import ThemeSettingsTab


class SettingsDialog(*Component, QDialog):

    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        self.setModal(True)
        
        self.l = QVBoxLayout(self)
        self.lT = QTabWidget()
        self.lT.setUsesScrollButtons(True)
        self.l.addWidget(self.lT)
        
        self.lT1 = ThemeSettingsTab(tab_widget=self.lT)
        self.lT2 = LocalesSettingsTab(tab_widget=self.lT)

        # self.lB = QDialogButtonBox()

        # self.lB1 = QPushButton()
        # self.lB2 = QPushButton()

        # self.lB.addButton(self.lB1, QDialogButtonBox.ButtonRole.AcceptRole)
        # self.lB.addButton(self.lB2, QDialogButtonBox.ButtonRole.RejectRole)

        # self.lB.accepted.connect(self.accept)
        # self.lB.rejected.connect(self.reject)

        # self.l.addWidget(self.lB, alignment=Qt.AlignmentFlag.AlignBottom)

    def onLanguageChange(self):
        self.setWindowTitle(self.tr("Settings"))
        
        # self.lB1.setText(self.tr("OK"))
        # self.lB2.setText(self.tr("Cancel"))


    def onThemeChange(self):
        pass