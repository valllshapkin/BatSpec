from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction

from BatSpec.App.中Menu.中App.中Settings.Widget import SettingsDialog
from BatSpec.App.中Menu.中App.中UserData.Widget import UserDataDialog
from BatSpec.App.中Menu.中App.中Help.Widget import HelpDialog

from W.PySide6.QtLocales import Locales
from W.PySide6.QtSсheme import ComponentLifecycle

class AppMenu(Locales.TranslateComponent, ComponentLifecycle, QMenu):
    def __init__(self, parent=None):
        super().__init__(parent)

    def __init_graph__(self):
        self.settings_dialog = SettingsDialog(self.parent())
        self.user_data_dialog = UserDataDialog(self.parent())
        self.help_dialog = HelpDialog(self.parent())

        self.act_settings = QAction(self)
        self.act_user_data = QAction(self)
        self.act_help = QAction(self)

        self.addAction(self.act_settings)
        self.addAction(self.act_user_data)
        self.addSeparator()
        self.addAction(self.act_help)

    def __init_signal__(self):
        self.act_settings.triggered.connect(self.settings_dialog.exec)
        self.act_user_data.triggered.connect(self.user_data_dialog.exec)
        self.act_help.triggered.connect(self.help_dialog.exec)

    def onLanguageChange(self):
        self.setTitle(self.tr("App"))
        self.act_settings.setText(self.tr("Settings"))
        self.act_user_data.setText(self.tr("User Data"))
        self.act_help.setText(self.tr("Help"))
        super().onLanguageChange()
