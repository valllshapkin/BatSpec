from PySide6.QtWidgets import QMainWindow, QMenu
from PySide6.QtGui import QAction

# Предполагаемые импорты ваших диалогов
from BatSpec.QtApp.中Menu.中App.中Settings.Widget import SettingsDialog
from BatSpec.QtApp.中Menu.中App.中UserData.Widget import UserDataDialog
from BatSpec.QtApp.中Menu.中App.中Help.Widget import HelpDialog


from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component


# --- 1. Отдельный класс для меню "Приложение" (3 пункта) ---
class AppMenu(*Component, QMenu):
    def __init__(self, parent=None):
        QMenu.__init__(self, parent)
        
        # Инициализация диалогов. Передаём parent (главное окно), 
        # чтобы диалоги появлялись ровно по центру программы
        self.settings_dialog = SettingsDialog(parent)
        self.user_data_dialog = UserDataDialog(parent)
        self.help_dialog = HelpDialog(parent)

        # Действие 1: Настройки
        self.act_settings = QAction(self)
        self.act_settings.triggered.connect(self.settings_dialog.exec)

        # Действие 2: Данные пользователя
        self.act_user_data = QAction(self)
        self.act_user_data.triggered.connect(self.user_data_dialog.exec)

        # Действие 3: Помощь
        self.act_help = QAction(self)
        self.act_help.triggered.connect(self.help_dialog.exec)

        # Добавляем пункты в меню
        self.addAction(self.act_settings)
        self.addAction(self.act_user_data)
        self.addSeparator() # Разделитель перед "Помощью" (Опционально, для красоты)
        self.addAction(self.act_help)

        self.onLanguageChange()
        self.onThemeChange()

    def onLanguageChange(self):
        self.setTitle(self.tr("App"))
        self.act_settings.setText(self.tr("Settings"))
        self.act_user_data.setText(self.tr("User Data"))
        self.act_help.setText(self.tr("Help"))

    def onThemeChange(self):
        pass