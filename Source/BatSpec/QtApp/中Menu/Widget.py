from PySide6.QtWidgets import QMainWindow, QMenu
from PySide6.QtGui import QAction

# Предполагаемые импорты ваших диалогов
from BatSpec.QtApp.中Menu.中App.Widget import AppMenu
from BatSpec.QtApp.中Menu.中Mode.Widget import ModeMenu

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component


# --- 3. Главное окно, собирающее меню ---
class AddMenu(*Component, QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # Инициализируем наши модульные меню
        self.app_menu = AppMenu(self)
        self.mode_menu = ModeMenu(self)

        # Добавляем их в верхнюю панель (MenuBar) главного окна
        self.menuBar().addMenu(self.app_menu)
        self.menuBar().addMenu(self.mode_menu)

    def onLanguageChange(self):
        pass

    def onThemeChange(self):
        pass