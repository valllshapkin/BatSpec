from PySide6.QtWidgets import QMainWindow
from BatSpec.App.中Menu.中App.Widget import AppMenu
# from BatSpec.App.中Menu.中Mode.Widget import ModeMenu

from BatSpec.QtUp.Locales import Locales
from BatSpec.QtUp.Themes import Themes

class AddMenu(Locales.TranslateComponent, Locales.Trigger, Themes.Trigger, QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # Инициализируем наши модульные меню
        self.app_menu = AppMenu(self)
        # self.mode_menu = ModeMenu(self)

        # Добавляем их в верхнюю панель (MenuBar) главного окна
        self.menuBar().addMenu(self.app_menu)
        # self.menuBar().addMenu(self.mode_menu)

    def onLanguageChange(self):
        super().onLanguageChange()

    def onThemeChange(self):
        super().onThemeChange()