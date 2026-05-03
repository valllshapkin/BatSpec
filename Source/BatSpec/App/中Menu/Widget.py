from PySide6.QtWidgets import QMainWindow
from BatSpec.App.中Menu.中App.Widget import AppMenu
from W.PySide6.QtLocales import Locales
from W.PySide6.QtSсheme import ComponentLifecycle

class AddMenu(Locales.TranslateComponent, ComponentLifecycle, QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __init_graph__(self):
        if hasattr(super(), '__init_graph__'):
            super().__init_graph__()
            
        self.app_menu = AppMenu(self)
        self.menuBar().addMenu(self.app_menu)
