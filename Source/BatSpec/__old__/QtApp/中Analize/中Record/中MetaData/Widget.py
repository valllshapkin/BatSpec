from PySide6.QtWidgets import QTabWidget, QWidget
from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from .中Info.Widget import Info
from .中Guano.Widget import Guano


class MetaData(QTabWidget):
    def __init__(self, parent: QWidget | None = None):
        QTabWidget.__init__(self, parent)
        
        # Создаём вкладки
        self.info_tab = Info(self, self)
        self.guano_tab = Guano(self, self)
    
