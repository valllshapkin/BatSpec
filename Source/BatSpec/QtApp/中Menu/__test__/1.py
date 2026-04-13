from BatSpec.QtApp.中Menu.Widget import AddMenu

import sys
from PySide6.QtWidgets import QApplication
    
app = QApplication(sys.argv)

window = AddMenu()
window.setWindowTitle(AddMenu.__qualname__)
window.resize(350, 150)

from BatSpec.QtApp.Services.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Services.中LocalesSettings.Logic import Locales
Themes.initThemes() 
Locales.initLocales()
window.show()
sys.exit(app.exec())