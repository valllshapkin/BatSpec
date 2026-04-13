import sys
from PySide6.QtWidgets import QApplication

from BatSpec.QtApp.Services.中LocalesSettings.Widget import LocalesSettingsPanel

app = QApplication(sys.argv)

window = LocalesSettingsPanel()
window.setWindowTitle("Тест панели настроек")
window.resize(350, 150)


from BatSpec.QtApp.Services.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Services.中LocalesSettings.Logic import Locales
Themes.initThemes() 
Locales.initLocales()
window.show()
sys.exit(app.exec())