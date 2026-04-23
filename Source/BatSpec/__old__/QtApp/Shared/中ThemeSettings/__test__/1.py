import sys
from PySide6.QtWidgets import QApplication
from BatSpec.QtApp.Services.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Services.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Services.中ThemeSettings.Widget import ThemeSettingsPanel

app = QApplication(sys.argv)

window = ThemeSettingsPanel()
window.setWindowTitle("Тест панели настроек")
window.resize(350, 150)

Themes.initThemes() 
Locales.initLocales()
window.show()
sys.exit(app.exec())