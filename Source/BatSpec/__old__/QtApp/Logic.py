
import sys
from qt_pg_theme import setup_theme_integration; setup_theme_integration()
from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from PySide6 import QtCore, QtWidgets

import pyqtgraph as pg
pg.setConfigOptions(useOpenGL=True, antialias=False)

import contextlib
import sys

from PySide6 import QtWidgets

@contextlib.contextmanager
def application() -> QtWidgets.QApplication:
    if app := QtWidgets.QApplication.instance():
        yield app
        return
        
    QtCore.QCoreApplication.setOrganizationName("BatSpec")
    QtCore.QCoreApplication.setOrganizationDomain("batspec.com")
    QtCore.QCoreApplication.setApplicationName("BatSpec")
    app = QtWidgets.QApplication(sys.argv)

    Themes.initThemes() 
    Locales.initLocales()

    yield app
    sys.exit(app.exec())

