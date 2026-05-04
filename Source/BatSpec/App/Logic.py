import sys
import contextlib
from PySide6 import QtWidgets, QtCore

from qt_pg_theme import setup_theme_integration; setup_theme_integration()
import pyqtgraph as pg
pg.setConfigOptions(useOpenGL=True, antialias=False)

from BatSpec.App.中Menu.中ThemeSettings.Logic import AppThemes
from BatSpec.App.中Menu.中LocalesSettings.Logic import AppLocales, SUPPORTED_LANGUAGES

# --- ИСПРАВЛЕНИЕ: Этот импорт должен быть здесь, чтобы запустить код инициализации
# в Logic.py модуля аннотаций ДО того, как любые виджеты попытаются импортировать модели.
import BatSpec.App.中Annotation.Logic

@contextlib.contextmanager
def application():
    if app := QtWidgets.QApplication.instance():
        yield app
        return
        
    QtCore.QCoreApplication.setOrganizationName("BatSpec")
    QtCore.QCoreApplication.setOrganizationDomain("batspec.com")
    QtCore.QCoreApplication.setApplicationName("BatSpec")
    app = QtWidgets.QApplication(sys.argv)

    # Инициализация сервисов
    AppThemes.init(debug=False) 
    AppLocales.init(supported_languages=SUPPORTED_LANGUAGES, jit_compile=False, debug=False)

    yield app
    sys.exit(app.exec())
