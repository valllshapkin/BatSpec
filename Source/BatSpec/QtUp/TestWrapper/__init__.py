import contextlib
import sys

from qt_themes import set_theme
from PySide6 import QtWidgets


@contextlib.contextmanager
def application() -> QtWidgets.QApplication:
    if app := QtWidgets.QApplication.instance():
        yield app
        return

    app = QtWidgets.QApplication(sys.argv)
    set_theme("catppuccin_mocha")
    yield app
    app.exec()
