import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from W.PySide6.App.Locales import Locales

if __name__ == "__main__":
    app = QApplication(sys.argv)
    l = Locales(Path("test_locales.ini"))
    print("Locales System Created")
    sys.exit()
