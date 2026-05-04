import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from W.PySide6.App.Themes import Themes, System

if __name__ == "__main__":
    app = QApplication(sys.argv)
    t = Themes(Path("test_themes.ini"))
    print(f"Is system dark? {System.isDark()}")
    sys.exit()
