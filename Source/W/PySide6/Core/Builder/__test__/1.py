import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from W.PySide6.Core.Builder import build_node as b

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QWidget()
    
    with b(win, QVBoxLayout()) as layout:
        with b(layout, QPushButton("Button 1")) as btn1:
            btn1.setStyleSheet("color: red;")
        with b(layout, QPushButton("Button 2")) as btn2:
            pass

    win.show()
    sys.exit(app.exec())
