import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
from W.PySide6.Widgets.Frameless import FramelessMixin

class MyWin(FramelessMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_frameless()
        layout = self.build_frameless_ui()
        layout.addWidget(QLabel("Frameless Window Magic!"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyWin()
    w.resize(400, 300)
    w.show()
    sys.exit(app.exec())
