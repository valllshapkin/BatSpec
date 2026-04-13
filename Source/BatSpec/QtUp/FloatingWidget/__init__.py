import sys
from PySide6.QtWidgets import QGroupBox, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QApplication
from PySide6.QtCore import Qt

class FloatingPanel(QWidget):

    title = "FloatingWidget"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)

        
        self.l = QVBoxLayout(self)
        self.l.setContentsMargins(0, 0, 0, 0)

        self.lG = QGroupBox()
        self.l.addWidget(self.lG)

        self.lGv = QVBoxLayout(self.lG)
        self.lG.setLayout(self.lGv)

        self.lGvh = QHBoxLayout()
        self.lGv.addLayout(self.lGvh)

        self.lGvhL = QLabel(self.title)
        self.lGvh.addWidget(self.lGvhL)

        self.lGvhS = self.lGvh.addStretch()

        self.lGvhB = QPushButton("X")
        self.lGvhB.clicked.connect(self.close)
        self.lGvh.addWidget(self.lGvhB)

        self.lGv.addWidget(self.initMainWidget(), alignment=Qt.AlignmentFlag.AlignCenter)

    def initMainWidget(self) -> QWidget:
        return QLabel("Not Implemented")

