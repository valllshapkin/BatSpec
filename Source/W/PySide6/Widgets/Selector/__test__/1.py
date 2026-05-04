import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from W.PySide6.Widgets.Selector import Selector
from W.PySide6.Data.Reactive import ReactiveDict

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QWidget()
    win.resize(400, 200)
    layout = QVBoxLayout(win)

    db = ReactiveDict({"Key 1": "Hello", "Key 2": "World"})
    selector = Selector("Test Selector")
    selector.set_dictionary(db)
    layout.addWidget(selector)

    lbl = QLabel("Value: ")
    layout.addWidget(lbl)

    selector.signals.activeValueChanged.connect(lambda v: lbl.setText(f"Value: {v}"))
    
    btn = QPushButton("Add 'Key 3'")
    btn.clicked.connect(lambda: db.update({"Key 3": "PySide6 Magic!"}))
    layout.addWidget(btn)

    win.show()
    sys.exit(app.exec())
