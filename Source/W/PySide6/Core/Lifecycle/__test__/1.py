import sys
from PySide6.QtWidgets import QApplication, QLabel
from W.PySide6.Core.Lifecycle import ComponentLifecycle

class MyWidget(ComponentLifecycle, QLabel):
    def __init_state__(self):
        self.click_count = 0

    def __init_graph__(self):
        self.setText(f"Lifecycle Initialized! Clicks: {self.click_count}")

    def mousePressEvent(self, event):
        self.click_count += 1
        self.setText(f"Lifecycle Initialized! Clicks: {self.click_count}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyWidget()
    w.resize(300, 100)
    w.show()
    sys.exit(app.exec())
