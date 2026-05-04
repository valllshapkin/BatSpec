import sys
from PySide6.QtWidgets import QApplication, QTabWidget
from W.PySide6.Widgets.TabInstance import TabInstance

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tabs = QTabWidget()
    
    class MyTab(TabInstance):
        tab_name = "Dynamic Tab"

    t = MyTab(tab_widget=tabs)
    tabs.show()
    sys.exit(app.exec())
