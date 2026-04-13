from typing import Self
from PySide6.QtWidgets import QWidget, QTabWidget

class TabInstance(QWidget):
    tab_name: str | None = None
    tab_widget: QTabWidget

    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.tab_widget = tab_widget
        self.setContentsMargins(0, 0, 0, 0)
        self.tab_widget.addTab(self, self.tab_name) if tab_widget else None

    @property
    def tabIndex(self: Self) -> int:
        return self.tab_widget.indexOf(self)

    def setTabName(self: Self, name: str | None) -> Self:
        self.tab_name = name
        self.tab_widget.setTabText(self.tabIndex, name) if self.tab_widget else None
        return self
    
    def setTabToolTip(self: Self, tooltip: str | None) -> Self:
        self.tab_widget.setTabToolTip(self.tabIndex, tooltip) if self.tab_widget else None
        return self