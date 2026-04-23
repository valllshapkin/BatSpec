from PySide6.QtWidgets import QMainWindow, QMenu
from PySide6.QtGui import QAction

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component


# --- 2. Отдельный класс для меню "Режим" (2 пункта) ---
class ModeMenu(*Component, QMenu):
    def __init__(self, parent=None):
        QMenu.__init__(self, parent)

        # Действие 1: Одиночный
        self.act_single = QAction(self)
        self.act_single.setCheckable(True) # Делаем пункт отмечаемым (с галочкой)
        self.act_single.setChecked(True)   # По умолчанию включен этот
        self.act_single.triggered.connect(self.set_single_mode)

        # Действие 2: Пакетный
        self.act_batch = QAction(self)
        self.act_batch.setCheckable(True)
        self.act_batch.triggered.connect(self.set_batch_mode)

        self.addAction(self.act_single)
        self.addAction(self.act_batch)

        self.onLanguageChange()
        self.onThemeChange()

    def set_single_mode(self):
        self.act_single.setChecked(True)
        self.act_batch.setChecked(False)
        print("Режим: Одиночный") # TODO: Заглушка логики

    def set_batch_mode(self):
        self.act_single.setChecked(False)
        self.act_batch.setChecked(True)
        print("Режим: Пакетный")  # TODO: Заглушка логики

    def onLanguageChange(self):
        self.setTitle(self.tr("Mode"))
        self.act_single.setText(self.tr("Single"))
        self.act_batch.setText(self.tr("Batch"))

    def onThemeChange(self):
        pass