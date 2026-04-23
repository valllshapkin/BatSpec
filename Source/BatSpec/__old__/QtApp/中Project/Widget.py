import os
import scipy.signal as sig
import pyqtgraph as pg
from pathlib import Path
from PySide6.QtWidgets import (QFileDialog, QFileSystemModel, QTreeView, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                               QSpinBox, QPushButton, QFormLayout, QSplitter, 
                               QTextBrowser, QTabWidget, QLabel, QStackedWidget)
from PySide6.QtCore import QDir, Qt

from BatSpec.QtUp.TabInst import TabInstance

from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtApp.Shared.Component import Component, GLOBAL_MARGINS

from .Logic import Project, PROJECT_STATE

class ProjectTab(*Component, PROJECT_STATE.Trigger, TabInstance):

    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)
        
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(*GLOBAL_MARGINS)


        # Метка с путём текущего проекта (оставляем как есть)
        self.vL = QLabel()
        self.v.addWidget(self.vL)

        # Кнопки загрузки/закрытия проекта
        self.button_select_project = QPushButton()
        self.button_select_project.clicked.connect(self.select_project)
        self.v.addWidget(self.button_select_project)

        self.button_close_project = QPushButton()
        self.button_close_project.clicked.connect(lambda: PROJECT_STATE.loadProject(None))
        self.v.addWidget(self.button_close_project)

        # Стек для переключения между сообщением и деревом
        self.stacked_widget = QStackedWidget()

        # Виджет-заглушка для случая без проекта
        self.message_label = QLabel()
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stacked_widget.addWidget(self.message_label)  # индекс 0

        # Дерево файлов для отображения содержимого проекта
        self.tree_view = QTreeView()
        self.model = QFileSystemModel()
        self.model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        self.tree_view.setModel(self.model)
        self.tree_view.setColumnHidden(1, True)  # размер
        self.tree_view.setColumnHidden(2, True)  # тип
        self.tree_view.setColumnHidden(3, True)  # дата изменения
        self.stacked_widget.addWidget(self.tree_view)  # индекс 1

        # Подключаем сигнал изменения проекта для обновления стека и дерева
        self.view_model_callback()  # установим начальное состояние

        self.v.addWidget(self.stacked_widget)

    def onProjectNewLoaded(self):
        self.vL_callback()
        self.view_model_callback()

    def vL_callback(self):
        if PROJECT_STATE.currentProject is None:
            self.vL.setText("None")
        else:
            self.vL.setText(str(PROJECT_STATE.currentProject.folder_root))

    def view_model_callback(self):
        """Обновляет стек и дерево при изменении проекта."""
        if PROJECT_STATE.currentProject is None:
            # Проект не загружен – показываем сообщение
            self.stacked_widget.setCurrentIndex(0)
        else:
            # Проект загружен – показываем дерево и устанавливаем корневой путь
            self.stacked_widget.setCurrentIndex(1)
            path = str(PROJECT_STATE.currentProject.folder_root)
            if path and QDir(path).exists():
                PROJECT_STATE.currentProject.mkdirs()
                index = self.model.setRootPath(path)
                self.tree_view.setRootIndex(index)

    def select_project(self):
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Select Folder")
        )
        if not folder:
            return

        folder = Path(folder)
        PROJECT_STATE.loadProject(folder)

    def onLanguageChange(self):
        self.button_select_project.setText(self.tr("Load Project"))
        self.button_close_project.setText(self.tr("Close Project"))
        self.message_label.setText(self.tr("No project loaded"))
        self.setTabName(self.tr("Project"))

    def onThemeChange(self):
        pass