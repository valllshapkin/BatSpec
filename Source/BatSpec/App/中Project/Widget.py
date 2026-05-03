from pathlib import Path
from PySide6.QtWidgets import (
    QFileDialog, QFileSystemModel, QTreeView, QWidget, QVBoxLayout, 
    QPushButton, QStackedWidget, QTabWidget, QLabel
)
from PySide6.QtCore import QDir, Qt

from BatSpec.QtUp.TabInst import TabInstance
from BatSpec.QtUp.Locales import Locales
from BatSpec.QtUp.Themes import Themes
from BatSpec.QtUp.Builder import build_node as b

from .Logic import PROJECT_STATE

class ProjectTab(Locales.TranslateComponent, Locales.Trigger, Themes.Trigger, PROJECT_STATE.Trigger, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent, tab_widget)
        
        with b(self, QVBoxLayout()) as self.v:
            self.v.setContentsMargins(4, 0, 4, 0)

            with b(self.v, QLabel()) as self.vL: pass

            with b(self.v, QPushButton()) as self.button_select_project:
                self.button_select_project.clicked.connect(self.select_project)

            with b(self.v, QPushButton()) as self.button_close_project:
                self.button_close_project.clicked.connect(lambda: PROJECT_STATE.loadProject(None))

            with b(self.v, QStackedWidget()) as self.stacked_widget:
                with b(self.stacked_widget, QLabel()) as self.message_label:
                    self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                with b(self.stacked_widget, QTreeView()) as self.tree_view:
                    self.model = QFileSystemModel()
                    self.model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
                    self.tree_view.setModel(self.model)
                    self.tree_view.setColumnHidden(1, True)
                    self.tree_view.setColumnHidden(2, True)
                    self.tree_view.setColumnHidden(3, True)

        self.view_model_callback()

    def onProjectNewLoaded(self):
        self.vL_callback()
        self.view_model_callback()

    def vL_callback(self):
        if PROJECT_STATE.currentProject is None:
            self.vL.setText("None")
        else:
            self.vL.setText(str(PROJECT_STATE.currentProject.folder_root))

    def view_model_callback(self):
        if PROJECT_STATE.currentProject is None:
            self.stacked_widget.setCurrentIndex(0)
        else:
            self.stacked_widget.setCurrentIndex(1)
            path = str(PROJECT_STATE.currentProject.folder_root)
            if path and QDir(path).exists():
                PROJECT_STATE.currentProject.mkdirs()
                index = self.model.setRootPath(path)
                self.tree_view.setRootIndex(index)

    def select_project(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr("Select Folder"))
        if not folder: return
        
        folder = Path(folder)
        PROJECT_STATE.loadProject(folder)

    def onLanguageChange(self):
        self.button_select_project.setText(self.tr("Load Project"))
        self.button_close_project.setText(self.tr("Close Project"))
        self.message_label.setText(self.tr("No project loaded"))
        if self.tab_widget:
            self.setTabName(self.tr("Project"))
        super().onLanguageChange()

    def onThemeChange(self):
        super().onThemeChange()
