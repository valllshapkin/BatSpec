from pathlib import Path
from PySide6.QtWidgets import (
    QFileDialog, QFileSystemModel, QTreeView, QWidget, QVBoxLayout, 
    QPushButton, QStackedWidget, QTabWidget, QLabel
)
from PySide6.QtCore import QDir, Qt

from W.PySide6.App.Locales import Locales
from W.PySide6.Core.Lifecycle import ComponentLifecycle
from W.PySide6.Widgets.TabInstance import TabInstance
from W.PySide6.Core.Builder import build_node as b

from .Logic import PROJECT_STATE

class ProjectTab(Locales.TranslateComponent, ComponentLifecycle, PROJECT_STATE.Trigger, TabInstance):
    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        super().__init__(parent, tab_widget)
        
    def __init_graph__(self):
        with b(self, QVBoxLayout()) as self.v:
            self.v.setContentsMargins(4, 0, 4, 0)

            with b(self.v, QLabel()) as self.vL: pass

            with b(self.v, QPushButton()) as self.button_select_project: pass
            with b(self.v, QPushButton()) as self.button_close_project: pass

            with b(self.v, QStackedWidget()) as self.stacked_widget:
                with b(self.stacked_widget, QLabel()) as self.message_label:
                    self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                with b(self.stacked_widget, QTreeView()) as self.tree_view:
                    self.model = QFileSystemModel()
                    self.model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
                    self.tree_view.setModel(self.model)
                    for i in range(1, 4):
                        self.tree_view.setColumnHidden(i, True)

    def __init_signal__(self):
        self.button_select_project.clicked.connect(self.select_project)
        self.button_close_project.clicked.connect(lambda: PROJECT_STATE.loadProject(None))

    def __init_ready__(self):
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
        if not hasattr(self, 'stacked_widget'): return
        
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
        PROJECT_STATE.loadProject(Path(folder))

    def onLanguageChange(self):
        self.button_select_project.setText(self.tr("Load Project"))
        self.button_close_project.setText(self.tr("Close Project"))
        self.message_label.setText(self.tr("No project loaded"))
        if self.tab_widget:
            self.setTabName(self.tr("Project"))
        super().onLanguageChange()
