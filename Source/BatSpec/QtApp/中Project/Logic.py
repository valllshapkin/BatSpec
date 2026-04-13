from PySide6.QtCore import QObject, Signal
from BatSpec.QtApp.Shared.Component import ComponentSettings
from BatSpec.QtUp.Settings import SettingField
from pathlib import Path
import logging

ScriptDir = Path(__file__).parent

class Project:
    def __init__(self, path: Path):
        self.folder_root = path
        self.folder_storage = path / "storage"
        self.folder_windows = self.folder_storage / "windows"
        self.folder_records = self.folder_storage / "records"
        self.folder_settings = self.folder_root / "settings"

    def is_valid(self):
        return self.folder_root.is_dir()

    def mkdirs(self):
        self.folder_root.mkdir(parents=True, exist_ok=True)
        self.folder_storage.mkdir(parents=True, exist_ok=True)
        self.folder_windows.mkdir(parents=True, exist_ok=True)
        self.folder_records.mkdir(parents=True, exist_ok=True)
        self.folder_settings.mkdir(parents=True, exist_ok=True)

        
class ProjectState:

    class Signals(QObject):
        projectNewLoaded = Signal()
        windowStorageChanged = Signal()

    class ProjectSettings(ComponentSettings):
        COMPONENT_DIR = ScriptDir
        open_project = SettingField[str](default="", type=str, key="Project")

    class Trigger:
        def __init_subclass__(cls, *args, **kwargs):
            
            old_init = cls.__init__

            def new_init(self, *args, **kwargs):
                old_init(self, *args, **kwargs)

                PROJECT_STATE.signals.projectNewLoaded.connect(self.onProjectNewLoaded)
                PROJECT_STATE.signals.windowStorageChanged.connect(self.onWindowStorageChanged)

                self.onProjectNewLoaded()
                self.onWindowStorageChanged()

            cls.__init__ = new_init

            super().__init_subclass__(*args, **kwargs)

        def onProjectNewLoaded(self):
            pass

        def onWindowStorageChanged(self):
            pass

    currentProject: Project | None = None

    def __init__(self) -> None:
        self.signals = self.Signals()
        self.settings = self.ProjectSettings()
        if self.settings.open_project:
            self.loadProject(Path(self.settings.open_project))

    def isUploaded(self) -> bool:
        return self.currentProject is not None
    
    def loadProject(self, path: Path | None = None):

        if path is None and self.currentProject is None:
            return
            
        if path is None:
            self.currentProject = None
            self.signals.projectNewLoaded.emit()
            self.settings.open_project = ""
            return

        project = Project(path)

        if not project.is_valid(): 
            raise Exception("Invalid project")
        
        if self.currentProject and self.currentProject.folder_root == project.folder_root:
            return

        self.currentProject = project
        self.signals.projectNewLoaded.emit()
        self.settings.open_project = str(path)


PROJECT_STATE = ProjectState()

@PROJECT_STATE.signals.projectNewLoaded.connect
def debug(): logging.info(f"Project loaded:\t{PROJECT_STATE.currentProject.folder_root if PROJECT_STATE.currentProject is not None else 'None'}")

