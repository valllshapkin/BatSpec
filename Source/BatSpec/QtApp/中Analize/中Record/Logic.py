from numpy.typing import NDArray
import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, Signal
from pathlib import Path

ScriptDir = Path(__file__).parent

from BatSpec.Logic.Functions import TimeFunc
from BatSpec.Logic.Loader import loadRecord
from BatSpec.Logic.SoundWorker import correctDC

class Record:
    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def row(self) -> TimeFunc:
        return loadRecord(self.path)

    @property
    def corected(self) -> TimeFunc:
        return correctDC(self.row)
    
    @property
    def samplerate(self) -> int:
        return self.row.sr

    @property
    def duration(self):
        return self.row.duration

    @property
    def info(self):
        return sf.info(self.path)

from BatSpec.QtApp.Shared.Component import ComponentSettings
from BatSpec.QtUp.Settings import SettingField


class RecordState:
    path: Path | None = None
    record: Record | None = None

    class RecordSettings(ComponentSettings):
        COMPONENT_DIR = ScriptDir
        open_path = SettingField[str](default="", type=str, key="Analize/Record/open_path")

    class Signals(QObject):
        recordChanged = Signal()

    
    class Trigger:
        def __init_subclass__(cls, *args, **kwargs):
            
            old_init = cls.__init__

            def new_init(self, *args, **kwargs):
                old_init(self, *args, **kwargs)

                RECORD_STATE.signals.recordChanged.connect(self.onRecordChange)
                self.onRecordChange()

            cls.__init__ = new_init

            super().__init_subclass__(*args, **kwargs)

        def onRecordChange(self):
            pass

    def __init__(self) -> None:
        self.signals = self.Signals()
        self.settings = self.RecordSettings()
        if self.settings.open_path: 
            print("Restore", self.settings.open_path)
            try:
                self.setRecordFile(Path(self.settings.open_path))
            except:
                self.settings.open_path = ""
            
    def setRecordFile(self, path: Path | None):
        if not path:
            self.path = None
            self.record = None
            self.settings.open_path = ""
        else:
            if path and not path.exists(): raise FileNotFoundError(f"Файл не найден: {path}")
            self.path = path
            self.record = Record(path)
            self.settings.open_path = str(path)

        self.signals.recordChanged.emit()

RECORD_STATE = RecordState()