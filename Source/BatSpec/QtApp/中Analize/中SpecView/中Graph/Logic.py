from typing import Optional, Self

from BatSpec.Logic.Functions import SpecFunc, TimeFunc
from BatSpec.QtApp.Shared.Component import ComponentSettings
from BatSpec.QtUp.Settings import SettingField
from BatSpec.QtUp.PolyROI import ROIGroup, ROIData
from PySide6.QtCore import QObject, Signal
import pyqtgraph as pg
from pathlib import Path
ScriptDir = Path(__file__).parent

class GraphState:
    
    class Settings(ComponentSettings):
        COMPONENT_DIR = ScriptDir
        colormap = SettingField[str](default="viridis", type=str, key="Analize/SpecView/Graph/colormap")
        locking = SettingField[bool](default=True, type=bool, key="Analize/SpecView/Graph/locking")
    
    class Signals(QObject):
        sigMouseHover = Signal(float, float, float)  # t, f, val
        sigMouseOut = Signal()
        sigChangeColorMap = Signal()
        sigChangeLockY = Signal()
        sigChangeSpecFunc = Signal()
        sigChangeTimeFunc = Signal()

    class TriggerGraph:
        def __init_subclass__(cls, *args, **kwargs):
            
            old_init = cls.__init__

            def new_init(self: Self, *args, **kwargs):
                old_init(self, *args, **kwargs)

                GRAPH_STATE.signals.sigChangeColorMap.connect(self.onChangeColorMap)
                GRAPH_STATE.signals.sigChangeLockY.connect(self.onChangeLockY)
                GRAPH_STATE.signals.sigChangeSpecFunc.connect(self.onChangeSpecFunc)

                self.onChangeColorMap()
                self.onChangeLockY()
                self.onChangeSpecFunc()

            cls.__init__ = new_init

            super().__init_subclass__(*args, **kwargs)

        def onChangeColorMap(self): pass
        def onChangeLockY(self): pass
        def onChangeSpecFunc(self): pass

    class TriggerMinimap:
        def __init_subclass__(cls, *args, **kwargs):
            
            old_init = cls.__init__

            def new_init(self: Self, *args, **kwargs):
                old_init(self, *args, **kwargs)

                GRAPH_STATE.signals.sigChangeTimeFunc.connect(self.onChangeTimeFunc)
                self.onChangeTimeFunc()

            cls.__init__ = new_init

            super().__init_subclass__(*args, **kwargs)

        def onChangeTimeFunc(self):
            pass 

    def __init__(self) -> None:
        self.signals = self.Signals()
        self.settings = self.Settings()
        self.time_func = None
        self.spec_func = None

    @property
    def colormap(self):
        return pg.colormap.get(self.settings.colormap)

    @property
    def lock_y(self):
        return self.settings.locking

    def setMinimapFunc(self, func: TimeFunc | None):
        self.time_func = func
        self.signals.sigChangeTimeFunc.emit()

    def setSpecFunc(self, func: SpecFunc | None):
        self.spec_func = func
        self.signals.sigChangeSpecFunc.emit()

    def setColorMap(self, name: str | None):
        if not name or not name in pg.colormap.listMaps(): name = "viridis"
        self.settings.colormap = name
        self.signals.sigChangeColorMap.emit()

    def setLockY(self, val: bool):
        self.settings.locking = val
        self.signals.sigChangeLockY.emit()

GRAPH_STATE = GraphState()




