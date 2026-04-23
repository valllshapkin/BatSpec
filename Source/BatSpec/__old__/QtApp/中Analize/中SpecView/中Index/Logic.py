from typing import Optional, Self

from BatSpec.QtUp.PolyROI import ROIGroup, ROIData
from PySide6.QtCore import QObject, Signal
from pathlib import Path
ScriptDir = Path(__file__).parent






class IndexState:
     
    class Signals(QObject):
        sigChangeSize = Signal()
        sigChangeToolMode = Signal()     # Смена инструмента (select, create, erase)
        sigChangeRoiGroups = Signal()    # Данные ROI изменились (добавили/удалили)
        sigChangeRoiSelection = Signal() # Изменился активный ROI
        sigChangeActiveGroup = Signal()

    class Trigger:
        def __init_subclass__(cls, *args, **kwargs):
            
            old_init = cls.__init__

            def new_init(self: Self, *args, **kwargs):
                old_init(self, *args, **kwargs)

                INDEX_STATE.signals.sigChangeSize.connect(self.onChangeSize)
                INDEX_STATE.signals.sigChangeToolMode.connect(self.onChangeToolMode)
                INDEX_STATE.signals.sigChangeRoiGroups.connect(self.onChangeRoiGroups)
                INDEX_STATE.signals.sigChangeRoiSelection.connect(self.onChangeRoiSelection)
                INDEX_STATE.signals.sigChangeActiveGroup.connect(self.onChangeActiveGroup)

                self.onChangeSize()
                self.onChangeToolMode()
                self.onChangeRoiGroups()
                self.onChangeRoiSelection()
                self.onChangeActiveGroup()


            cls.__init__ = new_init

            super().__init_subclass__(*args, **kwargs)

        def onChangeSize(self): pass
        def onChangeToolMode(self): pass
        def onChangeRoiGroups(self): pass
        def onChangeRoiSelection(self): pass
        def onChangeActiveGroup(self): pass

    def __init__(self) -> None:
        self.signals = self.Signals()
        self.tool_mode: str = "select" # "select", "create", "erase"
        self.roi_groups: list[ROIGroup] = []
        self.size = (1, 1)
        self.active_group_idx: int = 0
        self.active_roi_idx: Optional[int] = None

    def setToolMode(self, mode: str):
        if mode not in ("select", "create", "erase"): return
        if self.tool_mode != mode:
            self.tool_mode = mode
            self.signals.sigChangeToolMode.emit()

    def setRoiGroups(self, groups: list[ROIGroup]):
        self.roi_groups = groups
        self.active_group_idx = 0 if groups else -1
        self.active_roi_idx = None
        self.signals.sigChangeRoiGroups.emit()
        self.signals.sigChangeRoiSelection.emit()

    def setActiveRoi(self, idx: Optional[int]):
        if self.active_roi_idx != idx:
            self.active_roi_idx = idx
            self.signals.sigChangeRoiSelection.emit()

    def setActiveGroup(self, idx: int):
        if 0 <= idx < len(self.roi_groups) and self.active_group_idx != idx:
            self.active_group_idx = idx
            self.setActiveRoi(None) # Сбрасываем выделение при смене слоя
            self.signals.sigChangeActiveGroup.emit()

    def addRoiToCurrentGroup(self, roi: ROIData):
        if not self.roi_groups or self.active_group_idx < 0: return
        self.roi_groups[self.active_group_idx].rois.append(roi)
        self.signals.sigChangeRoiGroups.emit()

    def updateGroup(self, idx: int, rois: list[ROIData]):
        if 0 <= idx < len(self.roi_groups):
            self.roi_groups[idx].rois.extend(rois)
            self.signals.sigChangeRoiGroups.emit()

    def removeRoiFromCurrentGroup(self, roi: ROIData):
        if not self.roi_groups or self.active_group_idx < 0: return
        group = self.roi_groups[self.active_group_idx]
        if roi in group.rois:
            group.rois.remove(roi)
            if self.active_roi_idx == roi.idx:
                self.setActiveRoi(None)
            self.signals.sigChangeRoiGroups.emit()

    def deleteActiveRoi(self):
        if self.active_roi_idx is not None and self.active_group_idx >= 0:
            group = self.roi_groups[self.active_group_idx]
            roi_to_delete = next((r for r in group.rois if r.idx == self.active_roi_idx), None)
            if roi_to_delete:
                self.removeRoiFromCurrentGroup(roi_to_delete)

    def setSize(self, size):
        self.size = size
        self.signals.sigChangeSize.emit()

INDEX_STATE = IndexState()

