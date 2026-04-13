from typing import Any, Optional, Self
import numpy as np
from PySide6.QtCore import QObject, Signal
from BatSpec.QtUp.PolyROI import ROIData

class CorpState:
    class Signals(QObject):
        sigDataChanged = Signal()      # Когда сменился ROI или SpecFunc
        sigPaddingChanged = Signal()
        sigProcessFinished = Signal()   # Когда результат готов

    class Trigger:
        def __init_subclass__(cls, *args, **kwargs):
            old_init = cls.__init__
            def new_init(self: Self, *args, **kwargs):
                old_init(self, *args, **kwargs)
                CORP_STATE.signals.sigDataChanged.connect(self.onDataUpdate)
                CORP_STATE.signals.sigPaddingChanged.connect(self.onPaddingUpdate)
                CORP_STATE.signals.sigProcessFinished.connect(self.onResultUpdate)
                self.onDataUpdate()
                self.onPaddingUpdate()
                self.onResultUpdate()
            cls.__init__ = new_init
            super().__init_subclass__(*args, **kwargs)

        def onDataUpdate(self): pass
        def onPaddingUpdate(self): pass
        def onResultUpdate(self): pass

    def __init__(self) -> None:
        self.signals = self.Signals()
        self.spec_func: Any = None  # Изолированная SpecFunc
        self.roi: Optional[ROIData] = None
        self.padding: int = 5
        
        # Результаты
        self.cropped_matrix: Optional[np.ndarray] = None
        self.result_data: Any = None # Сюда пишем результат аппроксимации (набор точек)
        
        self.user_code: str = """# 'crop' is np.ndarray
# 'result' must be list of (x, y) points
import numpy as np
# Находим самую яркую точку в каждой колонке
y_indices = np.argmax(crop, axis=0)
x_indices = np.arange(len(y_indices))
result = list(zip(x_indices, y_indices))
"""

    def setData(self, spec_func, roi: ROIData):
        self.spec_func = spec_func
        self.roi = roi
        self.signals.sigDataChanged.emit()

    def setPadding(self, val: int):
        self.padding = val
        self.signals.sigPaddingChanged.emit()

    def run_process(self):
        if self.spec_func is None or self.roi is None: return
        
        # 1. Вырезаем Crop с педдингом
        m = self.spec_func.matrix
        h, w = m.shape
        
        x1 = max(0, int(self.roi.x) - self.padding)
        y1 = max(0, int(self.roi.y) - self.padding)
        x2 = min(w, int(self.roi.x + self.roi.w) + self.padding)
        y2 = min(h, int(self.roi.y + self.roi.h) + self.padding)
        
        self.cropped_matrix = m[y1:y2, x1:x2].copy()
        
        # 2. Выполняем пользовательский код
        loc = {"crop": self.cropped_matrix, "np": np}
        try:
            exec(self.user_code, {}, loc)
            self.result_data = loc.get("result", [])
        except Exception as e:
            print(f"Error in user code: {e}")
            self.result_data = None
            
        self.signals.sigProcessFinished.emit()

CORP_STATE = CorpState()