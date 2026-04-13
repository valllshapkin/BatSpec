from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool
from pathlib import Path
import traceback

from ..中Record.Logic import RECORD_STATE
from BatSpec.Logic.TransitionWorker import makeSpec
from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.ConvWindow import Window, load_file_window
from BatSpec.QtApp.Shared.Component import ComponentSettings
from BatSpec.QtUp.Settings import SettingField

ScriptDir = Path(__file__).parent


class STFTDataWorker(QRunnable):
    class Signals(QObject):
        finished = Signal(object)
        error = Signal(str)

    def __init__(self):
        super().__init__()
        self.signals = self.Signals()

    def run(self):
        try:
            if not RECORD_STATE.record: 
                raise RuntimeError("Нет записи")
            if not STFT_STATE.window:
                raise RuntimeError("Нет окна")

            # Читаем ВСЁ напрямую из глобальных стейтов (никакого клонирования)
            spec = makeSpec(
                RECORD_STATE.record.corected, 
                STFT_STATE.window, 
                STFT_STATE.overlap, 
                STFT_STATE.bins
            )
            self.signals.finished.emit(spec)
            
        except Exception as e:
            print("Ошибка внутри потока STFTDataWorker:")
            traceback.print_exc()
            self.signals.error.emit(str(e))


class StftState:
    class Settings(ComponentSettings):
        COMPONENT_DIR=ScriptDir
        window = SettingField[str](default="", type=str, key="STFT/window")
        overlap = SettingField[float](default=0.8, type=float, key="STFT/overlap")
        bins = SettingField[int](default=500, type=int, key="STFT/bins")
        
    class Signals(QObject):
        calculationStarted = Signal()
        calculationFinished = Signal() 
        calculationError = Signal(str)
        parametersChanged = Signal()

    window_name: str = ""
    window: Window | None = None
    overlap: float = 0.8
    bins: int = 500
    
    input_cache: tuple = () 
    output_stft: SpecFunc | None = None
    
    # ССЫЛКА ДЛЯ ЗАЩИТЫ ОТ СБОРЩИКА МУСОРА
    _current_worker: STFTDataWorker | None = None

    def __init__(self):
        super().__init__()
        self.signals = self.Signals()
        self.settings = self.Settings()

        if self.settings.window: 
            self.setWindowFile(Path(self.settings.window))
        self.overlap = self.settings.overlap
        self.bins = self.settings.bins

        self.threadpool = QThreadPool.globalInstance()

    def isNewInput(self):
        if self.output_stft is None:
            return True
        # Сравниваем через tuple, округляя float, чтобы избежать проблем с 0.80000000004
        current = (self.window_name, round(self.overlap, 4), self.bins)
        return current != self.input_cache

    def setWindowFile(self, path: Path | None):
        if path:
            d = load_file_window(path)
            if d: self.window_name, self.window = next(iter(d.items()))
        self.settings.window = str(path)
        self.signals.parametersChanged.emit()

    def setOverlap(self, ol: float):
        self.overlap = ol
        self.settings.overlap = ol
        self.signals.parametersChanged.emit()

    def setBins(self, b: int):
        self.bins = b
        self.settings.bins = b
        self.signals.parametersChanged.emit()

    def startCalculation(self):
        if self.window is None or not RECORD_STATE.record: 
            return self.signals.calculationError.emit("Нет данных для генерации спектрограммы.")
            
        self.signals.calculationStarted.emit()
        worker = STFTDataWorker()
        
        # Сохраняем стейт в локальную переменную перед запуском
        current_state = (self.window_name, round(self.overlap, 4), self.bins)

        # === ТО САМОЕ ИЗМЕНЕНИЕ: СПАСАЕМ ОТ GC ===
        self._current_worker = worker

        @worker.signals.finished.connect
        def _(spec: SpecFunc):
            self.output_stft = spec
            self.input_cache = current_state
            self._current_worker = None # Освобождаем память!
            self.signals.calculationFinished.emit()
            
        @worker.signals.error.connect
        def _(err: str):
            self._current_worker = None # Освобождаем память!
            self.signals.calculationError.emit(err)

        self.threadpool.start(worker)

STFT_STATE = StftState()