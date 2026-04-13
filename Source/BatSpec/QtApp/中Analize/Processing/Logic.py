import logging
import numpy as np
from enum import Enum
from scipy.ndimage import label, find_objects, uniform_filter, laplace
from skimage import morphology
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool
from dataclasses import dataclass
import cv2

from ..STFTModel.Logic import STFT_MODEL_STATE
from BatSpec.Logic.Echo import remove_echo_wiener2 as remove_echo_wiener, TimeDomainEchoModel

logger = logging.getLogger(__name__)

class PipelineMode(Enum):
    SKIP = "Skip Detection"
    SMOOTHNESS = "Smoothness (Laplacian)"
    ECHO = "Echo Cancellation"

def build_filters():
    filters = []
    ksize = 25 
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 12.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

def process_gabor(img, filters):
    accum = np.zeros_like(img, dtype=np.float32)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_32F, kern)
        np.maximum(accum, fimg, accum) 
    return accum

def normalize_globally(db_matrix: np.ndarray, rms: np.ndarray, percentile: float, freq_smooth_window: int = 3):
    noise_threshold = np.percentile(rms, percentile)
    noise_idx       = np.where(rms <= noise_threshold)[0]
    noise_mask      = np.zeros(db_matrix.shape[1], dtype=bool)

    if len(noise_idx) < 2:
        logger.warning("Too few noise frames — skipping global normalisation.")
        return db_matrix.copy()

    noise_mask[noise_idx] = True
    noise_frames = db_matrix[:, noise_idx]
    mean = np.mean(noise_frames, axis=1)
    std  = np.std(noise_frames,  axis=1)
    std[std < 1e-6] = 1.0

    if freq_smooth_window > 1:
        kernel = np.ones(freq_smooth_window) / freq_smooth_window
        mean = np.convolve(mean, kernel, mode="same")
        std  = np.convolve(std,  kernel, mode="same")
        std[std < 1e-6] = 1.0 

    return (db_matrix - mean[:, np.newaxis]) / std[:, np.newaxis]

@dataclass
class ROIBoundingBox:
    f_min_idx: int
    f_max_idx: int
    t_min_idx: int
    t_max_idx: int

@dataclass
class ProcessingResult:
    duration:      float
    max_freq:      float
    time_envelope: np.ndarray
    roi_boxes:     list[ROIBoundingBox]
    layers:        dict[str, np.ndarray] # Динамический словарь слоев!
    is_manual:     bool = False

class ProcessingWorker(QRunnable):
    def __init__(self, state_config, is_manual):
        super().__init__()
        self.cfg = state_config
        self.is_manual = is_manual
        self.signals = _WorkerSignals()

    def run(self):
        try:
            db_matrix = self.cfg['_raw_db'].astype(np.float32)
            rms = self.cfg['_raw_rms']
            mode = self.cfg['pipeline_mode']
            
            layers = {"Original (dB)": db_matrix}
            roi_boxes = []
            
            # Базовый Z-norm (используется для envelope и пропуска/гладкости)
            z_base = normalize_globally(db_matrix, rms, self.cfg['percentile']).astype(np.float32)
            layers["Normalized (Z)"] = z_base
            
            time_envelope = z_base.sum(axis=0)
            if time_envelope.max() > 0:
                time_envelope = time_envelope / time_envelope.max()

            if mode == PipelineMode.SKIP:
                pass 

            elif mode == PipelineMode.SMOOTHNESS:
                window_size = 3
                variance = -laplace(z_base)
                c1 = uniform_filter(variance**2, size=window_size)
                c2 = uniform_filter(variance, size=window_size)**2
                variance = np.maximum(c1 - c2, 0.0)
                inv_variance = 1 / (1 + variance)
                
                filters = build_filters()
                gabor = process_gabor(inv_variance, filters)**2
                binary = morphology.dilation(gabor >= 0.9, morphology.disk(4))

                layers.update({
                    "Negative Laplace": variance.astype(np.float32),
                    "Invert Variance": inv_variance.astype(np.float32),
                    "Gabor Output": gabor.astype(np.float32),
                    "Binary Mask": binary.astype(np.float32)
                })
                roi_boxes = self._extract_rois(binary)

            elif mode == PipelineMode.ECHO:
                num_freqs, num_times = db_matrix.shape
                stft_sr = num_times / self.cfg['duration']
                freqs = np.linspace(0, self.cfg['max_freq'], num_freqs)

                model = TimeDomainEchoModel(
                    decay_rate_base=self.cfg['echo_decay'],
                    f_ref=40000.0, 
                    freq_exp=self.cfg['echo_n'],
                    time_power=self.cfg['echo_beta']
                )

                # 1. ЧИСТИМ СЫРЫЕ ДАННЫЕ (db_matrix), а не Z-score
                cleaned_db = np.zeros_like(db_matrix)
                
                # Для фильтра Винера полезно сместить сигнал так, чтобы "пол" был около нуля.
                # Вычитаем медиану из каждой полосы перед деконволюцией.
                for f_idx in range(num_freqs):
                    f_val = freqs[f_idx]
                    row_data = db_matrix[f_idx, :]
                    
                    if f_val < 100: 
                        cleaned_db[f_idx, :] = row_data
                        continue
                        
                    baseline = np.median(row_data)
                    row_shifted = row_data - baseline
                    
                    cleaned_row = remove_echo_wiener(
                        signal=row_shifted, 
                        frequency=f_val, 
                        sample_rate=stft_sr, 
                        model=model, 
                        eps_factor=self.cfg['echo_eps']
                    )
                    
                    # Возвращаем бейслайн обратно
                    cleaned_db[f_idx, :] = cleaned_row + baseline

                layers["Echo Cleaned (dB)"] = cleaned_db.astype(np.float32)

                # 2. ВОТ ТЕПЕРЬ ДЕЛАЕМ Z-НОРМАЛИЗАЦИЮ ОЧИЩЕННОГО СИГНАЛА
                # RMS тоже можно взять от очищенной матрицы, либо использовать старый RMS. 
                # Проще просто взять дисперсию шума прямо из cleaned_db.
                cleaned_z = normalize_globally(cleaned_db, rms, self.cfg['percentile']).astype(np.float32)
                
                # Убираем негативный звон, оставляем только положительные пики (сигмы > 0)
                cleaned_z = np.clip(cleaned_z, 0, None)
                layers["Cleaned Normalized (Z)"] = cleaned_z

                # 3. ПРИМЕНЯЕМ ГАБОРА К ОЧИЩЕННЫМ СИГМАМ
                # Масштабируем так, чтобы макс пик был около 1.0 (Габор любит диапазон 0-1)
                c_max = cleaned_z.max()
                gabor_input = cleaned_z / c_max if c_max > 0 else cleaned_z

                filters = build_filters()
                gabor = process_gabor(gabor_input, filters)**2
                
                # Порог: например, берем всё, что сильнее 10% от максимума отклика
                threshold = np.percentile(gabor[gabor > 0], 90) if np.any(gabor > 0) else 0.5
                binary = morphology.dilation(gabor >= threshold, morphology.disk(4))

                layers.update({
                    "Gabor Output": gabor.astype(np.float32),
                    "Binary Mask": binary.astype(np.float32)
                })
                
                roi_boxes = self._extract_rois(binary)
                
                # Обновляем огибающую для миникарты по очищенным данным
                time_envelope = cleaned_z.sum(axis=0)
                if time_envelope.max() > 0:
                    time_envelope = time_envelope / time_envelope.max()

            result = ProcessingResult(
                duration=self.cfg['duration'],
                max_freq=self.cfg['max_freq'],
                time_envelope=time_envelope,
                roi_boxes=roi_boxes,
                layers=layers,
                is_manual=self.is_manual
            )
            self.signals.finished.emit(result)
        except Exception as exc:
            logger.exception("ProcessingWorker failed")
            self.signals.error.emit(str(exc))


    def _extract_rois(self, binary_mask):
        labeled_array, _ = label(binary_mask)
        areas = np.bincount(labeled_array.ravel())
        slices = find_objects(labeled_array)
        boxes = []
        for label_idx, slc in enumerate(slices, start=1):
            if slc is None: continue
            if areas[label_idx] > 500:
                f_slc, t_slc = slc
                boxes.append(ROIBoundingBox(
                    f_min_idx=f_slc.start, f_max_idx=f_slc.stop,
                    t_min_idx=t_slc.start, t_max_idx=t_slc.stop
                ))
        return boxes

class _WorkerSignals(QObject):
    finished = Signal(object)
    error    = Signal(str)

class _ProcessingState(QObject):
    calculationStarted  = Signal()
    calculationFinished = Signal(object)
    calculationError    = Signal(str)
    readyToCalculate    = Signal()

    def __init__(self):
        super().__init__()
        self._threadpool = QThreadPool.globalInstance()
        self.result: ProcessingResult | None = None

        # Конфиг собираем в словарь для удобной передачи в Воркер
        self.config = {
            'pipeline_mode': PipelineMode.SMOOTHNESS,
            'percentile': 20.0,
            'echo_decay': 40.0,
            'echo_n': 2.0,
            'echo_beta': 20.0,
            'echo_eps': 0.01,
            'duration': 1.0,
            'max_freq': 1.0,
            '_raw_db': None,
            '_raw_rms': None
        }
        STFT_MODEL_STATE.calculationFinished.connect(self._on_stft_finished)

    def _on_stft_finished(self, db_t, duration, max_freq, rms):
        self.config['duration'] = duration
        self.config['max_freq'] = max_freq
        self.config['_raw_db'] = db_t
        self.config['_raw_rms'] = rms
        self.readyToCalculate.emit()
        self.start_calculation(is_manual=False)

    def set_param(self, key, value): self.config[key] = value

    def recalculate(self): self.start_calculation(is_manual=True)

    def start_calculation(self, is_manual=False):
        if self.config['_raw_db'] is None:
            self.calculationError.emit("Нет спектрограммы.")
            return
        self.calculationStarted.emit()
        worker = ProcessingWorker(self.config.copy(), is_manual)
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self.calculationError.emit)
        self._threadpool.start(worker)

    def _on_worker_finished(self, result: ProcessingResult):
        self.result = result
        self.calculationFinished.emit(result)

PROCESSING_STATE = _ProcessingState()