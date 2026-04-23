"""
Source/BatSpec/QtApp/中Analize/中SpecView/中Auto/Logic.py

Модуль автодетекции сигналов.

Пайплайн:
    1. Пользователь пишет скрипт в редакторе
    2. Скрипт выполняется в потоке с доступом ко всем версиям спектрограмм
    3. Скрипт заполняет словарь results: dict[str, list[ROIData]]
       где ключ — имя группы ROI (совпадает с именами в INDEX_STATE)
    4. Каждый список ROI записывается в соответствующую группу INDEX_STATE

Переменные доступные в exec-окружении скрипта:
    specs      : dict[str, SpecFunc]       — все версии из SPEC_VIEW_STATE
    record     : TimeFunc | None           — исходная запись из RECORD_STATE
    np         : numpy
    label      : scipy.ndimage.label
    bbox_to_roi: callable                  — утилита bins→ROI с осями SpecFunc
    results    : dict[str, list[ROIData]]  — СЮДА пишем результаты

Пример использования в скрипте:
    mask = (norm.matrix > 10)
    labeled, n = label(mask)
    results["call"] = bbox_to_roi(labeled, n, ref_spec=spec, min_area=50)
    results["context"] = bbox_to_roi(labeled2, n2, ref_spec=raw, min_area=10)
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional, Self

import numpy as np
from scipy.ndimage import label as ndimage_label
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from BatSpec.QtApp.中Analize.中SpecView.Logic import SPEC_VIEW_STATE
from BatSpec.QtApp.中Analize.中SpecView.中Index.Logic import INDEX_STATE
from BatSpec.QtApp.中Analize.中Record.Logic import RECORD_STATE
from BatSpec.QtUp.PolyROI import ROIData

ScriptDir = Path(__file__).parent


# ── Вспомогательная утилита для скриптов ──────────────────────────────────────

def bbox_to_roi(
    labeled: np.ndarray,
    n_components: int,
    ref_spec,
    min_area: int = 0,
) -> list[ROIData]:
    """
    Конвертирует результат scipy.ndimage.label в список ROIData.

    Args:
        labeled       : массив меток из ndimage.label, shape (time, freq)
        n_components  : количество компонент (второй результат ndimage.label)
        ref_spec      : SpecFunc для перевода bins → секунды/Гц
        min_area      : минимальная площадь компоненты в bins², меньше — отбросить

    Returns:
        list[ROIData] с координатами в единицах времени и частоты
    """
    t_axis = getattr(ref_spec, "time", None)
    f_axis = getattr(ref_spec, "freq", None)

    rois: list[ROIData] = []
    roi_idx = 0

    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        area = int(comp_mask.sum())
        if area < min_area:
            continue

        t_indices = np.where(comp_mask.any(axis=1))[0]
        f_indices = np.where(comp_mask.any(axis=0))[0]
        if len(t_indices) == 0 or len(f_indices) == 0:
            continue

        t0_bin = int(t_indices[0])
        t1_bin = int(t_indices[-1])
        f0_bin = int(f_indices[0])
        f1_bin = int(f_indices[-1])

        if t_axis is not None and f_axis is not None:
            t0 = float(t_axis[t0_bin])
            t1 = float(t_axis[min(t1_bin, len(t_axis) - 1)])
            f0 = float(f_axis[f0_bin])
            f1 = float(f_axis[min(f1_bin, len(f_axis) - 1)])
        else:
            t0, t1 = float(t0_bin), float(t1_bin)
            f0, f1 = float(f0_bin), float(f1_bin)

        rois.append(ROIData(
            idx=roi_idx,
            x=t0,
            y=f0,
            w=max(t1 - t0, 1e-9),
            h=max(f1 - f0, 1e-9),
        ))
        roi_idx += 1

    return rois


# ── Дефолтный скрипт ──────────────────────────────────────────────────────────

DEFAULT_SCRIPT = '''\
from typing import Any

from numpy.typing import NDArray

from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.Functions import SpecFunc
# from BatSpec.Logic.SpecWorker import (
#     normalizeFrecZ, removeEchoRichardsonLucy, enhanceCurvesGabor, subEtalonNoise, makeBinarization, makeLog
# )
from BatSpec.Logic.SpecWorker.GRAPH import (
    subEtalonNoise, removeEchoRichardsonLucy, makeLog, enhanceCurvesGabor
)
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.Echo import TimeDomainEchoModel


SPEC = getSpectrogramm()

noise = etalonNoise(SPEC, 5)

DENOISED = subEtalonNoise(SPEC, noise, alpha=1)
updateVisLayer("DENOISED", DENOISED)

GABOR = enhanceCurvesGabor(DENOISED, ksize=17, sigma=1, lambd=5, pre_blur=0)
updateVisLayer("GABOR", GABOR)

RICHLUCY = removeEchoRichardsonLucy(GABOR, TimeDomainEchoModel(
    decay_rate_base=400, f_ref=40000.0,
    freq_exp=2.0, time_power=20.0,
), iterations=20)
updateVisLayer("RICHLUCY", RICHLUCY)

LOG = makeLog(RICHLUCY)
updateVisLayer("LOG", LOG)


# NORM = normalizeFrecZ(GABOR, noise)
# updateVisLayer("NORM", NORM)

# BINARY = makeBinarization(NORM, trashhold=32)
# updateVisLayer("BINARY", BINARY)

'''


# ── Воркер ────────────────────────────────────────────────────────────────────

class _AutoWorkerSignals(QObject):
    finished = Signal(object)  # dict[str, list[ROIData]]
    progress = Signal(str)
    error    = Signal(str)


class _AutoWorker(QRunnable):
    """Выполняет скрипт автодетекции в фоновом потоке."""

    def __init__(self, script: str) -> None:
        super().__init__()
        self.script  = script
        self.signals = _AutoWorkerSignals()

    def run(self) -> None:
        try:
            self._run()
        except Exception:
            self.signals.error.emit(traceback.format_exc())

    def _run(self) -> None:
        sig = self.signals

        # Инициализируем results пустыми списками для всех известных групп
        results: dict[str, list[ROIData]] = {
            g.name: [] for g in INDEX_STATE.roi_groups
        }

        env = {
            "specs"      : dict(SPEC_VIEW_STATE.specs),
            "record"     : RECORD_STATE.record.corected if RECORD_STATE.record else None,
            "np"         : np,
            "label"      : ndimage_label,
            "bbox_to_roi": bbox_to_roi,
            "results"    : results,
        }

        sig.progress.emit("▶ Выполняю скрипт…")
        exec(self.script, env)

        # results мог быть переприсвоен внутри скрипта — берём из env
        results = env.get("results", results)
        if not isinstance(results, dict):
            raise RuntimeError(
                f"`results` должен быть dict[str, list[ROIData]], "
                f"получен {type(results).__name__}"
            )

        total = sum(len(v) for v in results.values())
        for group_name, rois in results.items():
            if rois:
                sig.progress.emit(f"  '{group_name}': {len(rois)} ROI")
        sig.progress.emit(f"✓ Итого ROI: {total}")

        sig.finished.emit(results)


# ── Стейт ─────────────────────────────────────────────────────────────────────

class AutoState:
    """Стейт модуля автодетекции."""

    class Signals(QObject):
        sigRunningChanged = Signal(bool)
        sigLogAppended    = Signal(str)
        sigFinished       = Signal(int)   # суммарное количество ROI
        sigError          = Signal(str)

    class Trigger:
        def __init_subclass__(cls, *args, **kwargs):
            old_init = cls.__init__

            def new_init(self: Self, *args, **kwargs):
                old_init(self, *args, **kwargs)
                AUTO_STATE.signals.sigRunningChanged.connect(self.onRunningChanged)
                AUTO_STATE.signals.sigLogAppended.connect(self.onLogAppended)
                AUTO_STATE.signals.sigFinished.connect(self.onFinished)
                AUTO_STATE.signals.sigError.connect(self.onError)
                self.onRunningChanged(AUTO_STATE.running)

            cls.__init__ = new_init
            super().__init_subclass__(*args, **kwargs)

        def onRunningChanged(self, running: bool): pass
        def onLogAppended(self, line: str): pass
        def onFinished(self, count: int): pass
        def onError(self, msg: str): pass

    def __init__(self) -> None:
        self.signals       = self.Signals()
        self.script: str   = DEFAULT_SCRIPT
        self.running: bool = False
        self._worker: Optional[_AutoWorker] = None  # защита от GC

    # ── Публичный API ──────────────────────────────────────────────────────────

    def setScript(self, text: str) -> None:
        self.script = text

    def run(self) -> None:
        if self.running:
            return
        if not SPEC_VIEW_STATE.specs:
            self.signals.sigError.emit("Нет доступных спектрограмм.")
            return

        self._set_running(True)
        self.signals.sigLogAppended.emit("═" * 40)

        worker = _AutoWorker(self.script)
        self._worker = worker  # защита от GC

        worker.signals.progress.connect(self.signals.sigLogAppended)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)

        QThreadPool.globalInstance().start(worker)

    # ── Внутреннее ────────────────────────────────────────────────────────────

    def _on_finished(self, results: dict[str, list[ROIData]]) -> None:
        self._worker = None
        self._set_running(False)

        total = 0
        missing: list[str] = []

        for group_name, rois in results.items():
            if not rois:
                continue
            target = next(
                (g for g in INDEX_STATE.roi_groups if g.name == group_name),
                None,
            )
            if target is None:
                missing.append(group_name)
                continue

            existing_max = max((r.idx for r in target.rois), default=-1)
            for i, roi in enumerate(rois):
                roi.idx = existing_max + 1 + i
            target.rois.extend(rois)
            total += len(rois)

        if missing:
            self.signals.sigLogAppended.emit(
                f"⚠ Группы не найдены в INDEX_STATE: {missing}"
            )

        # Один emit после всех изменений
        INDEX_STATE.signals.sigChangeRoiGroups.emit()

        self.signals.sigLogAppended.emit(f"✓ Записано ROI: {total}")
        self.signals.sigFinished.emit(total)

    def _on_error(self, msg: str) -> None:
        self._worker = None
        self._set_running(False)
        self.signals.sigLogAppended.emit(f"✗ ОШИБКА:\n{msg}")
        self.signals.sigError.emit(msg)

    def _set_running(self, val: bool) -> None:
        if self.running != val:
            self.running = val
            self.signals.sigRunningChanged.emit(val)


AUTO_STATE = AutoState()