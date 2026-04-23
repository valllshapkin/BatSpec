"""
Source/BatSpec/QtApp/中Analize/中SpecView/Logic.py

Центральный стейт для панели просмотра спектрограмм.

Зависимости:
    STFT_STATE.signals.calculationFinished  → _on_stft_ready()
    RECORD_STATE.signals.recordChanged      → _on_record_changed()
    INDEX_STATE.signals.sigChangeRoiGroups  → _on_roi_changed()
    INDEX_STATE.signals.sigChangeRoiSelection → _on_roi_changed()
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, Self

import numpy as np
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool

from BatSpec.Logic.Functions import SpecFunc, TimeFunc
from BatSpec.Logic.TransitionWorker import integrateFreq
from BatSpec.Logic.SpecWorker import subEtalonNoise
from BatSpec.Logic.TransitionWorker import etalonNoise

from BatSpec.QtApp.中Analize.中STFT.Logic import STFT_STATE
from BatSpec.QtApp.中Analize.中Record.Logic import RECORD_STATE
from BatSpec.QtApp.中Analize.中SpecView.中Graph.Logic import GRAPH_STATE
from BatSpec.QtApp.中Analize.中SpecView.中Index.Logic import INDEX_STATE
from BatSpec.QtApp.中Analize.中SpecView.中Corp.Logic import CORP_STATE
from BatSpec.QtUp.PolyROI import ROIData, ROIGroup

import pyqtgraph as pg

ScriptDir = Path(__file__).parent


# ── CSV синхронизация ──────────────────────────────────────────────────────────

class ROISync:
    """Статические методы сохранения/загрузки ROI в CSV рядом с wav-файлом.

    Соглашение об именовании:
        NYCNOC_20230624_021154.wav
        NYCNOC_20230624_021154.call.csv
        NYCNOC_20230624_021154.context.csv
    """

    # Соответствие имени группы → суффиксу файла.
    # Можно расширять при добавлении новых групп.
    GROUP_SUFFIXES: dict[str, str] = {
        "call":    ".call.csv",
        "context": ".context.csv",
    }
    DEFAULT_SUFFIX = ".roi.csv"

    @staticmethod
    def _csv_path(wav_path: Path, group_name: str) -> Path:
        suffix = ROISync.GROUP_SUFFIXES.get(group_name, f".{group_name}.csv")
        return wav_path.with_suffix("").with_suffix(suffix)

    @staticmethod
    def save_group(rois: list[ROIData], wav_path: Path, group_name: str) -> None:
        path = ROISync._csv_path(wav_path, group_name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "x", "y", "w", "h"])
            for r in rois:
                writer.writerow([r.idx, r.x, r.y, r.w, r.h])

    @staticmethod
    def load_group(wav_path: Path, group_name: str) -> list[ROIData]:
        path = ROISync._csv_path(wav_path, group_name)
        rois: list[ROIData] = []
        if not path.exists():
            return rois
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rois.append(ROIData(
                    int(row["idx"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["w"]),
                    float(row["h"]),
                ))
        return rois

    @staticmethod
    def save_all(groups: list[ROIGroup], wav_path: Path) -> None:
        for group in groups:
            ROISync.save_group(group.rois, wav_path, group.name)

    @staticmethod
    def load_all(groups: list[ROIGroup], wav_path: Path) -> None:
        """Загружает ROI из CSV в существующие группы (по имени группы)."""
        for group in groups:
            loaded = ROISync.load_group(wav_path, group.name)
            if loaded:
                group.rois = loaded


# ── Вспомогательная утилита crop ──────────────────────────────────────────────

def crop_spec(func: SpecFunc, roi: ROIData) -> Optional[np.ndarray]:
    """Вырезает прямоугольник из матрицы SpecFunc по координатам ROI.

    ROI координаты — в единицах времени (x, w) и частоты (y, h).
    Возвращает np.ndarray shape (freq_bins, time_bins) или None.
    """
    if func is None or roi is None:
        return None

    t = func.time
    f = func.freq
    m = func.matrix  # shape: (time, freq)

    t0 = np.searchsorted(t, roi.x)
    t1 = np.searchsorted(t, roi.x + roi.w)
    f0 = np.searchsorted(f, roi.y)
    f1 = np.searchsorted(f, roi.y + roi.h)

    t0, t1 = int(t0), int(t1)
    f0, f1 = int(f0), int(f1)

    if t1 <= t0 or f1 <= f0:
        return None

    # Транспонируем: (time, freq) → (freq, time) для отображения
    return m[t0:t1, f0:f1].T.copy()


# ── Фоновый воркер денойзинга ─────────────────────────────────────────────────

class _DenoiseWorkerSignals(QObject):
    finished = Signal(object, object, object)  # raw_db, denoised_db | None, minimap_func
    error    = Signal(str)


class _DenoiseWorker(QRunnable):
    """Выполняет денойзинг и построение минимапа в фоновом потоке."""

    def __init__(self, raw: SpecFunc) -> None:
        super().__init__()
        self.raw     = raw
        self.signals = _DenoiseWorkerSignals()

    def run(self) -> None:
        try:
            raw = self.raw
            raw_db = raw.cloneApply(lambda arr: 20 * np.log10(np.clip(arr, 1e-9, None)))

            try:
                noise       = etalonNoise(raw, percentile=5)
                denoised    = subEtalonNoise(raw, noise)
                denoised_db = denoised.cloneApply(
                    lambda arr: 20 * np.log10(np.clip(arr, 1e-9, None))
                )
                minimap_func = integrateFreq(denoised)
            except Exception as e:
                print(f"[_DenoiseWorker] Денойзинг не удался, fallback на raw: {e}")
                denoised_db  = None
                minimap_func = integrateFreq(raw)

            self.signals.finished.emit(raw_db, denoised_db, minimap_func)

        except Exception as e:
            import traceback; traceback.print_exc()
            self.signals.error.emit(str(e))


# ── Основной стейт ────────────────────────────────────────────────────────────

class SpecViewState:
    """Центральный стейт панели SpecView.

    Хранит словарь версий спектрограмм и координирует
    GRAPH_STATE, INDEX_STATE и CORP_STATE.
    """

    class Signals(QObject):
        sigSpecDictChanged  = Signal()      # добавили/удалили ключ из словаря
        sigActiveKeyChanged = Signal()      # переключили активную версию
        sigLoadingChanged   = Signal(bool)  # True = идёт расчёт, показать спиннер
        sigDirtyChanged     = Signal(bool)  # True = ROI изменились, не сохранено

    class Trigger:
        """Миксин: подписывает подклассы на сигналы SPEC_VIEW_STATE."""

        def __init_subclass__(cls, *args, **kwargs):
            old_init = cls.__init__

            def new_init(self: Self, *args, **kwargs):
                old_init(self, *args, **kwargs)

                SPEC_VIEW_STATE.signals.sigSpecDictChanged.connect(self.onSpecDictChanged)
                SPEC_VIEW_STATE.signals.sigActiveKeyChanged.connect(self.onActiveKeyChanged)
                SPEC_VIEW_STATE.signals.sigLoadingChanged.connect(self.onLoadingChanged)
                SPEC_VIEW_STATE.signals.sigDirtyChanged.connect(self.onDirtyChanged)

                self.onSpecDictChanged()
                self.onActiveKeyChanged()
                self.onLoadingChanged(SPEC_VIEW_STATE.loading)
                self.onDirtyChanged(SPEC_VIEW_STATE._dirty)

            cls.__init__ = new_init
            super().__init_subclass__(*args, **kwargs)

        def onSpecDictChanged(self): pass
        def onActiveKeyChanged(self): pass
        def onLoadingChanged(self, loading: bool): pass
        def onDirtyChanged(self, dirty: bool): pass

    # ── Состояние ─────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.signals  = self.Signals()
        self.specs:    dict[str, SpecFunc] = {}
        self.active_key: str = ""
        self.loading:  bool  = False
        self._dirty:   bool  = False
        self._denoise_worker: _DenoiseWorker | None = None  # защита от GC

        # Подписываемся на внешние стейты
        STFT_STATE.signals.calculationFinished.connect(self._on_stft_ready)
        STFT_STATE.signals.calculationStarted.connect(self._on_stft_started)
        RECORD_STATE.signals.recordChanged.connect(self._on_record_changed)

        # Грязность ROI
        INDEX_STATE.signals.sigChangeRoiGroups.connect(self._on_roi_changed)
        INDEX_STATE.signals.sigChangeRoiSelection.connect(self._on_selection_changed)

        # Инициализируем дефолтные группы сразу — на случай если RECORD_STATE
        # уже имеет файл (восстановление из настроек) и сигнал мы пропустили.
        # Если файла нет — группы всё равно создаются пустыми.
        self.loadCSV()

    # ── Публичный API ─────────────────────────────────────────────────────────

    def addSpec(self, key: str, func: SpecFunc) -> None:
        """Добавляет (или заменяет) версию спектрограммы в словарь."""
        self.specs[key] = func
        self.signals.sigSpecDictChanged.emit()
        # Если активного ключа ещё нет — автоматически выбираем первый
        if not self.active_key:
            self.setActiveKey(key)

    def removeSpec(self, key: str) -> None:
        if key not in self.specs:
            return
        del self.specs[key]
        self.signals.sigSpecDictChanged.emit()
        if self.active_key == key:
            new_key = next(iter(self.specs), "")
            self.setActiveKey(new_key)

    def setActiveKey(self, key: str) -> None:
        if key == self.active_key and key in self.specs:
            return
        self.active_key = key
        self._sync_graph_index(key)
        self.signals.sigActiveKeyChanged.emit()

    # ── CSV ───────────────────────────────────────────────────────────────────

    def saveCSV(self) -> None:
        """Сохраняет все ROI-группы в CSV рядом с wav-файлом."""
        if not RECORD_STATE.path:
            return
        ROISync.save_all(INDEX_STATE.roi_groups, RECORD_STATE.path)
        self._set_dirty(False)

    # ── Дефолтные группы ──────────────────────────────────────────────────────

    DEFAULT_GROUPS: list[tuple[str, str, str]] = [
        # (name, color, selected_color)
        ("call",    "#00ff00", "#ffff00"),
        ("detect",  "#00ff00", "#ffff00"),
        ("context", "#ff4444", "#ffffff"),
    ]

    @staticmethod
    def _make_default_groups() -> list[ROIGroup]:
        """Создаёт пустые дефолтные группы call и context."""
        groups = []
        for name, color, sel_color in SpecViewState.DEFAULT_GROUPS:
            groups.append(ROIGroup(
                name,
                rois=[],
                base_color=pg.mkColor(color),
                active_color=pg.mkColor(sel_color),
            ))
        return groups

    def loadCSV(self) -> None:
        """Инициализирует дефолтные группы и загружает ROI из CSV если есть."""
        # Временно отключаем слежение за dirty чтобы инициализация не пометила
        # группы как "изменённые"
        INDEX_STATE.signals.sigChangeRoiGroups.disconnect(self._on_roi_changed)

        groups = self._make_default_groups()
        if RECORD_STATE.path:
            ROISync.load_all(groups, RECORD_STATE.path)
        INDEX_STATE.setRoiGroups(groups)

        INDEX_STATE.signals.sigChangeRoiGroups.connect(self._on_roi_changed)
        self._set_dirty(False)

    # ── Внутренняя логика ─────────────────────────────────────────────────────

    def _on_stft_started(self) -> None:
        self._set_loading(True)

    def _on_stft_ready(self) -> None:
        """Вызывается когда STFT_STATE закончил расчёт.
        Запускает денойзинг в отдельном потоке чтобы не блокировать UI.
        """
        raw: SpecFunc | None = STFT_STATE.output_stft
        if raw is None:
            self._set_loading(False)
            return

        worker = _DenoiseWorker(raw)
        self._denoise_worker = worker  # защита от GC
        worker.signals.finished.connect(self._on_denoise_ready)
        worker.signals.error.connect(lambda e: (
            print(f"[SpecViewState] Ошибка денойзинга: {e}"),
            self._set_loading(False),
        ))
        QThreadPool.globalInstance().start(worker)

    def _on_denoise_ready(self, raw_db: SpecFunc, denoised_db: SpecFunc | None,
                          minimap_func: TimeFunc) -> None:
        """Вызывается из потока когда денойзинг завершён."""
        self._denoise_worker = None  # освобождаем
        self.specs["raw"] = raw_db
        if denoised_db is not None:
            self.specs["denoised"] = denoised_db

        GRAPH_STATE.setMinimapFunc(minimap_func)
        self.signals.sigSpecDictChanged.emit()

        default_key = "denoised" if "denoised" in self.specs else "raw"
        self.active_key = ""
        self.setActiveKey(default_key)

        self._set_loading(False)

    def _on_record_changed(self) -> None:
        """Сброс при смене файла записи."""
        self.specs.clear()
        self.active_key = ""
        self._set_dirty(False)

        # Сбрасываем дочерние стейты
        GRAPH_STATE.setSpecFunc(None)
        GRAPH_STATE.setMinimapFunc(None)
        INDEX_STATE.setSize((1, 1))
        CORP_STATE.setData(None, None)

        self.signals.sigSpecDictChanged.emit()
        self.signals.sigActiveKeyChanged.emit()

        # Загружаем сохранённые ROI для нового файла
        self.loadCSV()

        # Если STFT уже посчитан для этого файла — сразу показываем
        # (на случай когда файл меняется, но пересчёт не запускается)
        if STFT_STATE.output_stft is not None and not STFT_STATE.isNewInput():
            self._on_stft_ready()
        else:
            self._set_loading(True)

    def _on_roi_changed(self) -> None:
        """ROI изменились — помечаем как несохранённые."""
        self._set_dirty(True)
        self._update_corp()

    def _on_selection_changed(self) -> None:
        """Сменился активный ROI — обновляем CORP_STATE."""
        self._update_corp()

    def _update_corp(self) -> None:
        """Передаёт актуальные данные в CORP_STATE."""
        func = self.specs.get(self.active_key)
        roi  = self._get_active_roi()
        CORP_STATE.setData(func, roi)

    def _get_active_roi(self) -> Optional[ROIData]:
        idx = INDEX_STATE.active_roi_idx
        gidx = INDEX_STATE.active_group_idx
        if idx is None or gidx < 0 or not INDEX_STATE.roi_groups:
            return None
        group = INDEX_STATE.roi_groups[gidx]
        return next((r for r in group.rois if r.idx == idx), None)

    def _sync_graph_index(self, key: str) -> None:
        """Единственное место синхронизации GRAPH_STATE + INDEX_STATE + CORP_STATE."""
        func = self.specs.get(key)

        GRAPH_STATE.setSpecFunc(func)

        if func is not None:
            t_max = float(func.time[-1]) if len(func.time) > 0 else 1.0
            f_max = float(func.freq[-1]) if len(func.freq) > 0 else 1.0
            INDEX_STATE.setSize((t_max, f_max))
        else:
            INDEX_STATE.setSize((1, 1))

        self._update_corp()

    def _set_loading(self, value: bool) -> None:
        if self.loading != value:
            self.loading = value
            self.signals.sigLoadingChanged.emit(value)

    def _set_dirty(self, value: bool) -> None:
        if self._dirty != value:
            self._dirty = value
            self.signals.sigDirtyChanged.emit(value)


SPEC_VIEW_STATE = SpecViewState()