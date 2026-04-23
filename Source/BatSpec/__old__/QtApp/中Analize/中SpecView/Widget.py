"""
Source/BatSpec/QtApp/中Analize/中SpecView/Widget.py

Главный виджет панели просмотра спектрограмм.

Компоновка:
    [TOP BAR]  SpecKeySelector | ColorMapSelector | LockYCheckBox |
               RoiGroupSelector | RoiToolSelector | SaveROIButton
    [CENTER]
        [ВЕРХ]  MainGraph + IndexLayer + LoadingOverlay
        [НИЗ]
            [ЛЕВО]   CorpusPreviewPanel
            [ПРАВО]  CorpProcessMainWidget (из 中Corp)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QStackedWidget, QToolBar,
    QVBoxLayout, QWidget,
)

from BatSpec.QtApp.中Analize.中SpecView.中Corp.Widget import CorpProcessMainWidget
from BatSpec.QtApp.中Analize.中SpecView.中Graph.Widget import (
    ColorMapSelector, DebugValue, LockYCheckBox, MainGraph,
)
from BatSpec.QtApp.中Analize.中SpecView.中Index.Logic import INDEX_STATE
from BatSpec.QtApp.中Analize.中SpecView.中Index.Widget import (
    IndexLayer, KeyboardToolController,
    RoiGroupSelector, RoiStatusLabel, RoiToolSelector,
)
from BatSpec.QtApp.中Analize.中SpecView.Logic import SPEC_VIEW_STATE, crop_spec


# ── SpecKeySelector ────────────────────────────────────────────────────────────

class SpecKeySelector(QComboBox, SPEC_VIEW_STATE.Trigger):
    """Выпадающий список версий спектрограмм из словаря SPEC_VIEW_STATE.specs."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        QComboBox.__init__(self, parent)
        self.currentTextChanged.connect(self._on_user_selection)

    def onSpecDictChanged(self) -> None:
        self.blockSignals(True)
        self.clear()
        self.addItems(list(SPEC_VIEW_STATE.specs.keys()))
        # восстанавливаем текущий ключ
        idx = self.findText(SPEC_VIEW_STATE.active_key)
        if idx >= 0:
            self.setCurrentIndex(idx)
        self.blockSignals(False)

    def onActiveKeyChanged(self) -> None:
        self.blockSignals(True)
        idx = self.findText(SPEC_VIEW_STATE.active_key)
        if idx >= 0:
            self.setCurrentIndex(idx)
        self.blockSignals(False)

    def _on_user_selection(self, text: str) -> None:
        if text and text in SPEC_VIEW_STATE.specs:
            SPEC_VIEW_STATE.setActiveKey(text)


# ── SaveROIButton ──────────────────────────────────────────────────────────────

class SaveROIButton(QPushButton, SPEC_VIEW_STATE.Trigger):
    """Кнопка сохранения ROI. Подсвечивается при наличии несохранённых изменений."""

    _STYLE_DIRTY  = "color: orange; font-weight: bold;"
    _STYLE_CLEAN  = ""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        QPushButton.__init__(self, "Save ROI", parent)
        self.clicked.connect(SPEC_VIEW_STATE.saveCSV)

    def onDirtyChanged(self, dirty: bool) -> None:
        if dirty:
            self.setText("● Save ROI")
            self.setStyleSheet(self._STYLE_DIRTY)
        else:
            self.setText("Save ROI")
            self.setStyleSheet(self._STYLE_CLEAN)


# ── LoadingOverlay ─────────────────────────────────────────────────────────────

class LoadingOverlay(QWidget, SPEC_VIEW_STATE.Trigger):
    """Полупрозрачный оверлей поверх MainGraph во время расчёта STFT."""

    def __init__(self, parent: QWidget) -> None:
        QWidget.__init__(self, parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background: rgba(0,0,0,160);")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel("⏳ Calculating spectrogram…")
        self._label.setStyleSheet("color: white; font-size: 16px;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        # Анимация точек
        self._dot_count = 0
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

        self.hide()

    def _tick(self) -> None:
        self._dot_count = (self._dot_count + 1) % 4
        self._label.setText("⏳ Calculating spectrogram" + "." * self._dot_count)

    def onLoadingChanged(self, loading: bool) -> None:
        if loading:
            self._timer.start()
            self.show()
            self.raise_()
        else:
            self._timer.stop()
            self.hide()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        if self.parent():
            self.setGeometry(self.parent().rect())  # type: ignore[union-attr]
        super().resizeEvent(event)


# ── CorpusPreviewPanel ─────────────────────────────────────────────────────────

class CorpusPreviewPanel(QWidget, INDEX_STATE.Trigger, SPEC_VIEW_STATE.Trigger):
    """
    Превью активного ROI с интегральными проекциями.

    Геометрия pg.GraphicsLayoutWidget:
        col:   0              1
        row 0  [corp_plot  ·  freq_plot (вертикальный)]
        row 1  [time_plot  ·  пусто                   ]

    corp_plot  — вырезка из спектрограммы по активному ROI
    time_plot  — интеграл по частоте → f(t)   (горизонтальный, снизу)
    freq_plot  — интеграл по времени → f(freq) (вертикальный, справа)
    """

    _AXIS_W = 42

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        QWidget.__init__(self, parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._gl = pg.GraphicsLayoutWidget()
        layout.addWidget(self._gl)

        # ── corp_plot ──────────────────────────────────────────────────────────
        self._corp_plot = self._gl.addPlot(row=0, col=0, rowspan=1, colspan=1)
        self._corp_plot.setLabel("left", "Frequency")
        self._corp_plot.setLabel("bottom", "Time")
        self._corp_img = pg.ImageItem()
        self._corp_plot.addItem(self._corp_img)

        # ── freq_plot (вертикальный, справа от corp) ───────────────────────────
        self._freq_plot = self._gl.addPlot(row=0, col=1, rowspan=1, colspan=1)
        self._freq_plot.setMaximumWidth(100)
        self._freq_plot.hideAxis("left")
        self._freq_plot.setLabel("bottom", "∫ time")
        self._freq_curve = self._freq_plot.plot(pen=pg.mkPen("y", width=1))

        # ── time_plot (горизонтальный, снизу) ─────────────────────────────────
        self._time_plot = self._gl.addPlot(row=1, col=0, rowspan=1, colspan=1)
        self._time_plot.setMaximumHeight(80)
        self._time_plot.hideAxis("left")
        self._time_plot.setLabel("bottom", "∫ freq")
        self._time_curve = self._time_plot.plot(pen=pg.mkPen("c", width=1))

        # Блокируем мышь на всех панелях (только отображение)
        for plot in (self._corp_plot, self._freq_plot, self._time_plot):
            plot.setMouseEnabled(x=False, y=False)

        self._clear()

    # ── Тригеры ───────────────────────────────────────────────────────────────

    def onChangeRoiSelection(self) -> None:
        self._rebuild()

    def onActiveKeyChanged(self) -> None:
        self._rebuild()

    def onSpecDictChanged(self) -> None:
        self._rebuild()

    # ── Внутреннее ────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        func = SPEC_VIEW_STATE.specs.get(SPEC_VIEW_STATE.active_key)
        roi  = self._get_active_roi()

        if func is None or roi is None:
            self._clear()
            return

        crop = crop_spec(func, roi)
        if crop is None:
            self._clear()
            return

        # crop shape: (freq_bins, time_bins) — транспонируем для pg.ImageItem
        self._corp_img.setImage(crop.T)

        time_integral = crop.sum(axis=0)   # sum по freq → f(t)
        freq_integral = crop.sum(axis=1)   # sum по time → f(freq)

        self._time_curve.setData(time_integral)
        self._freq_curve.setData(freq_integral, np.arange(len(freq_integral)))

    def _clear(self) -> None:
        self._corp_img.clear()
        self._time_curve.setData([], [])
        self._freq_curve.setData([], [])

    @staticmethod
    def _get_active_roi():
        idx  = INDEX_STATE.active_roi_idx
        gidx = INDEX_STATE.active_group_idx
        if idx is None or gidx < 0 or not INDEX_STATE.roi_groups:
            return None
        group = INDEX_STATE.roi_groups[gidx]
        return next((r for r in group.rois if r.idx == idx), None)


# ── SpecViewWidget ─────────────────────────────────────────────────────────────

class SpecViewWidget(QWidget):
    """
    Главный виджет панели SpecView.

    Публичные атрибуты для монтажа в родительское окно:
        self.status_bar_widget      — QWidget с DebugValue + RoiStatusLabel
        self.keyboard_controller    — KeyboardToolController (нужен installEventFilter)
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        QWidget.__init__(self, parent)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QToolBar("SpecView Controls")
        toolbar.setMovable(False)

        toolbar.addWidget(QLabel("  Spec: "))
        self._key_selector = SpecKeySelector()
        toolbar.addWidget(self._key_selector)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Colormap: "))
        self._cmap_selector = ColorMapSelector()
        toolbar.addWidget(self._cmap_selector)

        toolbar.addSeparator()
        self._lock_y = LockYCheckBox()
        toolbar.addWidget(self._lock_y)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Layer: "))
        self._group_selector = RoiGroupSelector()
        toolbar.addWidget(self._group_selector)

        self._tool_selector = RoiToolSelector()
        toolbar.addWidget(self._tool_selector)

        toolbar.addSeparator()
        self._save_btn = SaveROIButton()
        toolbar.addWidget(self._save_btn)

        root_layout.addWidget(toolbar)

        # ── Центральная область ───────────────────────────────────────────────
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        root_layout.addWidget(v_splitter)

        # ── Верхняя часть: граф + оверлей ─────────────────────────────────────
        graph_container = QWidget()
        graph_container.setMinimumHeight(200)
        graph_layout = QVBoxLayout(graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)

        self.main_graph = MainGraph()
        graph_layout.addWidget(self.main_graph)

        # IndexLayer монтируется поверх PlotItem из MainGraph
        self.index_layer = IndexLayer(
            self.main_graph.spec_plot,
            self.main_graph.view_box,
        )

        # LoadingOverlay поверх graph_container
        self._overlay = LoadingOverlay(graph_container)
        self._overlay.setGeometry(graph_container.rect())

        v_splitter.addWidget(graph_container)

        # ── Нижняя часть: превью + инструменты ────────────────────────────────
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._preview = CorpusPreviewPanel()
        h_splitter.addWidget(self._preview)

        self._corp_tool = CorpProcessMainWidget()
        h_splitter.addWidget(self._corp_tool)

        h_splitter.setSizes([400, 400])

        v_splitter.addWidget(h_splitter)
        v_splitter.setSizes([600, 300])

        # ── Status bar виджеты (монтирует родитель) ───────────────────────────
        self.debug_label  = DebugValue(self.main_graph.view_box, unit="dB")
        self.roi_status   = RoiStatusLabel()

        self.status_bar_widget = QWidget()
        sb_layout = QHBoxLayout(self.status_bar_widget)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.addWidget(self.debug_label)
        sb_layout.addSpacing(20)
        sb_layout.addWidget(self.roi_status)

        # ── Клавиатурный контроллер (монтирует caller) ────────────────────────
        self.keyboard_controller = KeyboardToolController()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        """Подгоняем оверлей под размер контейнера."""
        super().resizeEvent(event)
        parent = self._overlay.parent()
        if parent:
            self._overlay.setGeometry(parent.rect())  # type: ignore[union-attr]