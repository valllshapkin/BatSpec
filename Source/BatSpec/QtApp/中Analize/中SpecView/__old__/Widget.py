import numpy as np
import pyqtgraph as pg
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout,
    QComboBox, QRadioButton, QButtonGroup, QTabWidget,
    QCheckBox, QPushButton, QGroupBox, QSplitter
)
from PySide6.QtCore import Qt, Signal, Slot, QPointF
from PySide6.QtGui import QColor

from BatSpec.QtApp.Shared.Component import Component
from BatSpec.QtUp.TabInst import TabInstance
from BatSpec.QtApp.Shared.中LocalesSettings.Logic import Locales
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes

from BatSpec.QtUp.Image import AdaptiveImageItem
from BatSpec.QtUp.PolyROI import ROIData, ROIGroup, FastROILayer, ROIController, ROIStatusLabel, ROISync
from BatSpec.QtUp.Tools import ToolManager

# Импорты новых стейтов
from ...中Record.Logic import RECORD_STATE  
from ...中STFT.Logic import STFT_STATE

pg.setConfigOption('imageAxisOrder', 'row-major')
AXIS_LEFT_WIDTH = 42

class MinimapPlot(pg.PlotItem):
    AXIS_LEFT_WIDTH = 42
    
    def __init__(self) -> None:
        super().__init__()
        self.setMaximumHeight(60)
        self.hideAxis('bottom')
        ax = self.getAxis('left')
        ax.setWidth(self.AXIS_LEFT_WIDTH)
        ax.setStyle(showValues=False)
        ax.setTicks([])
        ax.setPen(pg.mkPen(color='w', width=0))
        self.setMouseEnabled(x=False, y=False)
        
        self._curve = self.plot(pen=pg.mkPen('c', width=1), fillLevel=0, brush=(0, 255, 255, 80))
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.addItem(self.region)

    def update_data(self, values: np.ndarray, duration: float) -> None:
        time_axis = np.linspace(0, duration, len(values))
        self._curve.setData(x=time_axis, y=values)
        self.setLimits(xMin=0, xMax=duration)
        self.setXRange(0, duration, padding=0)
        self.region.setBounds([0, duration])
        self.region.setRegion([0, duration])

class StftGraphTab(*Component, TabInstance):

    _GROUP_COLORS: dict[str, tuple[QColor, QColor]] = {
        "Signal":  (QColor(255, 50,  50,  160), QColor(255, 100, 100, 255)),
        "Context": (QColor(50,  150, 255, 160), QColor(100, 180, 255, 255)),
    }

    # ==========================================
    # --- ВЛОЖЕННЫЕ ВИДЖЕТЫ ---
    # ==========================================

    class TopToolbar(Locales.Trigger, QWidget):
        layerChanged = Signal(str)
        cmapChanged = Signal(str)
        lockYChanged = Signal(bool)

        def __init__(self):
            QWidget.__init__(self)
            self.layout = QHBoxLayout(self)
            self.layout.setContentsMargins(0, 0, 0, 0)

            self.lbl_layer = QLabel()
            self.display_combo = QComboBox()
            
            self.lbl_cmap = QLabel()
            self.cmap_combo = QComboBox()
            self.cmap_combo.addItems(pg.colormap.listMaps())
            self.cmap_combo.setCurrentText('viridis')

            self.lock_y_cb = QCheckBox()
            self.lock_y_cb.setChecked(True)

            self.lbl_pixel_info = QLabel("t=-.-- s  |  f=-.-- Hz  |  val=-.--")
            self.lbl_pixel_info.setStyleSheet("font-family: monospace; font-size: 11px; color: #aaffee; padding: 0 6px;")

            self.layout.addWidget(self.lbl_layer)
            self.layout.addWidget(self.display_combo)
            self.layout.addSpacing(10)
            self.layout.addWidget(self.lbl_cmap)
            self.layout.addWidget(self.cmap_combo)
            self.layout.addSpacing(10)
            self.layout.addWidget(self.lock_y_cb)
            self.layout.addSpacing(10)
            self.layout.addWidget(self.lbl_pixel_info)
            self.layout.addStretch()

            # Сигналы
            self.display_combo.currentTextChanged.connect(self.layerChanged.emit)
            self.cmap_combo.currentTextChanged.connect(self.cmapChanged.emit)
            self.lock_y_cb.toggled.connect(self.lockYChanged.emit)

        def update_layers(self, layers: list[str]):
            current = self.display_combo.currentText()
            self.display_combo.blockSignals(True)
            self.display_combo.clear()
            self.display_combo.addItems(layers)
            if current in layers:
                self.display_combo.setCurrentText(current)
            self.display_combo.blockSignals(False)
            self.layerChanged.emit(self.display_combo.currentText())

        def set_pixel_info(self, t: float, f: float, val: float):
            self.lbl_pixel_info.setText(f"t={t:.4f} s  |  f={f:.1f} Hz  |  val={val:.4f}")
            
        def reset_pixel_info(self):
            self.lbl_pixel_info.setText("t=-.-- s  |  f=-.-- Hz  |  val=-.--")

        def onLanguageChange(self):
            self.lbl_layer.setText(self.tr("Слой:"))
            self.lbl_cmap.setText(self.tr("Палитра:"))
            self.lock_y_cb.setText(self.tr("Lock Y"))


    class ROIPanel(Locales.Trigger, QGroupBox):
        modeChanged = Signal(str) # "Signal" or "Context"

        def __init__(self, tool_manager: ToolManager, status_label: ROIStatusLabel):
            QGroupBox.__init__(self)
            self.layout = QVBoxLayout(self)

            # Выбор режима
            self.mode_group = QButtonGroup(self)
            self.radio_sig = QRadioButton("Signal")
            self.radio_ctx = QRadioButton("Context")
            self.radio_sig.setChecked(True)
            self.mode_group.addButton(self.radio_sig, 0)
            self.mode_group.addButton(self.radio_ctx, 1)

            mode_layout = QHBoxLayout()
            mode_layout.addWidget(self.radio_sig)
            mode_layout.addWidget(self.radio_ctx)
            self.layout.addLayout(mode_layout)

            self.lbl_mode_status = QLabel("Mode: Signal")
            self.lbl_mode_status.setStyleSheet("font-weight: bold; color: yellow;")
            self.layout.addWidget(self.lbl_mode_status)
            self.layout.addWidget(status_label)

            # Инструменты
            self.btn_select = QPushButton("🖱️ Select")
            self.btn_create = QPushButton("➕ Draw")
            self.btn_delete = QPushButton("🗑️ Erase")
            
            self.tool_btns = QButtonGroup(self)
            for i, btn in enumerate([self.btn_select, self.btn_create, self.btn_delete]):
                btn.setCheckable(True)
                self.tool_btns.addButton(btn, i)
                self.layout.addWidget(btn)
            self.btn_select.setChecked(True)
            self.layout.addStretch()

            # Сигналы
            self.tool_btns.idClicked.connect(tool_manager.set_tool)
            self.mode_group.idToggled.connect(self._on_mode_toggled)

        def _on_mode_toggled(self, btn_id: int, checked: bool):
            if not checked: return
            mode = "Signal" if btn_id == 0 else "Context"
            self.lbl_mode_status.setText(f"Mode: {mode}")
            self.modeChanged.emit(mode)

        def onLanguageChange(self):
            self.setTitle(self.tr("Панель инструментов"))

    # ==========================================
    # --- ОСНОВНАЯ СБОРКА И ЛОГИКА ---
    # ==========================================

    def __init__(self, parent: QWidget | None = None, tab_widget: QTabWidget | None = None) -> None:
        TabInstance.__init__(self, parent=parent, tab_widget=tab_widget)

        # Базы данных
        self._layers_cache: dict[str, np.ndarray] = {}
        self._roi_db: dict[str, list[ROIData]] = {"Signal": [], "Context": []}
        self._roi_groups: dict[str, ROIGroup] = {}
        self._updating_region: bool = False
        self._last_selected_roi: tuple | None = None

        self.layout_main = QVBoxLayout(self)

        # 1. Верхний тулбар
        self.toolbar = self.TopToolbar()
        self.layout_main.addWidget(self.toolbar)

        self.splitter_main = QSplitter(Qt.Orientation.Vertical)
        self.layout_main.addWidget(self.splitter_main, stretch=1)

        # 2. Главный график + Миникарта
        self.w_top = QWidget()
        l_top = QVBoxLayout(self.w_top)
        l_top.setContentsMargins(0, 0, 0, 0)
        
        self.glw_top = pg.GraphicsLayoutWidget()
        l_top.addWidget(self.glw_top)

        self.vb_spec = pg.ViewBox()
        self.p_spec  = self.glw_top.addPlot(row=0, col=0, viewBox=self.vb_spec)
        self.p_spec.getAxis('left').setWidth(AXIS_LEFT_WIDTH)

        self.image_item = AdaptiveImageItem()
        self.image_item.attachTo(self.p_spec)
        self.p_spec.addItem(self.image_item)

        self.roi_layer = FastROILayer(max_x=1.0, max_y=1.0)
        self.roi_layer.setZValue(10)
        self.p_spec.addItem(self.roi_layer)

        self.p_mini = MinimapPlot()
        self.glw_top.addItem(self.p_mini, row=1, col=0)
        
        self.splitter_main.addWidget(self.w_top)

        # 3. Нижняя часть (Кроп + ROI Инструменты)
        self.w_bot = QSplitter(Qt.Orientation.Horizontal)
        
        self.glw_crop = pg.GraphicsLayoutWidget()
        self.p_crop   = self.glw_crop.addPlot()
        self.p_crop.setMouseEnabled(x=False, y=False)
        self.p_crop.hideAxis('left')
        self.p_crop.hideAxis('bottom')

        self.img_crop = pg.ImageItem()
        self.p_crop.addItem(self.img_crop)
        self.w_bot.addWidget(self.glw_crop)

        # Инициализация ROI Контроллеров
        self.controller = ROIController(self.p_spec, self.roi_layer)
        self.status_label = ROIStatusLabel(self.controller)
        
        
        self.tool_manager = ToolManager(
            plot_item  = self.p_spec,
            widget     = self.glw_top,
            controller = self.controller,
            roi_layer  = self.roi_layer,
            all_groups = list(self._roi_groups.values()),
        )
        self.tool_manager.active_tool.activate(self.p_spec, self.glw_top)
        self._build_roi_groups()

        self.panel_tools = self.ROIPanel(self.tool_manager, self.status_label)
        self.w_bot.addWidget(self.panel_tools)

        self.w_bot.setSizes([800, 220])
        self.splitter_main.addWidget(self.w_bot)
        self.splitter_main.setSizes([600, 200])

        # --- ПОДКЛЮЧЕНИЕ СИГНАЛОВ ---
        self.toolbar.layerChanged.connect(self._render_active_layer)
        self.toolbar.cmapChanged.connect(self._apply_colormap)
        self.toolbar.lockYChanged.connect(self._on_lock_y_changed)

        self.panel_tools.modeChanged.connect(self._on_roi_mode_changed)

        self.p_mini.region.sigRegionChanged.connect(self._sync_main_from_mini)
        self.p_spec.sigRangeChanged.connect(self._sync_mini_from_main)
        self.vb_spec.scene().sigMouseClicked.connect(self.tool_manager.on_click)
        self.vb_spec.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self.controller.signals.roi_selected.connect(self._on_roi_selected)
        self.controller.signals.roi_deselected.connect(self._on_roi_deselected)
        self.controller.signals.rois_changed.connect(self._save_rois_to_csv)

        # Глобальные стейты
        STFT_STATE.signals.calculationStarted.connect(self._on_stft_started)
        STFT_STATE.signals.calculationFinished.connect(self._on_calc_finished)
        RECORD_STATE.signals.recordChanged.connect(self._on_record_changed)

        self._apply_colormap('viridis')
        self.onLanguageChange()

    # --- ИНИЦИАЛИЗАЦИЯ ДАННЫХ И ROI ---

    def _build_roi_groups(self):
        for key, colors in self._GROUP_COLORS.items():
            self._roi_groups[key] = ROIGroup(
                name=key, rois=self._roi_db[key],
                base_color=colors[0], active_color=colors[1]
            )
        groups_list = list(self._roi_groups.values())
        self.controller.set_groups(groups_list)
        self.tool_manager.update_groups(groups_list)

    def _get_csv_path(self, group_name: str) -> Path | None:
        if not RECORD_STATE.path: return None
        suffix = "call.csv" if group_name == "Signal" else "context.csv"
        return RECORD_STATE.path.with_name(f"{RECORD_STATE.path.name}.{suffix}")

    @Slot()
    def _on_record_changed(self):
        """Когда меняется аудио, загружаем сохраненные ROI"""
        self._roi_db = {"Signal": [], "Context": []}
        for key in self._GROUP_COLORS.keys():
            csv_path = self._get_csv_path(key)
            if csv_path and csv_path.exists():
                self._roi_db[key] = ROISync.load_csv(csv_path)
        self._build_roi_groups()

    @Slot(str)
    def _save_rois_to_csv(self, group_name: str):
        csv_path = self._get_csv_path(group_name)
        if csv_path:
            ROISync.save_csv(self._roi_groups[group_name].rois, csv_path)

    # --- ОБРАБОТКА STFT ---

    @Slot()
    def _on_stft_started(self):
        self.tool_manager.reset_tool_state()
        self.controller.clear()
        self.img_crop.clear()
        self._last_selected_roi = None
        self.toolbar.reset_pixel_info()

    @Slot()
    def _on_calc_finished(self):
        spec = STFT_STATE.output_stft
        if not spec or not RECORD_STATE.record: return
        
        duration = RECORD_STATE.record.duration
        max_freq = RECORD_STATE.record.samplerate / 2  # Или берем из SpecFunc если там есть

        # 1. Генерируем слои "на лету" из сырой матрицы STFT
        # Если в SpecFunc уже есть db_matrix, используй её, иначе считаем тут:
        mat_amp = spec.matrix.T
        mat_db = 20 * np.log10(np.clip(mat_amp, 1e-9, None))
        
        self._layers_cache = {
            "Amplitude": mat_amp,
            "dB (Log)": mat_db
        }
        
        # 2. Обновляем UI
        self.toolbar.update_layers(list(self._layers_cache.keys()))
        self.roi_layer.resize(max_x=duration, max_y=max_freq)

        self.p_spec.setLimits(xMin=0, xMax=duration, yMin=0, yMax=max_freq)
        self.p_spec.setXRange(0, duration, padding=0)
        self.p_spec.setYRange(0, max_freq, padding=0)
        self._on_lock_y_changed(self.toolbar.lock_y_cb.isChecked())

        # Огибающая (RMS) для миникарты по оси времени
        envelope = np.sqrt(np.mean(mat_amp**2, axis=0)) 
        self.p_mini.update_data(envelope, duration)

    @Slot(str)
    def _render_active_layer(self, layer_name: str):
        if layer_name not in self._layers_cache or not RECORD_STATE.record: return
        
        arr = self._layers_cache[layer_name]
        duration = RECORD_STATE.record.duration
        max_freq = RECORD_STATE.record.samplerate / 2 

        self.image_item._levels = (float(arr.min()), float(arr.max()))
        self.image_item.setFullData(arr=arr, x_range=(0.0, duration), y_range=(0.0, max_freq))
        
        # Обновляем кроп-превью для нового слоя, если выбран ROI
        if self._last_selected_roi:
            self._update_crop_view(*self._last_selected_roi)

    # --- UI ЛОГИКА ---

    @Slot(str)
    def _on_roi_mode_changed(self, mode: str):
        idx = list(self._GROUP_COLORS.keys()).index(mode)
        self.controller.switch_group(idx)

    @Slot(str)
    def _apply_colormap(self, name: str):
        try: cmap = pg.colormap.get(name)
        except Exception: cmap = pg.colormap.get('viridis')
        self.image_item.setColorMap(cmap)
        self.img_crop.setColorMap(cmap)
        self.image_item._doUpdate()

    @Slot(bool)
    def _on_lock_y_changed(self, locked: bool):
        self.p_spec.setMouseEnabled(x=True, y=not locked)
        if locked and RECORD_STATE.record:
            self.p_spec.setYRange(0, RECORD_STATE.record.samplerate / 2, padding=0)

    # --- СИНХРОНИЗАЦИЯ И ROI ПРЬЕВЮ ---

    def _sync_main_from_mini(self):
        if self._updating_region: return
        self._updating_region = True
        self.p_spec.setXRange(*self.p_mini.region.getRegion(), padding=0)
        self._updating_region = False

    def _sync_mini_from_main(self):
        if self._updating_region: return
        self._updating_region = True
        self.p_mini.region.setRegion(self.p_spec.viewRange()[0])
        self._updating_region = False

    @Slot(str, int, float, float, float, float)
    def _on_roi_selected(self, _g: str, idx: int, x: float, y: float, w: float, h: float):
        self._last_selected_roi = (x, y, w, h)
        self._update_crop_view(x, y, w, h)

    @Slot(str)
    def _on_roi_deselected(self, _g: str):
        self._last_selected_roi = None
        self.img_crop.clear()

    def _update_crop_view(self, x: float, y: float, w: float, h: float):
        layer_name = self.toolbar.display_combo.currentText()
        if layer_name not in self._layers_cache or not RECORD_STATE.record: return
        
        arr = self._layers_cache[layer_name]
        duration = RECORD_STATE.record.duration
        max_freq = RECORD_STATE.record.samplerate / 2

        F, T = arr.shape
        t_scale = T / duration
        f_scale = F / max_freq

        t_start = max(0, int(x * t_scale))
        t_end   = min(T, int((x + w) * t_scale))
        f_start = max(0, int(y * f_scale))
        f_end   = min(F, int((y + h) * f_scale))

        sl = arr[f_start:f_end, t_start:t_end]
        if sl.size > 0:
            self.img_crop.setImage(sl, autoLevels=True)
            self.p_crop.setXRange(0, sl.shape[1], padding=0)
            self.p_crop.setYRange(0, sl.shape[0], padding=0)

    @Slot(QPointF)
    def _on_mouse_moved(self, pos: QPointF):
        if not RECORD_STATE.record or not self.vb_spec.sceneBoundingRect().contains(pos): 
            return

        mouse_point = self.vb_spec.mapSceneToView(pos)
        t, f = mouse_point.x(), mouse_point.y()
        
        duration = RECORD_STATE.record.duration
        max_freq = RECORD_STATE.record.samplerate / 2

        if not (0 <= t <= duration and 0 <= f <= max_freq):
            self.toolbar.reset_pixel_info()
            return

        layer_name = self.toolbar.display_combo.currentText()
        if layer_name not in self._layers_cache: return
        
        arr = self._layers_cache[layer_name]
        F, T = arr.shape
        t_idx = max(0, min(T - 1, int(t / duration * T)))
        f_idx = max(0, min(F - 1, int(f / max_freq * F)))

        val = arr[f_idx, t_idx]
        self.toolbar.set_pixel_info(t, f, val)

    def onLanguageChange(self) -> None:
        self.setTabName(self.tr("Спектрограмма & Разметка"))
        self.p_spec.setLabel('left', self.tr("Частота"), units='Hz')
        self.p_spec.setLabel('bottom', self.tr("Время"), units='s')

    def onThemeChange(self) -> None: pass