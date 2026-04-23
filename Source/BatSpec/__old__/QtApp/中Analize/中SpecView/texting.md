```python path="Source/BatSpec/QtApp/中Analize/中SpecView/中Graph/Logic.py"
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


```

```python path="Source/BatSpec/QtApp/中Analize/中SpecView/中Graph/Widget.py"
from typing import Optional

from PySide6.QtWidgets import QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from PySide6.QtCore import QEvent, QObject, Qt, Signal, Slot, QPointF
from PySide6 import QtGui
import numpy as np
import pyqtgraph as pg

from BatSpec.Logic.Functions import TimeFunc, SpecFunc
from BatSpec.QtApp.Shared.中ThemeSettings.Logic import Themes
from BatSpec.QtUp.Image import AdaptiveImageItem

from .Logic import GRAPH_STATE

AXIS_LEFT_WIDTH = 42

class MinimapPlot(Themes.Trigger, GRAPH_STATE.TriggerMinimap, pg.PlotItem):
    """
    UI Hierarchy:
    MinimapPlot (pg.PlotItem)
    ├── self._curve : pg.PlotDataItem     (Main graph line and fill)
    └── self.region : pg.LinearRegionItem (Draggable selection bounds)
    """
    
    def __init__(self) -> None:
        pg.PlotItem.__init__(self)
        self.setMaximumHeight(60)
        self.hideAxis('bottom')
        ax = self.getAxis('left')
        ax.setWidth(AXIS_LEFT_WIDTH)
        ax.setStyle(showValues=False)
        ax.setTicks([])
        ax.setPen(pg.mkPen(color='w', width=0))
        self.setMouseEnabled(x=False, y=False)
        
        self._curve = self.plot(pen=pg.mkPen('c', width=1), fillLevel=0, brush=(0, 255, 255, 80))
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.addItem(self.region)

    def onChangeTimeFunc(self):
        if GRAPH_STATE.time_func is None:
            self._curve.setData([], [])
            self.setLimits(xMin=None, xMax=None)
            self.setXRange(0, 1, padding=0)  # временный диапазон
            self.region.setBounds([0, 1])    # фиктивные границы
            self.region.setRegion([0, 1])
            return
        
        func = GRAPH_STATE.time_func
        self._curve.setData(x=func.time_axis, y=func.data)
        self.setLimits(xMin=0, xMax=func.duration)
        self.setXRange(0, func.duration, padding=0)
        self.region.setBounds([0, func.duration])
        self.region.setRegion([0, func.duration])

    def onThemeChange(self):
        self._curve.setPen(pg.mkPen(Themes.get_curent_theme().primary, width=1))
        color_brush = QtGui.QColor(Themes.get_curent_theme().primary)
        color_brush.setAlpha(color_brush.alpha() // 2)
        self._curve.setBrush(pg.mkBrush(color=color_brush))


class Syncer:
    def __init__(self, minimap: MinimapPlot, graph: pg.PlotItem):
        self._minimap = minimap
        self._graph = graph
        self._updating_region = False

        self._minimap.region.sigRegionChanged.connect(self._sync_main_from_mini)
        self._graph.sigRangeChanged.connect(self._sync_mini_from_main)

    def _sync_main_from_mini(self):
        if self._updating_region: return
        self._updating_region = True
        self._graph.setXRange(*self._minimap.region.getRegion(), padding=0)
        self._updating_region = False

    def _sync_mini_from_main(self):
        if self._updating_region: return
        self._updating_region = True
        self._minimap.region.setRegion(self._graph.viewRange()[0])
        self._updating_region = False


class MainGraph(GRAPH_STATE.TriggerGraph, QWidget):
    """
    UI Hierarchy:
    MainGraph (QWidget)
    └── self.main_layout : QVBoxLayout
        └── self.graphics_layout : pg.GraphicsLayoutWidget
            │
            ├── [Row 0] self.spec_plot : pg.PlotItem  <-- uses self.view_box : pg.ViewBox
            │   ├── self.image_item : AdaptiveImageItem
            │   └── self.roi_layer : FastROILayer
            │
            └── [Row 1] self.minimap : MinimapPlot
    """

    def __init__(self):
        QWidget.__init__(self)
        

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.main_layout.addWidget(self.graphics_layout)

        self.view_box = pg.ViewBox()
        self.spec_plot = self.graphics_layout.addPlot(row=0, col=0, viewBox=self.view_box)
        self.spec_plot.getAxis('left').setWidth(AXIS_LEFT_WIDTH)

        self.image_item = AdaptiveImageItem()
        self.image_item.attachTo(self.spec_plot)
        self.spec_plot.addItem(self.image_item)

        self.minimap = MinimapPlot()
        self.graphics_layout.addItem(self.minimap, row=1, col=0)

        self.syncer = Syncer(self.minimap, self.spec_plot)
       
    def onChangeSpecFunc(self):
        if GRAPH_STATE.spec_func is None:
            self.image_item.clear()

            self.spec_plot.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
            self.spec_plot.setXRange(0, 1, padding=0)
            self.spec_plot.setYRange(0, 1, padding=0)
            return
        
        func = GRAPH_STATE.spec_func
        t0, t1 = (func.time[0], func.time[-1])
        f0, f1 = (func.freq[0], func.freq[-1])
        
        self.image_item.setFullData(func.matrix.T, (t0, t1), (f0, f1))

        self.spec_plot.setLimits(xMin=t0, xMax=t1, yMin=f0, yMax=f1)
        self.spec_plot.setXRange(t0, t1, padding=0)
        self.spec_plot.setYRange(f0, f1, padding=0)

    def onChangeColorMap(self):
        cm = GRAPH_STATE.colormap
        if not cm: raise RuntimeError()
        self.image_item.setColorMap(cm)
        self.image_item._doUpdate()

    def onChangeLockY(self):
        locked = GRAPH_STATE.lock_y
        self.spec_plot.setMouseEnabled(x=True, y=not locked)
        if locked and GRAPH_STATE.spec_func:
            f0, f1 = (GRAPH_STATE.spec_func.freq[0], GRAPH_STATE.spec_func.freq[-1])
            self.spec_plot.setYRange(f0, f1, padding=0)

class LockYCheckBox(QCheckBox, GRAPH_STATE.TriggerGraph):
    """
    Чекбокс для управления блокировкой оси Y.
    Синхронизируется с GRAPH_STATE через TriggerGraph.
    """
    def __init__(self, parent=None):
        QCheckBox.__init__(self, "Lock Y Axis", parent)
        
        # Подписываемся на клики пользователя
        self.toggled.connect(self._on_toggled)

    def onChangeLockY(self):
        """Вызывается автоматически при изменении GRAPH_STATE.lock_y"""
        # Блокируем сигналы, чтобы программное изменение не вызывало _on_toggled по кругу
        self.blockSignals(True)
        self.setChecked(GRAPH_STATE.lock_y)
        self.blockSignals(False)

    def _on_toggled(self, checked: bool):
        """Передаем состояние из UI в глобальный стейт"""
        GRAPH_STATE.setLockY(checked)


class DebugValue(QLabel, GRAPH_STATE.TriggerGraph):
    """
    QLabel, отображающая координаты (t, f) и значение спектрограммы под курсором мыши.
    Наследует TriggerGraph для автоматического обновления при изменении spec_func.
    """

    def __init__(self, view_box: pg.ViewBox, unit: str = "", parent=None):
        QLabel.__init__(self, parent)
        self._view_box = view_box
        self._unit = unit  # Сохраняем единицу измерения (например, "dB", "V", или "")
        self._matrix: Optional[np.ndarray] = None   
        self._t_min = self._t_max = 0.0
        self._f_min = self._f_max = 0.0

        # Подписываемся на движение мыши в сцене ViewBox
        scene = self._view_box.scene()
        if scene is not None:
            scene.sigMouseMoved.connect(self._on_mouse_moved)

        self.setText("No spec data")
        self.setMinimumWidth(300)
     
    def onChangeSpecFunc(self):
        """Обновляет внутренние ссылки на данные спектрограммы."""
        spec_func = GRAPH_STATE.spec_func
        if spec_func is not None and spec_func.matrix is not None:
            self._matrix = spec_func.matrix.T 
            self._t_min = float(spec_func.time[0])
            self._t_max = float(spec_func.time[-1])
            self._f_min = float(spec_func.freq[0])
            self._f_max = float(spec_func.freq[-1])
            self.setText("Ready")
        else:
            self._matrix = None
            self.setText("No spec data")

    @Slot(QPointF)
    def _on_mouse_moved(self, scene_pos: QPointF):
        """Обрабатывает перемещение мыши: вычисляет t, f, значение и обновляет текст."""
        if self._matrix is None:
            return

        if not self._view_box.sceneBoundingRect().contains(scene_pos):
            self.setText("Outside graph area")
            return

        view_pos = self._view_box.mapSceneToView(scene_pos)
        t = view_pos.x()
        f = view_pos.y()

        if not (self._t_min <= t <= self._t_max and self._f_min <= f <= self._f_max):
            self.setText(f"t={t:.3f} s, f={f:.1f} Hz  [outside bounds]")
            return

        time_bins = self._matrix.shape[1]
        freq_bins = self._matrix.shape[0]
        t_idx = int(np.clip((t - self._t_min) / (self._t_max - self._t_min) * time_bins, 0, time_bins - 1))
        f_idx = int(np.clip((f - self._f_min) / (self._f_max - self._f_min) * freq_bins, 0, freq_bins - 1))
        value = float(self._matrix[f_idx, t_idx])

        # Подставляем единицу измерения, если она передана
        unit_str = f" {self._unit}" if self._unit else ""
        self.setText(f"t = {t:.3f} s | f = {f:.1f} Hz | value = {value:.2f}{unit_str}")


class ColorMapSelector(QComboBox, GRAPH_STATE.TriggerGraph):
    """
    Выпадающий список для выбора цветовой карты (Colormap).
    Синхронизируется с GRAPH_STATE через TriggerGraph.
    """
    def __init__(self, parent=None):
        QComboBox.__init__(self, parent)
        
        # Получаем список доступных цветовых карт из pyqtgraph
        maps = pg.colormap.listMaps()
        self.addItems(maps)
        
        # Подписываемся на выбор пользователя
        self.currentTextChanged.connect(self._on_user_selection)

    def onChangeColorMap(self):
        """Вызывается автоматически при изменении GRAPH_STATE.colormap"""
        current_map = GRAPH_STATE.settings.colormap
        
        # Блокируем сигналы, чтобы не зациклить вызовы при программном изменении
        self.blockSignals(True)
        index = self.findText(current_map)
        if index >= 0:
            self.setCurrentIndex(index)
        self.blockSignals(False)

    def _on_user_selection(self, text: str):
        """Передаем новую палитру в глобальный стейт"""
        GRAPH_STATE.setColorMap(text)



```

```python path="Source/BatSpec/QtApp/中Analize/中SpecView/中Graph/__test__/2.py"
import numpy as np
from scipy.signal.windows import hann

from BatSpec.QtApp.Logic import application
from BatSpec.Logic.Loader import loadRecord
from BatSpec.Logic.ConvWindow import Window
from BatSpec.Logic.TransitionWorker import makeSpec, etalonNoise, integrateFreq
from BatSpec.Logic.SoundWorker import correctDC
from BatSpec.Logic.SpecWorker import subEtalonNoise

from PySide6.QtWidgets import QMainWindow, QToolBar, QLabel

from pathlib import Path
ScriptDir = Path(__file__).parent

with application():
    from BatSpec.QtApp.中Analize.中SpecView.中Graph.Widget import (
        MainGraph, DebugValue, LockYCheckBox, ColorMapSelector
    )
    from BatSpec.QtApp.中Analize.中SpecView.中Graph.Logic import GRAPH_STATE

    # --- Загрузка и подготовка данных ---
    spec = makeSpec(
        correctDC(loadRecord(ScriptDir / "MYODAS_20230624_005134.wav")),
        Window(hann, 0.001, norm="energy"),
        overlap=0.5,
        bins=200
    )
    noise = etalonNoise(spec, percentile=5)
    
    # Переводим в децибелы
    spec_db = spec.cloneApply(lambda arr: 20 * np.log10(np.clip(arr, 1e-9, None)))

    # --- Создание основных виджетов ---
    main_graph = MainGraph()
    GRAPH_STATE.setMinimapFunc(integrateFreq(subEtalonNoise(spec, noise)))
    GRAPH_STATE.setSpecFunc(spec_db)
    
    # --- Инициализация дополнительных элементов UI ---
    debug_label = DebugValue(main_graph.view_box, unit="dB")
    lock_checkbox = LockYCheckBox()
    cmap_selector = ColorMapSelector()

    # --- Настройка главного окна ---
    window = QMainWindow()
    window.setCentralWidget(main_graph)
    
    # 1. Создаем панель инструментов (ToolBar) сверху
    toolbar = QToolBar("Graph Controls")
    toolbar.addWidget(QLabel("  Colormap: "))
    toolbar.addWidget(cmap_selector)
    toolbar.addSeparator()
    toolbar.addWidget(lock_checkbox)
    window.addToolBar(toolbar)

    # 2. Дебаг-значения оставляем в строке состояния снизу
    window.statusBar().addWidget(debug_label)   
    
    window.setWindowTitle("Spectrogram Viewer")
    window.resize(1200, 800)
    
    # Устанавливаем начальные значения (UI обновится автоматически благодаря TriggerGraph)
    # GRAPH_STATE.setLockY(False)
    # GRAPH_STATE.setColorMap("viridis") # Можно задать любую дефолтную
    
    window.show()
```

```python path="Source/BatSpec/QtApp/中Analize/中SpecView/中Index/Logic.py"
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


```

```python path="Source/BatSpec/QtApp/中Analize/中SpecView/中Index/Widget.py"
from PySide6.QtWidgets import QButtonGroup, QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget
from PySide6.QtCore import QEvent, QObject, Qt
import pyqtgraph as pg

from BatSpec.QtApp.中Analize.中SpecView.中Index.Logic import INDEX_STATE
from BatSpec.QtUp.PolyROI import FastROILayer,  ROIController, ROIData

from PySide6.QtWidgets import QGraphicsRectItem
from PySide6.QtCore import Qt
import pyqtgraph as pg

CURSORS = {"select": Qt.CursorShape.ArrowCursor, "create": Qt.CursorShape.CrossCursor, "erase": Qt.CursorShape.ForbiddenCursor}

class IndexLayer(INDEX_STATE.Trigger):
    
    def __init__(self, spec_plot, view_box) -> None:
        self.spec_plot = spec_plot
        self.view_box = view_box
        self.roi_layer = FastROILayer(max_x=1.0, max_y=1.0)
        self.roi_layer.setZValue(10)
        spec_plot.addItem(self.roi_layer)

        self.roi_controller = ROIController(self.spec_plot, self.roi_layer)
        self.roi_creator = ROICreatorTool(self.spec_plot, self._on_roi_created)
        self.spec_plot.addItem(self.roi_creator)
        self.spec_plot.scene().sigMouseClicked.connect(self._on_scene_clicked)
                
        self.roi_controller.signals.roi_selected.connect(
            lambda name, idx, x, y, w, h: INDEX_STATE.setActiveRoi(idx)
        )
        self.roi_controller.signals.roi_deselected.connect(
            lambda name: INDEX_STATE.setActiveRoi(None)
        )


    def _on_roi_created(self, x: float, y: float, w: float, h: float):
        group = INDEX_STATE.roi_groups[INDEX_STATE.active_group_idx]
        new_idx = max([r.idx for r in group.rois], default=-1) + 1
        new_roi = ROIData(new_idx, x, y, w, h)
        INDEX_STATE.addRoiToCurrentGroup(new_roi) # Стейт сам разошлет обновления


    def _on_scene_clicked(self, ev):
        mode = INDEX_STATE.tool_mode
        if mode == "select":
            self.roi_controller.on_click(ev)
        elif mode == "erase" and ev.button() == Qt.MouseButton.LeftButton:
            self._erase_roi_at(ev.scenePos())


    def _erase_roi_at(self, scene_pos):
        if not INDEX_STATE.roi_groups or INDEX_STATE.active_group_idx < 0: return
        pos = self.view_box.mapSceneToView(scene_pos)
        b = int(pos.x()) // self.roi_layer.BUCKET_W
        
        # Ищем, куда попали
        rois = self.roi_layer._buckets.get(b, [])
        hit = next((r for r in reversed(rois) if r.contains(pos.x(), pos.y())), None)
        if hit:
            INDEX_STATE.removeRoiFromCurrentGroup(hit) # Стейт разошлет обновления

    def onChangeSize(self):
        if INDEX_STATE.size is None: return
        self.roi_layer.resize(max_x=INDEX_STATE.size[0], max_y=INDEX_STATE.size[1])

    def onChangeToolMode(self):
        mode = INDEX_STATE.tool_mode
        
        # ИСПРАВЛЕНИЕ: Вместо abort() вызываем finish(), чтобы сохранить рамку
        if mode != "create" and hasattr(self, 'roi_creator'):
            self.roi_creator.finish()

        self.roi_creator.active = (mode == "create")
        self.spec_plot.setCursor(CURSORS.get(mode, Qt.CursorShape.ArrowCursor))
        
        if mode != "select":
            self.roi_controller.deselect()

    def onChangeRoiGroups(self):
        # Обновляем слои в контроллере
        self.roi_controller.set_groups(INDEX_STATE.roi_groups)
        if INDEX_STATE.active_group_idx >= 0:
            self.roi_controller.switch_group(INDEX_STATE.active_group_idx)

    def onChangeRoiSelection(self):
        # Если стейт говорит, что выделения нет, снимаем его в контроллере
        if INDEX_STATE.active_roi_idx is None:
            if self.roi_controller.active_idx is not None:
                self.roi_controller.deselect()

        # Добавляем обработчик смены группы
    def onChangeActiveGroup(self):
        if INDEX_STATE.active_group_idx >= 0:
            self.roi_controller.switch_group(INDEX_STATE.active_group_idx)


class ROICreatorTool(pg.GraphicsObject):
    def __init__(self, plot_item: pg.PlotItem, on_create_callback) -> None:
        super().__init__()
        self.pi = plot_item
        self.on_create = on_create_callback
        self.active = False
        self.drawing = False  # НОВЫЙ ФЛАГ
        self.start_pos = None

        self.draw_rect = QGraphicsRectItem()
        self.draw_rect.setPen(pg.mkPen('y', width=2, style=Qt.PenStyle.DashLine))
        self.pi.addItem(self.draw_rect)
        self.draw_rect.hide()

    def boundingRect(self):
        return self.pi.vb.viewRect()

    def paint(self, *args): pass

    def finish(self):
        """ИСПРАВЛЕНИЕ: Принудительно завершает и сохраняет рисование (например, при отпускании Shift)"""
        if self.drawing:
            self.drawing = False
            self.draw_rect.hide()
            rect = self.draw_rect.rect()
            w, h = rect.width(), rect.height()
            if w > 1e-5 and h > 1e-5:
                self.on_create(rect.x(), rect.y(), w, h)

    def abort(self):
        """Принудительно отменяет рисование (если сменился инструмент)"""
        self.drawing = False
        self.draw_rect.hide()


    def mouseDragEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        
        if not self.active and not self.drawing:
            ev.ignore()
            return
        
        ev.accept()
        
        # ИСПРАВЛЕНИЕ ЛАГА: Берем точные координаты самого ПЕРВОГО клика!
        start_scene_pos = ev.buttonDownScenePos()
        self.start_pos = self.pi.vb.mapSceneToView(start_scene_pos)
        
        # Текущая позиция мыши
        pos = self.pi.vb.mapSceneToView(ev.scenePos())
        
        if ev.isStart() and self.active:
            self.drawing = True
            self.draw_rect.setRect(self.start_pos.x(), self.start_pos.y(), 0, 0)
            self.draw_rect.show()
            
        elif ev.isFinish() and self.drawing:
            self.finish() # Вызываем функцию завершения
                
        elif self.drawing:
            x = min(self.start_pos.x(), pos.x())
            y = min(self.start_pos.y(), pos.y())
            w = abs(self.start_pos.x() - pos.x())
            h = abs(self.start_pos.y() - pos.y())
            self.draw_rect.setRect(x, y, w, h)


class KeyboardToolController(QObject):
    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            # ИСПРАВЛЕНИЕ: Добавляем удаление на клавишу Delete
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Delete:
                INDEX_STATE.deleteActiveRoi()
                return False # Не блокируем ивент, просто обрабатываем

            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ShiftModifier:
                INDEX_STATE.setToolMode("create")
            elif mods & Qt.KeyboardModifier.AltModifier:
                INDEX_STATE.setToolMode("erase")
            else:
                INDEX_STATE.setToolMode("select")
        return False



class RoiToolSelector(QWidget, INDEX_STATE.Trigger):
    """(ВАРИАНТ 1) Классические кнопки для выбора инструмента"""
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_group = QButtonGroup(self)
        self.btns = {
            "select": QPushButton("🖱️ Select"),
            "create": QPushButton("➕ Draw"),
            "erase": QPushButton("🗑️ Erase")
        }

        for i, (mode, btn) in enumerate(self.btns.items()):
            btn.setCheckable(True)
            self.btn_group.addButton(btn, i)
            layout.addWidget(btn)

        # Связываем кнопки -> Стейт
        self.btn_group.idClicked.connect(self._on_btn_clicked)

    def _on_btn_clicked(self, btn_id):
        modes = ["select", "create", "erase"]
        INDEX_STATE.setToolMode(modes[btn_id])

    def onChangeToolMode(self):
        """Стейт -> Кнопки (если стейт изменился горячей клавишей)"""
        mode = INDEX_STATE.tool_mode
        self.btns[mode].setChecked(True)


class RoiGroupSelector(QComboBox, INDEX_STATE.Trigger):
    """Выпадающий список для переключения между слоями (группами) ROI"""
    def __init__(self, parent=None):
        QComboBox.__init__(self, parent)
        self.currentIndexChanged.connect(self._on_index_changed)

    def _on_index_changed(self, idx):
        if idx >= 0:
            INDEX_STATE.setActiveGroup(idx)

    def onChangeRoiGroups(self):
        # Если список групп изменился (например, загрузили новый файл)
        self.blockSignals(True)
        self.clear()
        self.addItems([g.name for g in INDEX_STATE.roi_groups])
        if INDEX_STATE.active_group_idx >= 0:
            self.setCurrentIndex(INDEX_STATE.active_group_idx)
        self.blockSignals(False)

    def onChangeActiveGroup(self):
        # Если группа изменилась программно (или из другого виджета)
        self.blockSignals(True)
        if INDEX_STATE.active_group_idx >= 0:
            self.setCurrentIndex(INDEX_STATE.active_group_idx)
        self.blockSignals(False)


class RoiStatusLabel(QLabel, INDEX_STATE.Trigger):
    """Показывает текущий выделенный ROI с деталями"""
    def __init__(self, parent=None):
        QLabel.__init__(self, "Нет выделения", parent)
        self.setMinimumWidth(300) # Сделаем пошире для текста

    def onChangeRoiSelection(self):
        idx = INDEX_STATE.active_roi_idx
        # ИСПРАВЛЕНИЕ: Достаем полные данные о ROI
        if idx is None or INDEX_STATE.active_group_idx < 0:
            self.setText("Нет выделения")
        else:
            group = INDEX_STATE.roi_groups[INDEX_STATE.active_group_idx]
            roi = next((r for r in group.rois if r.idx == idx), None)
            if roi:
                self.setText(f"ROI #{idx} | Pos: X={roi.x:.1f}, Y={roi.y:.1f} | Size: W={roi.w:.1f}, H={roi.h:.1f}")
            else:
                self.setText(f"Выделен ROI #{idx}")

```

```python path="Source/BatSpec/QtApp/中Analize/中SpecView/中Index/__test__/1.py"
import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QLabel

# --- Импорты ваших классов ---
# Замените пути на актуальные, если они отличаются
from BatSpec.QtApp.中Analize.中SpecView.中Index.Logic import INDEX_STATE
from BatSpec.QtApp.中Analize.中SpecView.中Index.Widget import (
    IndexLayer, RoiToolSelector, RoiGroupSelector, 
    RoiStatusLabel, KeyboardToolController
)
from BatSpec.QtUp.PolyROI import ROIGroup, ROIData


def create_mock_roi_groups() -> list[ROIGroup]:
    """Создаем 2 тестовые группы ROI для координат от 0 до 100"""
    # Группа 1
    g1_rois = [
        ROIData(0, 10, 10, 20, 20),
        ROIData(1, 60, 40, 15, 30)
    ]
    # Группа 2
    g2_rois = [
        ROIData(0, 30, 70, 40, 15)
    ]
    
    return [
        ROIGroup("Группа А (Зеленые)", g1_rois, pg.mkColor('g'), pg.mkColor('y')),
        ROIGroup("Группа Б (Красные)", g2_rois, pg.mkColor('r'), pg.mkColor('w'))
    ]


class TestWindow(QMainWindow, INDEX_STATE.Trigger):
    """
    Главное окно для тестирования слоя Index.
    Наследуем Trigger, чтобы управлять блокировкой ViewBox при рисовании.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Тестирование независимого IndexLayer")
        self.resize(800, 600)

        # 1. Создаем базовые элементы PyQtGraph
        self.plot_widget = pg.PlotWidget(background='#111')
        self.setCentralWidget(self.plot_widget)

        self.plot_item = self.plot_widget.getPlotItem()
        self.view_box = self.plot_item.getViewBox()

        # 2. Создаем фейковый фон (чтобы видеть размеры сетки: 100x100)
        np.random.seed(42)
        fake_image_data = np.random.normal(size=(100, 100))
        self.img_item = pg.ImageItem(fake_image_data)
        self.plot_item.addItem(self.img_item)
        self.plot_item.setLimits(xMin=-50, xMax=150, yMin=-50, yMax=150)
        self.plot_item.setXRange(0, 100, padding=0)
        self.plot_item.setYRange(0, 100, padding=0)

        # 3. Настраиваем глобальный Стейт (размер поля и тестовые данные)
        INDEX_STATE.setSize((100, 100))
        INDEX_STATE.setRoiGroups(create_mock_roi_groups())

        # 4. Инициализируем ваш изолированный слой, передав базовые типы
        self.index_layer = IndexLayer(self.plot_item, self.view_box)

        # 5. Инициализируем UI элементы
        self.tool_selector = RoiToolSelector()
        self.group_selector = RoiGroupSelector()
        self.status_label = RoiStatusLabel()

        # Настраиваем ToolBar
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        toolbar.addWidget(QLabel(" Инструмент: "))
        toolbar.addWidget(self.tool_selector)
        toolbar.addSeparator()
        
        toolbar.addWidget(QLabel(" Слой ROI: "))
        toolbar.addWidget(self.group_selector)
        toolbar.addSeparator()
        
        toolbar.addWidget(QLabel(" Подсказка: зажми [Shift] для рисования, [Alt] для стирания"))

        # Настраиваем StatusBar
        self.statusBar().addWidget(self.status_label)

    def onChangeToolMode(self):
        """
        Так как вы закомментировали setMouseEnabled в IndexLayer,
        мы управляем блокировкой графика (чтобы не тянулся фон при рисовании) здесь.
        """
        is_select = (INDEX_STATE.tool_mode == "select")
        self.view_box.setMouseEnabled(x=is_select, y=is_select)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Регистрация перехватчика горячих клавиш (Shift/Alt) на всё приложение
    kbd_controller = KeyboardToolController()
    app.installEventFilter(kbd_controller)

    win = TestWindow()
    win.show()
    
    sys.exit(app.exec())
```