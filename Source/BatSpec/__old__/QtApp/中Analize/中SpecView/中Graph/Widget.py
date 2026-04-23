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

AXIS_LEFT_WIDTH = 32

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

AXIS_LEFT_WIDTH = 32

class MinimapPlot(Themes.Trigger, GRAPH_STATE.TriggerMinimap, pg.PlotItem):
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
            self.setXRange(0, 1, padding=0)
            
            # Блокируем сигналы, чтобы не запустить Syncer с пустыми данными
            self.region.blockSignals(True)
            self.region.setBounds([0, 1])
            self.region.setRegion([0, 1])
            self.region.blockSignals(False)
            return
        
        func = GRAPH_STATE.time_func
        # Переводим в float на случай если приходят numpy скаляры
        t0, t1 = float(func.time_axis[0]), float(func.time_axis[-1])
        
        self._curve.setData(x=func.time_axis, y=func.data)
        
        # 1. СНАЧАЛА сбрасываем лимиты, чтобы они не конфликтовали с новым диапазоном
        self.setLimits(xMin=None, xMax=None)
        self.setXRange(t0, t1, padding=0)
        
        # 2. Двигаем регион "тихо", чтобы не спровоцировать Syncer сдвинуть MainGraph раньше времени
        self.region.blockSignals(True)
        self.region.setBounds([t0, t1])
        self.region.setRegion([t0, t1])
        self.region.blockSignals(False)
        
        # 3. ПОСЛЕ установки обзора возвращаем новые лимиты
        self.setLimits(xMin=t0, xMax=t1)

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
    def __init__(self):
        QWidget.__init__(self)
        self._data_loaded = False  

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.main_layout.addWidget(self.graphics_layout)

        self.view_box = pg.ViewBox()
        self.spec_plot = self.graphics_layout.addPlot(row=0, col=0, viewBox=self.view_box)
        self.spec_plot.getAxis('left').setWidth(AXIS_LEFT_WIDTH)

        self.spec_plot.setLabel('bottom', "Time", units="s")
        self.spec_plot.setLabel('left', "Frequency", units="Hz")

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
            self._data_loaded = False 
            return
        
        func = GRAPH_STATE.spec_func
        t0, t1 = float(func.time[0]), float(func.time[-1])
        f0, f1 = float(func.freq[0]), float(func.freq[-1])
        
        self.image_item.setFullData(func.matrix.T, (t0, t1), (f0, f1))

        # 1. ОБЯЗАТЕЛЬНО отключаем лимиты перед применением новых X/Y Range
        # Иначе pyqtgraph жестко сожмет обзор (clamp), если старый зум выходит за новые границы
        self.spec_plot.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)

        if not self._data_loaded:
            self.spec_plot.setXRange(t0, t1, padding=0)
            if GRAPH_STATE.lock_y:
                self.spec_plot.setYRange(f0, f1, padding=0)
            self._data_loaded = True
        else:
            # Если пользователь загрузил новый файл, чье время вообще не пересекается
            # со старым, старый зум оставит его смотреть в пустоту. Проверяем это:
            current_x = self.spec_plot.viewRange()[0]
            if current_x[1] <= t0 or current_x[0] >= t1:
                # Зум полностью за пределами новых данных -> сбрасываем на полный обзор
                self.spec_plot.setXRange(t0, t1, padding=0)
                
            if GRAPH_STATE.lock_y:
                self.spec_plot.setYRange(f0, f1, padding=0)

        # 2. ПОСЛЕ того как виджет сцентрирован на правильных координатах,
        # возвращаем ограничители (Limits), чтобы нельзя было улететь за границы.
        if func.duration > 1:
            self.spec_plot.setLimits(xMin=t0, xMax=t1, yMin=f0, yMax=f1)
        else:
            self.spec_plot.setLimits(xMin=None, xMax=None, yMin=f0, yMax=f1)

    # Остальные методы (onChangeColorMap, onChangeLockY и т.д.) без изменений
    def onChangeColorMap(self):
        cm = GRAPH_STATE.colormap
        if not cm: raise RuntimeError()
        self.image_item.setColorMap(cm)
        self.image_item._doUpdate()

    def onChangeLockY(self):
        locked = GRAPH_STATE.lock_y
        self.spec_plot.setMouseEnabled(x=True, y=not locked)
        if locked and GRAPH_STATE.spec_func:
            f0, f1 = (float(GRAPH_STATE.spec_func.freq[0]), float(GRAPH_STATE.spec_func.freq[-1]))
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



