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
