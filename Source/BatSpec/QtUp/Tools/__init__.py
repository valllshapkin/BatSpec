from __future__ import annotations

from dataclasses import dataclass
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QObject, Signal, QPointF
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QWidget, QGraphicsRectItem

from BatSpec.QtUp.PolyROI import ROIData, ROIGroup, FastROILayer, ROIController


@dataclass
class ToolContext:
    all_groups:   list[ROIGroup]
    roi_layer:    FastROILayer
    controller:   ROIController

    @property
    def active_group(self) -> ROIGroup | None:
        return self.controller.active_group


class BaseTool:
    cursor: Qt.CursorShape = Qt.CursorShape.ArrowCursor

    def on_click(self, ev: object, ctx: ToolContext) -> None: pass
    def on_drag(self,  ev: object, ctx: ToolContext) -> None: pass

    def activate(self, plot_item: pg.PlotItem, widget: QWidget) -> None:
        widget.setCursor(QCursor(self.cursor))
        plot_item.getViewBox().setMouseEnabled(x=False, y=False)

    def deactivate(self) -> None:
        pass


class SelectTool(BaseTool):
    cursor = Qt.CursorShape.ArrowCursor

    def activate(self, plot_item: pg.PlotItem, widget: QWidget) -> None:
        widget.setCursor(QCursor(self.cursor))
        plot_item.getViewBox().setMouseEnabled(x=True, y=True)

    def on_click(self, ev: object, ctx: ToolContext) -> None:
        ctx.controller.on_click(ev)


class CreateTool(BaseTool):
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self) -> None:
        self._start: QPointF | None = None
        self._start_scene: QPointF | None = None
        self._rect_item = QGraphicsRectItem()
        self._rect_item.setPen(pg.mkPen("y", width=2, style=Qt.PenStyle.DashLine))
        self._rect_item.setZValue(1e9)
        self._plot_item: pg.PlotItem | None = None
        self._is_added_to_scene: bool = False

    def activate(self, plot_item: pg.PlotItem, widget: QWidget) -> None:
        super().activate(plot_item, widget)
        self._plot_item = plot_item
        if not self._is_added_to_scene:
            plot_item.addItem(self._rect_item)
            self._is_added_to_scene = True
        self._rect_item.hide()

    def deactivate(self) -> None:
        if self._is_added_to_scene and self._plot_item is not None:
            self._rect_item.hide()
            self._plot_item.removeItem(self._rect_item)
            self._is_added_to_scene = False
        self._plot_item = None
        self._start = None
        self._start_scene = None

    def on_drag(self, ev: object, ctx: ToolContext) -> None:
        if ev.button() != Qt.MouseButton.LeftButton or ctx.active_group is None:
            return
            
        pi  = self._plot_item
        pos = pi.vb.mapSceneToView(ev.scenePos())

        if ev.isStart():
            self._start = pos
            self._start_scene = ev.scenePos()
            self._rect_item.setRect(QRectF(pos.x(), pos.y(), 0, 0))
            self._rect_item.show()

        elif ev.isFinish():
            self._rect_item.hide()
            if self._start is None or self._start_scene is None:
                return
                
            x = min(self._start.x(), pos.x())
            y = min(self._start.y(), pos.y())
            w = abs(self._start.x() - pos.x())
            h = abs(self._start.y() - pos.y())
            
            dx_pixels = abs(ev.scenePos().x() - self._start_scene.x())
            dy_pixels = abs(ev.scenePos().y() - self._start_scene.y())
            
            self._start = None
            self._start_scene = None
            
            if dx_pixels > 5 and dy_pixels > 5:
                self._add_roi(x, y, w, h, ctx)

        else:
            if self._start is None:
                return
            x = min(self._start.x(), pos.x())
            y = min(self._start.y(), pos.y())
            w = abs(self._start.x() - pos.x())
            h = abs(self._start.y() - pos.y())
            self._rect_item.setRect(QRectF(x, y, w, h))

    @staticmethod
    def _add_roi(x: float, y: float, w: float, h: float, ctx: ToolContext) -> None:
        group = ctx.active_group
        new_idx = max((r.idx for r in group.rois), default=-1) + 1
        roi = ROIData(new_idx, x, y, w, h)
        group.rois.append(roi)
        ctx.roi_layer._add_to_bucket(roi)

        import numpy as np
        new_row = np.array([[x, y, w, h]], dtype=np.float32)
        new_idx_arr = np.array([new_idx], dtype=np.int32)
        ctx.roi_layer._rois_xy  = np.concatenate([ctx.roi_layer._rois_xy,  new_row],     axis=0)
        ctx.roi_layer._rois_idx = np.concatenate([ctx.roi_layer._rois_idx, new_idx_arr], axis=0)
        ctx.roi_layer._invalidate_cache()

        ctx.controller.deselect()
        ctx.controller.select(roi)
        
        # Сигнализируем о добавлении (для сохранения в CSV)
        ctx.controller.signals.rois_changed.emit(group.name)


class DeleteTool(BaseTool):
    cursor = Qt.CursorShape.ForbiddenCursor

    def on_click(self, ev: object, ctx: ToolContext) -> None:
        if ev.button() != Qt.MouseButton.LeftButton or ctx.active_group is None:
            return
        pi  = self._plot_item
        pos = pi.vb.mapSceneToView(ev.scenePos())
        self._delete_at(pos.x(), pos.y(), ctx)

    def activate(self, plot_item: pg.PlotItem, widget: QWidget) -> None:
        super().activate(plot_item, widget)
        self._plot_item = plot_item

    def deactivate(self) -> None:
        self._plot_item = None

    @staticmethod
    def _delete_at(px: float, py: float, ctx: ToolContext) -> None:
        layer  = ctx.roi_layer
        group  = ctx.active_group
        b      = int(px) // layer.BUCKET_W
        bucket = layer._buckets.get(b, [])
        hit    = next((r for r in reversed(bucket) if r.contains(px, py)), None)
        if hit is None:
            return
        if ctx.controller.active_idx == hit.idx:
            ctx.controller.deselect()
        group.rois.remove(hit)
        for bkt in layer._buckets.values():
            if hit in bkt:
                bkt.remove(hit)
        import numpy as np
        mask = layer._rois_idx != hit.idx
        layer._rois_xy  = layer._rois_xy[mask]
        layer._rois_idx = layer._rois_idx[mask]
        layer._invalidate_cache()
        
        # Сигнализируем об удалении (для сохранения в CSV)
        ctx.controller.signals.rois_changed.emit(group.name)


class DragProxy(pg.GraphicsObject):
    def __init__(self, tool_manager: ToolManager) -> None:
        super().__init__()
        self._tm = tool_manager

    def boundingRect(self) -> QRectF:
        return QRectF(-1e9, -1e9, 2e9, 2e9)

    def paint(self, *args: object) -> None:
        pass

    def mouseDragEvent(self, ev: object) -> None:
        if isinstance(self._tm.active_tool, SelectTool):
            ev.ignore()
            return
        ev.accept()
        self._tm._on_drag(ev)


class ToolManager(QObject):
    tool_changed = Signal(int)

    def __init__(
        self,
        plot_item:  pg.PlotItem,
        widget:     QWidget,
        controller: ROIController,
        roi_layer:  FastROILayer,
        all_groups: list[ROIGroup],
    ) -> None:
        super().__init__()
        self._plot_item  = plot_item
        self._widget     = widget
        self._ctx        = ToolContext(
            all_groups=all_groups,
            roi_layer=roi_layer,
            controller=controller,
        )
        self._tools: dict[int, BaseTool] = {
            0: SelectTool(),
            1: CreateTool(),
            2: DeleteTool(),
        }
        self._active_id:   int      = 0
        self._active_tool: BaseTool = self._tools[0]

        self._drag_proxy = DragProxy(self)
        plot_item.addItem(self._drag_proxy)

    @property
    def active_tool(self) -> BaseTool:
        return self._active_tool

    def set_tool(self, tool_id: int) -> None:
        if tool_id not in self._tools or tool_id == self._active_id:
            return
        self._active_tool.deactivate()
        self._active_id   = tool_id
        self._active_tool = self._tools[tool_id]
        self._active_tool.activate(self._plot_item, self._widget)
        self.tool_changed.emit(tool_id)

    def on_click(self, ev: object) -> None:
        self._active_tool.on_click(ev, self._ctx)

    def _on_drag(self, ev: object) -> None:
        self._active_tool.on_drag(ev, self._ctx)

    def update_groups(self, all_groups: list[ROIGroup]) -> None:
        self._ctx = ToolContext(
            all_groups=all_groups,
            roi_layer=self._ctx.roi_layer,
            controller=self._ctx.controller,
        )

    def reset_tool_state(self) -> None:
        if self._active_tool:
            self._active_tool.deactivate()
            self._active_tool.activate(self._plot_item, self._widget)