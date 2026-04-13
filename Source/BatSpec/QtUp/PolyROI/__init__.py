from __future__ import annotations

import csv
import numpy as np
import pyqtgraph as pg
from pathlib import Path
from PySide6.QtCore import Qt, QRectF, QObject, Signal
from PySide6.QtGui import QColor, QPen, QBrush, QPainter, QImage
from PySide6.QtWidgets import QLabel


class ROIData:
    __slots__ = ("idx", "x", "y", "w", "h")

    def __init__(self, idx: int, x: float, y: float, w: float, h: float) -> None:
        self.idx, self.x, self.y, self.w, self.h = idx, x, y, w, h

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    @property
    def qrect(self) -> QRectF:
        return QRectF(self.x, self.y, self.w, self.h)


class ROIGroup:
    def __init__(
        self,
        name: str,
        rois: list[ROIData],
        base_color: QColor,
        active_color: QColor,
    ) -> None:
        self.name = name
        self.rois = rois
        self.base_color = base_color
        self.active_color = active_color


# ── Инструмент синхронизации CSV ──────────────────────────────────────────────
class ROISync:
    @staticmethod
    def save_csv(rois: list[ROIData], path: Path) -> None:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'x', 'y', 'w', 'h'])
            for r in rois:
                writer.writerow([r.idx, r.x, r.y, r.w, r.h])

    @staticmethod
    def load_csv(path: Path) -> list[ROIData]:
        rois = []
        if not path.exists():
            return rois
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rois.append(ROIData(
                    int(row['idx']),
                    float(row['x']),
                    float(row['y']),
                    float(row['w']),
                    float(row['h'])
                ))
        return rois


class EightHandleROI(pg.ROI):
    def __init__(self, pos: list[float], size: list[float], **kwargs: object) -> None:
        kwargs["rotatable"] = False
        super().__init__(pos, size, **kwargs)
        self.addScaleHandle([0,   0  ], [1,   1  ])
        self.addScaleHandle([1,   1  ], [0,   0  ])
        self.addScaleHandle([0,   1  ], [1,   0  ])
        self.addScaleHandle([1,   0  ], [0,   1  ])
        self.addScaleHandle([0.5, 0  ], [0.5, 1  ])
        self.addScaleHandle([0.5, 1  ], [0.5, 0  ])
        self.addScaleHandle([0,   0.5], [1,   0.5])
        self.addScaleHandle([1,   0.5], [0,   0.5])


class FastROILayer(pg.GraphicsObject):
    def __init__(self, max_x: float, max_y: float) -> None:
        super().__init__()
        self.max_x = max_x
        self.max_y = max_y

        self.hidden_idx: int | None = None
        self.current_group: ROIGroup | None = None

        self.BUCKET_W: int = 1000
        self._buckets: dict[int, list[ROIData]] = {}

        self._rois_xy: np.ndarray = np.empty((0, 4), dtype=np.float32)
        self._rois_idx: np.ndarray = np.empty(0, dtype=np.int32)

        self._fill_rgba: tuple[int, int, int, int] = (255, 255, 255, 50)
        self._line_rgba: tuple[int, int, int, int] = (255, 255, 255, 255)

        self._cache: QImage | None = None
        self._cache_vr: QRectF | None = None
        self._vb_connected: bool = False

    def resize(self, max_x: float, max_y: float) -> None:
        self.prepareGeometryChange()
        self.max_x = max_x
        self.max_y = max_y
        self._invalidate_cache()

    def clear(self) -> None:
        self.current_group = None
        self.hidden_idx = None
        self._buckets.clear()
        self._rois_xy = np.empty((0, 4), dtype=np.float32)
        self._rois_idx = np.empty(0, dtype=np.int32)
        self._invalidate_cache()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.max_x, self.max_y)

    def paint(self, p: QPainter, *args: object) -> None:
        if not self.current_group:
            return
        vb = self.getViewBox()
        if not vb:
            return
        if not self._vb_connected:
            vb.sigRangeChanged.connect(self._invalidate_cache)
            self._vb_connected = True

        vr = vb.viewRect()
        if self._cache is None or self._cache_vr != vr:
            self._rebuild_cache(vb, vr)

        if self._cache is not None:
            p.drawImage(vr, self._cache)

    def _invalidate_cache(self, *_: object) -> None:
        self._cache = None
        self.update()

    def _rebuild_cache(self, vb: pg.ViewBox, vr: QRectF) -> None:
        rect = vb.boundingRect() 
        pw, ph = max(int(rect.width()), 1), max(int(rect.height()), 1)

        sx = pw / max(vr.width(), 1e-9)
        sy = ph / max(vr.height(), 1e-9)
        vx, vy = vr.left(), vr.top()

        rois = self._rois_xy
        idxs = self._rois_idx

        x0 = rois[:, 0]
        y0 = rois[:, 1]
        x1 = x0 + rois[:, 2]
        y1 = y0 + rois[:, 3]

        visible_mask = (
            (x1 >= vr.left()) & (x0 <= vr.right()) &
            (y1 >= vr.top())  & (y0 <= vr.bottom())
        )
        if self.hidden_idx is not None:
            visible_mask &= (idxs != self.hidden_idx)

        rx  = rois[visible_mask, 0]
        ry  = rois[visible_mask, 1]
        rw  = rois[visible_mask, 2]
        rh  = rois[visible_mask, 3]

        if rx.size == 0:
            self._cache = QImage(pw, ph, QImage.Format.Format_ARGB32)
            self._cache.fill(Qt.GlobalColor.transparent)
            self._cache_vr = QRectF(vr)
            return

        px0 = np.clip(((rx      - vx) * sx).astype(np.int32), 0, pw - 1)
        py0 = np.clip(((ry      - vy) * sy).astype(np.int32), 0, ph - 1)
        px1 = np.clip(((rx + rw - vx) * sx).astype(np.int32), 0, pw - 1)
        py1 = np.clip(((ry + rh - vy) * sy).astype(np.int32), 0, ph - 1)

        fr, fg, fb, fa = self._fill_rgba
        lr, lg, lb, la = self._line_rgba

        fill_pixel = (fa << 24) | (fr << 16) | (fg << 8) | fb
        line_pixel = (la << 24) | (lr << 16) | (lg << 8) | lb

        buf = np.zeros((ph, pw), dtype=np.uint32)
        _fill_rects_numpy(buf, px0, py0, px1, py1, fill_pixel, line_pixel)

        buf_bytes = buf.tobytes()
        img = QImage(buf_bytes, pw, ph, pw * 4, QImage.Format.Format_ARGB32)
        self._cache = img.copy()
        self._cache_vr = QRectF(vr)

    def set_group(self, group: ROIGroup) -> None:
        self.current_group = group
        self.hidden_idx = None

        c = group.base_color
        self._line_rgba = (c.red(), c.green(), c.blue(), 255)
        self._fill_rgba = (c.red(), c.green(), c.blue(), 50)

        self._buckets = {}
        if group.rois:
            xs = np.array([r.x for r in group.rois], dtype=np.float32)
            ys = np.array([r.y for r in group.rois], dtype=np.float32)
            ws = np.array([r.w for r in group.rois], dtype=np.float32)
            hs = np.array([r.h for r in group.rois], dtype=np.float32)
            self._rois_xy  = np.stack([xs, ys, ws, hs], axis=1)
            self._rois_idx = np.array([r.idx for r in group.rois], dtype=np.int32)
            for r in group.rois:
                self._add_to_bucket(r)
        else:
            self._rois_xy  = np.empty((0, 4), dtype=np.float32)
            self._rois_idx = np.empty(0, dtype=np.int32)

        self._invalidate_cache()

    def _add_to_bucket(self, r: ROIData) -> None:
        for b in range(int(r.x) // self.BUCKET_W, int(r.x + r.w) // self.BUCKET_W + 1):
            self._buckets.setdefault(b, []).append(r)

    def hide_roi(self, idx: int | None) -> None:
        self.hidden_idx = idx
        self._invalidate_cache()

    def update_roi_geometry(self, idx: int, x: float, y: float, w: float, h: float) -> None:
        if not self.current_group:
            return
        roi = next((r for r in self.current_group.rois if r.idx == idx), None)
        if roi is None:
            return
        for b in range(int(roi.x) // self.BUCKET_W, int(roi.x + roi.w) // self.BUCKET_W + 1):
            bucket = self._buckets.get(b)
            if bucket and roi in bucket:
                bucket.remove(roi)
        roi.x, roi.y, roi.w, roi.h = x, y, w, h
        self._add_to_bucket(roi)

        mask = self._rois_idx == idx
        self._rois_xy[mask] = [x, y, w, h]
        self._invalidate_cache()


def _fill_rects_numpy(
    buf: np.ndarray, px0: np.ndarray, py0: np.ndarray,
    px1: np.ndarray, py1: np.ndarray, fill_pixel: int, line_pixel: int
) -> None:
    for i in range(len(px0)):
        x0, y0, x1, y1 = px0[i], py0[i], px1[i], py1[i]
        if x0 == x1 or y0 == y1:
            buf[y0:y1+1, x0:x1+1] = line_pixel
        else:
            buf[y0:y1+1, x0:x1+1] = fill_pixel
            buf[y0,    x0:x1+1] = line_pixel
            buf[y1,    x0:x1+1] = line_pixel
            buf[y0:y1+1, x0   ] = line_pixel
            buf[y0:y1+1, x1   ] = line_pixel


class ROIController(QObject):
    class Signals(QObject):
        group_changed  = Signal(str)
        roi_selected   = Signal(str, int, float, float, float, float)
        roi_deselected = Signal(str)
        # НОВЫЙ СИГНАЛ: срабатывает, когда ROI добавлен, удален или изменен
        rois_changed   = Signal(str)

    def __init__(self, plot_item: pg.PlotItem, roi_layer: FastROILayer) -> None:
        super().__init__()
        self.signals = ROIController.Signals()
        self.pi = plot_item
        self.layer = roi_layer
        self.groups: list[ROIGroup] = []
        self.active_group: ROIGroup | None = None
        self.active_item: EightHandleROI | None = None
        self.active_idx: int | None = None

    def clear(self) -> None:
        self.deselect()
        self.groups = []
        self.active_group = None
        self.layer.clear()
        self.signals.group_changed.emit("Нет данных")

    def set_groups(self, groups: list[ROIGroup]) -> None:
        self.groups = groups
        if groups:
            self.switch_group(0)

    def switch_group(self, group_index: int) -> None:
        if not self.groups:
            return
        self.deselect()
        self.active_group = self.groups[group_index]
        self.layer.set_group(self.active_group)
        self.signals.group_changed.emit(self.active_group.name)

    def on_click(self, ev: object) -> None:
        if ev.button() != Qt.LeftButton or not self.active_group:
            return
        pos = self.pi.vb.mapSceneToView(ev.scenePos())
        px, py = pos.x(), pos.y()
        b = int(px) // self.layer.BUCKET_W
        hit = next(
            (r for r in reversed(self.layer._buckets.get(b, [])) if r.contains(px, py)),
            None,
        )
        if hit is None:
            self.deselect()
        elif self.active_idx != hit.idx:
            self.deselect()
            self.select(hit)

    def select(self, roi: ROIData) -> None:
        self.active_idx = roi.idx
        self.layer.hide_roi(roi.idx)
        c_active = self.active_group.active_color
        c_handle = QColor(c_active)
        c_handle.setAlpha(200)
        self.active_item = EightHandleROI(
            pos=[roi.x, roi.y], size=[roi.w, roi.h],
            pen=pg.mkPen(color=c_active, width=2),
            hoverPen=pg.mkPen(color=QColor("white"), width=2),
            handlePen=pg.mkPen(color=c_handle, width=1),
        )
        self.active_item.sigRegionChangeFinished.connect(self._sync_roi)
        self.pi.addItem(self.active_item)
        self.signals.roi_selected.emit(
            self.active_group.name, roi.idx, roi.x, roi.y, roi.w, roi.h
        )

    def deselect(self) -> None:
        if self.active_item:
            self._sync_roi()
            self.pi.removeItem(self.active_item)
            try:
                self.active_item.deleteLater()
            except RuntimeError:
                pass
                
        self.active_item = None
        self.active_idx = None
        self.layer.hide_roi(None)
        if self.active_group:
            self.signals.roi_deselected.emit(self.active_group.name)

    def _sync_roi(self) -> None:
        if not self.active_item or self.active_idx is None:
            return
        p, s = self.active_item.pos(), self.active_item.size()
        x, y, w, h = p.x(), p.y(), s.x(), s.y()
        self.layer.update_roi_geometry(self.active_idx, x, y, w, h)
        self.signals.roi_selected.emit(
            self.active_group.name, self.active_idx, x, y, w, h
        )
        # Оповещаем подписчиков о том, что геометрия ROI была изменена
        self.signals.rois_changed.emit(self.active_group.name)


class ROIStatusLabel(QLabel):
    def __init__(self, controller: ROIController, parent: object = None) -> None:
        super().__init__("Нет данных", parent)
        sig = controller.signals
        sig.group_changed.connect(
            lambda name: self.setText(f"Активна: {name} | Нет выделения" if name != "Нет данных" else name)
        )
        sig.roi_selected.connect(
            lambda name, idx, x, y, w, h:
                self.setText(f"{name} | ROI #{idx}  x={x:.0f}  y={y:.0f}  w={w:.0f}  h={h:.0f}")
        )
        sig.roi_deselected.connect(
            lambda name: self.setText(f"Активна: {name} | Нет выделения")
        )


from PySide6.QtWidgets import QGraphicsRectItem
from PySide6.QtCore import Qt
import pyqtgraph as pg

class ROICreatorTool(pg.GraphicsObject):
    def __init__(self, plot_item: pg.PlotItem, on_create_callback) -> None:
        super().__init__()
        self.pi = plot_item
        self.on_create = on_create_callback
        self.active = False
        self.drawing = False  # НОВЫЙ ФЛАГ
        self.start_pos = None

        self.draw_rect = QGraphicsRectItem()
        self.draw_rect.setPen(pg.mkPen('y', width=2, style=Qt.DashLine))
        self.pi.addItem(self.draw_rect)
        self.draw_rect.hide()

    def boundingRect(self):
        return self.pi.vb.viewRect()

    def paint(self, *args): pass

    def abort(self):
        """Принудительно отменяет рисование (если сменился инструмент)"""
        self.drawing = False
        self.draw_rect.hide()

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        
        # Разрешаем продолжить/завершить, если мы УЖЕ начали рисовать
        if not self.active and not self.drawing:
            ev.ignore()
            return
        
        ev.accept()
        pos = self.pi.vb.mapSceneToView(ev.scenePos())
        
        if ev.isStart() and self.active:
            self.drawing = True
            self.start_pos = pos
            self.draw_rect.setRect(pos.x(), pos.y(), 0, 0)
            self.draw_rect.show()
            
        elif ev.isFinish() and self.drawing:
            self.drawing = False
            self.draw_rect.hide()
            x = min(self.start_pos.x(), pos.x())
            y = min(self.start_pos.y(), pos.y())
            w = abs(self.start_pos.x() - pos.x())
            h = abs(self.start_pos.y() - pos.y())
            
            if w > 1e-5 and h > 1e-5:
                self.on_create(x, y, w, h)
                
        elif self.drawing:
            x = min(self.start_pos.x(), pos.x())
            y = min(self.start_pos.y(), pos.y())
            w = abs(self.start_pos.x() - pos.x())
            h = abs(self.start_pos.y() - pos.y())
            self.draw_rect.setRect(x, y, w, h)