from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF, QObject, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QLabel


class LineData:
    __slots__ = ("idx", "points", "min_x", "max_x", "min_y", "max_y")

    def __init__(self, idx: int, points: list[tuple[float, float]] | np.ndarray) -> None:
        self.idx = idx
        self.points = np.empty((0, 2), dtype=np.float32)
        self.set_points(points)

    def set_points(self, points: list[tuple[float, float]] | np.ndarray) -> None:
        self.points = np.asarray(points, dtype=np.float32)
        if len(self.points) > 0:
            self.min_x, self.min_y = np.min(self.points, axis=0)
            self.max_x, self.max_y = np.max(self.points, axis=0)
        else:
            self.min_x = self.max_x = self.min_y = self.max_y = 0.0

    def distance_to(self, px: float, py: float, sx: float = 1.0, sy: float = 1.0) -> float:
        if len(self.points) < 2:
            return float('inf')
        
        # Быстрый AABB тест перед математикой
        dx_box = max(0.0, self.min_x - px, px - self.max_x)
        dy_box = max(0.0, self.min_y - py, py - self.max_y)
        if (dx_box * sx) > 15 or (dy_box * sy) > 15:
            return float('inf')

        p0 = self.points[:-1]
        p1 = self.points[1:]
        
        dx = p1[:, 0] - p0[:, 0]
        dy = p1[:, 1] - p0[:, 1]
        l2 = dx*dx + dy*dy
        l2[l2 == 0] = 1e-6 
        
        t = np.clip(((px - p0[:, 0]) * dx + (py - p0[:, 1]) * dy) / l2, 0, 1)
        proj_x = p0[:, 0] + t * dx
        proj_y = p0[:, 1] + t * dy
        
        dist_sq = ((px - proj_x) * sx)**2 + ((py - proj_y) * sy)**2
        return np.sqrt(np.min(dist_sq))


class LineGroup:
    def __init__(self, name: str, lines: list[LineData], base_color: QColor, active_color: QColor) -> None:
        self.name = name
        self.lines = lines
        self.base_color = base_color
        self.active_color = active_color


class FastLineLayer(pg.GraphicsObject):
    def __init__(self) -> None:
        super().__init__()
        self.current_group: LineGroup | None = None
        self.hidden_idx: int | None = None
        
        self.BUCKET_W: int = 1000
        self._buckets: dict[int, list[LineData]] = {}
        
        self._path = QPainterPath()
        self._pen = pg.mkPen('w', width=1)

    def boundingRect(self) -> QRectF:
        vb = self.getViewBox()
        return vb.viewRect() if vb else QRectF()

    def paint(self, p: QPainter, *args: object) -> None:
        if not self._path.isEmpty():
            # ВАЖНО: Сглаживание отключено для тысяч объектов, иначе будут жесткие лаги
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            p.setPen(self._pen)
            p.drawPath(self._path)

    def set_group(self, group: LineGroup) -> None:
        self.current_group = group
        self.hidden_idx = None
        self._pen = pg.mkPen(group.base_color, width=1)
        self._rebuild_buckets()
        self.rebuild_path()

    def hide_line(self, idx: int | None) -> None:
        if self.hidden_idx != idx:
            self.hidden_idx = idx
            self.rebuild_path()

    def _rebuild_buckets(self) -> None:
        self._buckets.clear()
        if not self.current_group:
            return
        for line in self.current_group.lines:
            self.add_to_bucket(line)

    def add_to_bucket(self, line: LineData) -> None:
        b_start = int(line.min_x) // self.BUCKET_W
        b_end = int(line.max_x) // self.BUCKET_W + 1
        for b in range(b_start, b_end):
            self._buckets.setdefault(b, []).append(line)

    def remove_from_bucket(self, line: LineData) -> None:
        b_start = int(line.min_x) // self.BUCKET_W
        b_end = int(line.max_x) // self.BUCKET_W + 1
        for b in range(b_start, b_end):
            if b in self._buckets and line in self._buckets[b]:
                self._buckets[b].remove(line)

    def rebuild_path(self) -> None:
        """
        Магия оптимизации: сливаем все кривые в один массив, разделяя NaN.
        Это переносит 100% нагрузки по отрисовке тысяч линий в движок C++ (Qt).
        """
        if not self.current_group or not self.current_group.lines:
            self._path = QPainterPath()
            self.update()
            return

        xs, ys = [], []
        nan_arr = np.array([np.nan], dtype=np.float32)
        
        for line in self.current_group.lines:
            if line.idx == self.hidden_idx or len(line.points) == 0:
                continue
            xs.append(line.points[:, 0])
            xs.append(nan_arr)
            ys.append(line.points[:, 1])
            ys.append(nan_arr)

        if xs:
            x_arr = np.concatenate(xs)
            y_arr = np.concatenate(ys)
            self._path = pg.arrayToQPath(x_arr, y_arr, connect='finite')
        else:
            self._path = QPainterPath()
            
        self.update()

    def clear(self) -> None:
        self.current_group = None
        self.hidden_idx = None
        self._buckets.clear()
        self._path = QPainterPath()
        self.update()


class LineController(QObject):
    class Signals(QObject):
        group_changed = Signal(str)
        line_selected = Signal(str, int, int)
        line_deselected = Signal(str)
        lines_changed = Signal(str)

    def __init__(self, plot_item: pg.PlotItem, layer: FastLineLayer) -> None:
        super().__init__()
        self.signals = LineController.Signals()
        self.pi = plot_item
        self.layer = layer
        self.groups: list[LineGroup] = []
        self.active_group: LineGroup | None = None
        self.active_item: pg.PolyLineROI | None = None
        self.active_idx: int | None = None

    def clear(self) -> None:
        self.deselect()
        self.groups = []
        self.active_group = None
        self.layer.clear()
        self.signals.group_changed.emit("Нет данных")

    def set_groups(self, groups: list[LineGroup]) -> None:
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
        
        # Получаем пиксельный масштаб
        vr = self.pi.vb.viewRect()
        pw = max(1, self.pi.vb.width())
        ph = max(1, self.pi.vb.height())
        sx = pw / vr.width()
        sy = ph / vr.height()

        b = int(px) // self.layer.BUCKET_W
        lines_in_bucket = self.layer._buckets.get(b, [])

        best_hit = None
        min_dist = 10.0 # Радиус захвата в пикселях экрана

        for line in reversed(lines_in_bucket):
            d = line.distance_to(px, py, sx, sy)
            if d < min_dist:
                min_dist = d
                best_hit = line

        if best_hit is None:
            self.deselect()
        elif self.active_idx != best_hit.idx:
            self.deselect()
            self.select(best_hit)

    def select(self, line: LineData) -> None:
        self.active_idx = line.idx
        self.layer.hide_line(line.idx)
        
        c_active = self.active_group.active_color
        c_handle = QColor(c_active)
        c_handle.setAlpha(200)
        
        # Редактируемый слой
        self.active_item = pg.PolyLineROI(
            line.points,
            closed=False,
            pen=pg.mkPen(c_active, width=3),
            hoverPen=pg.mkPen('w', width=4),
            handlePen=pg.mkPen(c_handle, width=1)
        )
        self.active_item.sigRegionChangeFinished.connect(self._sync_line)
        self.pi.addItem(self.active_item)
        
        self.signals.line_selected.emit(self.active_group.name, line.idx, len(line.points))

    def deselect(self) -> None:
        if self.active_item:
            self._sync_line()
            self.pi.removeItem(self.active_item)
            try:
                self.active_item.deleteLater()
            except RuntimeError:
                pass
                
        self.active_item = None
        self.active_idx = None
        self.layer.hide_line(None)
        
        if self.active_group:
            self.signals.line_deselected.emit(self.active_group.name)

    def _sync_line(self) -> None:
        if not self.active_item or self.active_idx is None or not self.active_group:
            return
        
        line = next((l for l in self.active_group.lines if l.idx == self.active_idx), None)
        if line:
            self.layer.remove_from_bucket(line)
            
            pts = []
            base_pos = self.active_item.pos()
            for h in self.active_item.getHandles():
                p = h.pos()
                pts.append([base_pos.x() + p.x(), base_pos.y() + p.y()])
                
            line.set_points(pts)
            self.layer.add_to_bucket(line)
            self.layer.rebuild_path()
            self.signals.lines_changed.emit(self.active_group.name)


class LineStatusLabel(QLabel):
    def __init__(self, controller: LineController, parent: object = None) -> None:
        super().__init__("Нет данных", parent)
        sig = controller.signals
        sig.group_changed.connect(
            lambda name: self.setText(f"Слой: {name} | Нет выделения" if name != "Нет данных" else name)
        )
        sig.line_selected.connect(
            lambda name, idx, pts: self.setText(f"{name} | Выделена кривая #{idx} (узлов: {pts})")
        )
        sig.line_deselected.connect(
            lambda name: self.setText(f"Слой: {name} | Нет выделения")
        )
        