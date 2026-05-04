import numpy as np
from numpy.typing import NDArray
import pyqtgraph as pg  # type: ignore
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from dataclasses import dataclass
from typing import Any, Sequence, cast


def _chaikin_smooth(pts: NDArray[np.float32], iterations: int = 3) -> NDArray[np.float32]:
    if len(pts) < 3:
        return pts
    for _ in range(iterations):
        p0 = pts[:-1]
        p1 = pts[1:]
        q1 = 0.75 * p0 + 0.25 * p1
        q2 = 0.25 * p0 + 0.75 * p1
        new_pts = np.empty((len(pts) * 2 - 2, 2), dtype=np.float32)
        new_pts[0::2], new_pts[1::2] = q1, q2
        new_pts[0], new_pts[-1] = pts[0], pts[-1]
        pts = new_pts
    return pts


class EightHandleROI(pg.ROI):  # type: ignore[misc]
    def __init__(
        self,
        pos: Sequence[float] | QtCore.QPointF,
        size: Sequence[float] | QtCore.QSizeF,
        pen: QtGui.QPen | None,
        hoverPen: QtGui.QPen | None = None,
        handlePen: QtGui.QPen | None = None,
    ) -> None:
        super().__init__(pos, size, movable=True, resizable=True, rotatable=False, pen=pen, hoverPen=hoverPen)  # type: ignore[arg-type]
        
        # Явная типизация списка координат для подавления ошибок "Type of x/y is unknown"
        coords: list[tuple[float, float]] = [
            (0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), 
            (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5)
        ]
        for x, y in coords:
            self.addScaleHandle([x, y], [1.0 - x, 1.0 - y])  # type: ignore[no-untyped-call]


class DraggablePointROI(pg.ROI):  # type: ignore[misc]
    def __init__(
        self,
        pos: Sequence[float] | QtCore.QPointF,
        pen: QtGui.QPen,
        brush: QtGui.QBrush,
    ) -> None:
        super().__init__(pos, [0, 0], movable=True, resizable=False, rotatable=False)  # type: ignore[arg-type]
        self._pen: QtGui.QPen = pen
        self._brush: QtGui.QBrush = brush

    def paint(
        self,
        p: QtGui.QPainter,
        opt: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(self._pen)
        p.setBrush(self._brush)
        px_size = self.pixelSize()  # type: ignore[no-untyped-call]
        px: float = float(px_size[0]) if px_size[0] is not None else 1.0
        py: float = float(px_size[1]) if px_size[1] is not None else 1.0
        rx: float = 6.0 * px
        ry: float = 6.0 * py
        p.drawEllipse(QtCore.QRectF(-rx, -ry, rx * 2.0, ry * 2.0))

    def boundingRect(self) -> QtCore.QRectF:
        px_size = self.pixelSize()  # type: ignore[no-untyped-call]
        px: float = float(px_size[0]) if px_size[0] is not None else 1.0
        py: float = float(px_size[1]) if px_size[1] is not None else 1.0
        rx: float = 8.0 * px
        ry: float = 8.0 * py
        return QtCore.QRectF(-rx, -ry, rx * 2.0, ry * 2.0)


class SmoothPolyLineROI(pg.PolyLineROI):  # type: ignore[misc]
    def __init__(
        self,
        positions: Sequence[Sequence[float]],
        pen: QtGui.QPen,
    ) -> None:
        super().__init__(  # type: ignore[no-untyped-call]
            positions,
            closed=False,
            pen=pg.mkPen(None),  # type: ignore[no-untyped-call]
            hoverPen=None,
            handlePen=pg.mkPen('w', width=2),  # type: ignore[no-untyped-call]
        )
        self.handleSize: int = 6
        for h in self.getHandles():  # type: ignore[no-untyped-call]
            h.radius = self.handleSize  # type: ignore[reportUnknownMemberType]
            h.buildPath()  # type: ignore[reportUnknownMemberType]
            h._shape = None  # type: ignore[reportUnknownMemberType]

        self.smooth_path_item: QtWidgets.QGraphicsPathItem = QtWidgets.QGraphicsPathItem(self)
        self.smooth_path_item.setPen(pen)
        self.smooth_path_item.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        self.smooth_path_item.setZValue(-1)
        self.sigRegionChanged.connect(self.update_smooth_path)  # type: ignore[no-untyped-call]
        self.update_smooth_path()

    def update_smooth_path(self) -> None:
        handles: list[Any] = self.getHandles()  # type: ignore[no-untyped-call]
        if len(handles) < 2:
            return
        QtCore.QTimer.singleShot(0, self._force_handles_to_top)  # type: ignore[reportUnknownMemberType]
        pts = np.array(
            [[float(h.pos().x()), float(h.pos().y())] for h in handles],
            dtype=np.float32,
        )
        smooth_pts = _chaikin_smooth(pts, iterations=4)
        path: QtGui.QPainterPath = pg.arrayToQPath(  # type: ignore[no-untyped-call]
            smooth_pts[:, 0], smooth_pts[:, 1], connect='finite'
        )
        self.smooth_path_item.setPath(path)

    def _force_handles_to_top(self) -> None:
        for h in self.getHandles():  # type: ignore[no-untyped-call]
            h.setZValue(99999)  # type: ignore[reportUnknownMemberType]
            if h.radius != self.handleSize:  # type: ignore[reportUnknownMemberType]
                h.radius = self.handleSize  # type: ignore[reportUnknownMemberType]
                h.buildPath()  # type: ignore[reportUnknownMemberType]
                h._shape = None  # type: ignore[reportUnknownMemberType]
                h.prepareGeometryChange()  # type: ignore[reportUnknownMemberType]

    def get_raw_points(self) -> list[list[float]]:
        return [
            [float(self.mapToParent(h.pos()).x()), float(self.mapToParent(h.pos()).y())]  # type: ignore[no-untyped-call]
            for h in self.getHandles()  # type: ignore[no-untyped-call]
        ]


# Separate typed geom aliases per layer kind so Pylance can narrow them.
RectGeom = tuple[float, float, float, float]   # x, y, w, h
PointGeom = tuple[float, float]                # x, y
CurveGeom = NDArray[np.float32]                # shape (N, 2)


@dataclass
class FastElement:
    id: str
    group_id: str
    geom: list[float] | list[list[float]] | NDArray[np.float32]


class AbstractFastLayer(pg.GraphicsObject):  # type: ignore[misc]
    def __init__(self, bucket_width: float = 1000.0) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.bucket_width: float = bucket_width
        self.elements: dict[str, FastElement] = {}
        self.group_colors: dict[str, tuple[QtGui.QColor, QtGui.QColor | None]] = {}
        self.hidden_ids: set[str] = set()
        self._path_cache: dict[str, QtGui.QPainterPath] = {}
        self._buckets: dict[int, list[str]] = {}
        self._bounding_rect: QtCore.QRectF = QtCore.QRectF()

    def set_colors(
        self,
        group_id: str,
        pen_color: QtGui.QColor,
        brush_color: QtGui.QColor | None = None,
    ) -> None:
        self.group_colors[group_id] = (pen_color, brush_color)

    def add_element(self, el: FastElement) -> None:
        self.elements[el.id] = el
        self.add_to_buckets(el)

    def remove_element(self, el_id: str) -> None:
        if el_id in self.elements:
            self.remove_from_buckets(self.elements[el_id])
            del self.elements[el_id]

    def hide_element(self, el_id: str) -> None:
        self.hidden_ids.add(el_id)

    def show_element(self, el_id: str) -> None:
        self.hidden_ids.discard(el_id)

    def _get_x_bounds(self, el: FastElement) -> tuple[float, float]:
        raise NotImplementedError

    def add_to_buckets(self, el: FastElement) -> None:
        x_min, x_max = self._get_x_bounds(el)
        b0: int = int(x_min // self.bucket_width)
        b1: int = int(x_max // self.bucket_width)
        for b in range(b0, b1 + 1):
            self._buckets.setdefault(b, []).append(el.id)

    def remove_from_buckets(self, el: FastElement) -> None:
        x_min, x_max = self._get_x_bounds(el)
        b0: int = int(x_min // self.bucket_width)
        b1: int = int(x_max // self.bucket_width)
        for b in range(b0, b1 + 1):
            if b in self._buckets and el.id in self._buckets[b]:
                self._buckets[b].remove(el.id)

    def hit_test(self, px: float, py: float, threshold: float = 0.0) -> FastElement | None:
        b: int = int(px // self.bucket_width)
        for el_id in reversed(self._buckets.get(b, [])):
            if el_id in self.hidden_ids:
                continue
            el = self.elements[el_id]
            if self._check_hit(el, px, py, threshold):
                return el
        return None

    def _check_hit(self, el: FastElement, px: float, py: float, threshold: float) -> bool:
        raise NotImplementedError

    def rebuild(self) -> None:
        self._path_cache.clear()
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        return self._bounding_rect

    def dataBounds(
        self,
        ax: int,
        frac: float = 1.0,
        orthoRange: Sequence[float] | None = None,
    ) -> list[float | None]:
        if self._bounding_rect.isNull():
            return [None, None]
        if ax == 0:
            return [self._bounding_rect.left(), self._bounding_rect.right()]
        if ax == 1:
            return [self._bounding_rect.top(), self._bounding_rect.bottom()]
        return [None, None]


class FastRectLayer(AbstractFastLayer):
    def _get_x_bounds(self, el: FastElement) -> tuple[float, float]:
        g = el.geom
        x = float(g[0])  # type: ignore[arg-type]
        w = float(g[2])  # type: ignore[arg-type]
        return x, x + w

    def _check_hit(self, el: FastElement, px: float, py: float, threshold: float) -> bool:
        g = el.geom
        x, y, w, h = float(g[0]), float(g[1]), float(g[2]), float(g[3])  # type: ignore[arg-type]
        return (x - threshold <= px <= x + w + threshold) and (y - threshold <= py <= y + h + threshold)

    def rebuild(self) -> None:
        self.prepareGeometryChange()
        self._path_cache.clear()

        for el in self.elements.values():
            if el.id in self.hidden_ids:
                continue
            if el.group_id not in self._path_cache:
                self._path_cache[el.group_id] = QtGui.QPainterPath()
            g = el.geom
            self._path_cache[el.group_id].addRect(
                QtCore.QRectF(float(g[0]), float(g[1]), float(g[2]), float(g[3]))  # type: ignore[arg-type]
            )

        br = QtCore.QRectF()
        for path in self._path_cache.values():
            br = br.united(path.boundingRect())
        self._bounding_rect = br
        self.update()

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        for gid, path in self._path_cache.items():
            if path.isEmpty() or gid not in self.group_colors:
                continue
            pen_c, brush_c = self.group_colors[gid]
            painter.setPen(pg.mkPen(pen_c, width=1, cosmetic=True))  # type: ignore[no-untyped-call]
            if brush_c is not None:
                painter.setBrush(pg.mkBrush(brush_c))  # type: ignore[no-untyped-call]
            painter.drawPath(path)


class FastCurveLayer(AbstractFastLayer):
    def _get_x_bounds(self, el: FastElement) -> tuple[float, float]:
        pts = np.asarray(el.geom, dtype=np.float32)
        return float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))

    def _check_hit(self, el: FastElement, px: float, py: float, threshold: float) -> bool:
        pts = np.asarray(el.geom, dtype=np.float32)
        p0 = pts[:-1]
        p1 = pts[1:]
        dx: NDArray[np.float32] = p1[:, 0] - p0[:, 0]
        dy: NDArray[np.float32] = p1[:, 1] - p0[:, 1]
        l2: NDArray[np.float32] = dx * dx + dy * dy
        l2[l2 == 0] = 1e-6
        t: NDArray[np.float32] = np.clip(((px - p0[:, 0]) * dx + (py - p0[:, 1]) * dy) / l2, 0, 1)
        dist_sq: NDArray[np.float32] = (px - (p0[:, 0] + t * dx)) ** 2 + (py - (p0[:, 1] + t * dy)) ** 2
        return bool(np.min(dist_sq) <= threshold ** 2)

    def rebuild(self) -> None:
        self.prepareGeometryChange()
        self._path_cache.clear()
        nan_arr = np.array([np.nan], dtype=np.float32)
        group_pts: dict[str, list[list[NDArray[np.float32]]]] = {}

        for el in self.elements.values():
            if el.id in self.hidden_ids:
                continue
            pts = np.asarray(el.geom, dtype=np.float32)
            if len(pts) < 2:
                continue
            s_pts = _chaikin_smooth(pts, iterations=4)
            group_pts.setdefault(el.group_id, [[], []])
            group_pts[el.group_id][0].extend([s_pts[:, 0], nan_arr])
            group_pts[el.group_id][1].extend([s_pts[:, 1], nan_arr])

        br = QtCore.QRectF()
        for gid, (xs, ys) in group_pts.items():
            path: QtGui.QPainterPath = pg.arrayToQPath(  # type: ignore[no-untyped-call]
                np.concatenate(xs), np.concatenate(ys), connect='finite'
            )
            self._path_cache[gid] = path
            br = br.united(path.boundingRect())

        self._bounding_rect = br
        self.update()

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for gid, path in self._path_cache.items():
            if path.isEmpty() or gid not in self.group_colors:
                continue
            pen_c, _ = self.group_colors[gid]
            painter.setPen(pg.mkPen(pen_c, width=2, cosmetic=True))  # type: ignore[no-untyped-call]
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(path)


class FastPointLayer(AbstractFastLayer):
    def __init__(self, point_radius: int = 4) -> None:
        super().__init__()
        self.radius: int = point_radius
        self._poly_cache: dict[str, QtGui.QPolygonF] = {}

    def _get_x_bounds(self, el: FastElement) -> tuple[float, float]:
        x = float(el.geom[0])  # type: ignore[arg-type]
        return x, x

    def _check_hit(self, el: FastElement, px: float, py: float, threshold: float) -> bool:
        x = float(el.geom[0])  # type: ignore[arg-type]
        y = float(el.geom[1])  # type: ignore[arg-type]
        return bool((x - px) ** 2 + (y - py) ** 2 <= threshold ** 2)

    def rebuild(self) -> None:
        self.prepareGeometryChange()
        self._poly_cache.clear()

        for el in self.elements.values():
            if el.id in self.hidden_ids:
                continue
            if el.group_id not in self._poly_cache:
                self._poly_cache[el.group_id] = QtGui.QPolygonF()
            self._poly_cache[el.group_id].append(
                QtCore.QPointF(float(el.geom[0]), float(el.geom[1]))  # type: ignore[arg-type]
            )

        br = QtCore.QRectF()
        for poly in self._poly_cache.values():
            br = br.united(poly.boundingRect())

        if not br.isNull():
            br.adjust(-self.radius, -self.radius, self.radius, self.radius)

        self._bounding_rect = br
        self.update()

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for gid, poly in self._poly_cache.items():
            if poly.isEmpty() or gid not in self.group_colors:
                continue
            pen_c, _ = self.group_colors[gid]
            painter.setPen(
                pg.mkPen(pen_c, width=self.radius * 2, cosmetic=True, cap=QtCore.Qt.PenCapStyle.RoundCap)  # type: ignore[no-untyped-call]
            )
            painter.drawPoints(poly)


class FastGraphEditor:
    """
    Класс-менеджер для управления редактированием элементов.
    Позволяет извлекать элемент со слоя, заменять его редактируемым ROI,
    а затем сохранять измененную геометрию обратно в слой.
    """
    def __init__(self, plot_widget: pg.PlotWidget) -> None:
        self.plot_widget: pg.PlotWidget = plot_widget
        self.active_rois: dict[str, tuple[AbstractFastLayer, FastElement, pg.ROI]] = {}

    def edit_element(self, layer: AbstractFastLayer, el_id: str) -> None:
        if el_id not in layer.elements or el_id in self.active_rois:
            return

        el: FastElement = layer.elements[el_id]
        
        # Получаем стили элемента для покраски ROI в те же цвета
        pen_c, _ = layer.group_colors.get(el.group_id, (QtGui.QColor(255, 255, 255), None))
        pen = cast(QtGui.QPen, pg.mkPen(pen_c, width=2))  # type: ignore[no-untyped-call]
        
        roi: pg.ROI | None = None
        
        if isinstance(layer, FastRectLayer):
            g = el.geom
            roi = EightHandleROI(
                pos=[float(g[0]), float(g[1])],  # type: ignore[arg-type]
                size=[float(g[2]), float(g[3])],  # type: ignore[arg-type]
                pen=pen
            )
        elif isinstance(layer, FastCurveLayer):
            roi = SmoothPolyLineROI(
                positions=el.geom,  # type: ignore[arg-type]
                pen=pen
            )
        elif isinstance(layer, FastPointLayer):
            brush: QtGui.QBrush = pg.mkBrush(pen_c)  # type: ignore[no-untyped-call]
            roi = DraggablePointROI(
                pos=[float(el.geom[0]), float(el.geom[1])],  # type: ignore[arg-type]
                pen=pen,
                brush=brush
            )
            
        if roi is not None:
            # Скрываем исходный отрисованный элемент
            layer.hide_element(el_id)
            layer.rebuild()
            
            # Добавляем интерактивный ROI на график
            self.plot_widget.addItem(roi)  # type: ignore[no-untyped-call]
            self.active_rois[el_id] = (layer, el, roi)

    def commit_element(self, el_id: str) -> None:
        if el_id not in self.active_rois:
            return
        
        layer, el, roi = self.active_rois.pop(el_id)
        
        # 1. Важно: удаляем элемент из бакетов поиска ПО СТАРЫМ КООРДИНАТАМ
        layer.remove_from_buckets(el)
        
        # 2. Обновляем координаты элемента из ROI
        if isinstance(layer, FastRectLayer):
            pos = cast(QtCore.QPointF, roi.pos())  # type: ignore[no-untyped-call]
            size = cast(QtCore.QPointF, roi.size())  # type: ignore[no-untyped-call]
            el.geom = [float(pos.x()), float(pos.y()), float(size.x()), float(size.y())]
        elif isinstance(layer, FastCurveLayer) and isinstance(roi, SmoothPolyLineROI):
            el.geom = roi.get_raw_points()
        elif isinstance(layer, FastPointLayer):
            pos = cast(QtCore.QPointF, roi.pos())  # type: ignore[no-untyped-call]
            el.geom = [float(pos.x()), float(pos.y())]
            
        # 3. Добавляем элемент обратно в бакеты ПО НОВЫМ КООРДИНАТАМ
        layer.add_to_buckets(el)
            
        # Убираем ROI с графика и возвращаем отрисовку FastElement
        self.plot_widget.removeItem(roi)  # type: ignore[no-untyped-call]
        layer.show_element(el_id)
        layer.rebuild()
        
    def commit_all(self) -> None:
        for el_id in list(self.active_rois.keys()):
            self.commit_element(el_id)