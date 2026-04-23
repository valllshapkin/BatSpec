import sys
import logging
import random
import numpy as np
import pyqtgraph as pg
from threading import Thread

from PySide6.QtCore import Qt, Signal, Slot, QRectF
from PySide6.QtGui import QTransform, QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                               QWidget, QLabel, QComboBox, QHBoxLayout)


# ------------------------------------------------------------------------------
# AdaptiveImageItem (остается без изменений - работает идеально)
# ------------------------------------------------------------------------------
class AdaptiveImageItem(pg.ImageItem):
    MIN_COLS = 64
    _sigMipmapReady = Signal()

    def __init__(self):
        super().__init__()
        self.setOpts(axisOrder='row-major')
        self._mipmap: list[np.ndarray] = []
        self._x_range = (0.0, 1.0)
        self._y_range = (0.0, 1.0)
        self._levels = (0.0, 1.0)
        self.setLevels(self._levels)
        self._plot_item: pg.PlotItem | None = None
        self._updating = False
        self._sigMipmapReady.connect(self._doUpdate)

    def attachTo(self, plot_item: pg.PlotItem) -> None:
        self._plot_item = plot_item
        plot_item.getViewBox().sigRangeChanged.connect(self._doUpdate)

    def setFullData(self, arr: np.ndarray, x_range: tuple, y_range: tuple) -> None:
        self._x_range = x_range
        self._y_range = y_range
        if self._levels is not None:
            self.setLevels(self._levels)
        self._mipmap = []
        Thread(target=self._buildMipmap, args=(arr,), daemon=True).start()

    def _buildMipmap(self, arr: np.ndarray) -> None:
        data = np.ascontiguousarray(arr, dtype=np.float32)
        levels = [data]
        cur = data
        while cur.shape[1] // 2 >= self.MIN_COLS:
            nf, nt = cur.shape
            nt2 = nt // 2
            down = (cur[:, :nt2 * 2].reshape(nf, nt2, 2).mean(axis=2).astype(np.float32))
            levels.append(down)
            cur = down
        self._mipmap = levels
        self._sigMipmapReady.emit()

    @Slot()
    def _doUpdate(self) -> None:
        if not self._mipmap or self._plot_item is None or self._updating:
            return
        self._updating = True
        try:
            self._renderTile()
        except Exception:
            logging.exception("AdaptiveImageItem._renderTile")
        finally:
            self._updating = False

    def _renderTile(self) -> None:
        vb = self._plot_item.getViewBox()
        [[x0, x1], [y0, y1]] = vb.viewRange()

        t_min, t_max = self._x_range
        f_min, f_max = self._y_range
        nfreq, ntime = self._mipmap[0].shape

        if ntime == 0 or nfreq == 0: return

        t_per_col = (t_max - t_min) / ntime
        f_per_row = (f_max - f_min) / nfreq

        c0 = max(0, int(np.floor((x0 - t_min) / t_per_col)))
        c1 = min(ntime, int(np.ceil((x1 - t_min) / t_per_col)) + 1)
        r0 = max(0, int(np.floor((y0 - f_min) / f_per_row)))
        r1 = min(nfreq, int(np.ceil((y1 - f_min) / f_per_row)) + 1)

        if c0 >= c1 or r0 >= r1: return

        vis_rows = r1 - r0
        vis_cols = c1 - c0

        geom = vb.screenGeometry()
        scr_h = geom.height() if (geom and geom.height() > 0) else 600
        scr_w = geom.width() if (geom and geom.width() > 0) else 1200

        tile_h = max(1, min(vis_rows, scr_h))
        tile_w = max(1, min(vis_cols, scr_w))

        col_step = vis_cols / tile_w
        k = min(max(0, int(np.floor(np.log2(max(1.0, col_step))))), len(self._mipmap) - 1)
        s = 1 << k
        level = self._mipmap[k]
        lw = level.shape[1]

        lc0 = c0 // s
        lc1 = min(lw, (c1 + s - 1) // s + 1)
        c0_a = lc0 * s

        crop = level[r0:r1, lc0:lc1]
        ch, cw = crop.shape

        bh = max(1, ch // tile_h)
        bw = max(1, cw // tile_w)

        if bh > 1 or bw > 1:
            out_h = ch // bh
            out_w = cw // bw
            sub = (crop[:bh * out_h, :bw * out_w].reshape(out_h, bh, out_w, bw).mean(axis=(1, 3)).astype(np.float32))
        else:
            sub = crop

        if sub.size == 0: return

        self.setLevels(self._levels)
        self.setImage(sub, autoLevels=False)

        tx = t_min + c0_a * t_per_col
        ty = f_min + r0 * f_per_row
        dx = bw * s * t_per_col
        dy = bh * f_per_row

        tr = QTransform()
        tr.translate(tx, ty)
        tr.scale(dx, dy)
        self.setTransform(tr)


# ------------------------------------------------------------------------------
# 8-Tочечный Кастомный ROI без вращения
# ------------------------------------------------------------------------------
class EightHandleROI(pg.ROI):
    """
    Кастомный ROI. 
    Наследуемся напрямую от базового pg.ROI, чтобы избежать ручек по умолчанию.
    Добавляем 8 ручек изменения размера и отключаем вращение через Alt.
    """
    def __init__(self, pos, size, **kwargs):
        # Принудительно отключаем вращение (блокирует клавишу Alt)
        kwargs['rotatable'] = False
        
        super().__init__(pos, size, **kwargs)
        
        # Добавляем 8 точек (позиция ручки -> якорь с противоположной стороны)
        # Углы
        self.addScaleHandle([0, 0], [1, 1])  # Левый нижний
        self.addScaleHandle([1, 1], [0, 0])  # Правый верхний
        self.addScaleHandle([0, 1], [1, 0])  # Левый верхний
        self.addScaleHandle([1, 0], [0, 1])  # Правый нижний
        
        # Грани (центры сторон)
        self.addScaleHandle([0.5, 0], [0.5, 1])  # Низ центр
        self.addScaleHandle([0.5, 1], [0.5, 0])  # Верх центр
        self.addScaleHandle([0, 0.5], [1, 0.5])  # Лево центр
        self.addScaleHandle([1, 0.5], [0, 0.5])  # Право центр


# ------------------------------------------------------------------------------
# Структуры данных и слои для ROI
# ------------------------------------------------------------------------------
class ROIData:
    __slots__ = ("idx", "x", "y", "w", "h")
    def __init__(self, idx, x, y, w, h):
        self.idx = idx
        self.x, self.y, self.w, self.h = x, y, w, h
        
    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    @property
    def qrect(self):
        return QRectF(self.x, self.y, self.w, self.h)


class ROIGroup:
    """Хранит данные одной группы (массива) ROI и ее стилистику"""
    def __init__(self, name: str, rois: list[ROIData], base_color: QColor, active_color: QColor):
        self.name = name
        self.rois = rois
        self.base_color = base_color
        self.active_color = active_color


class FastROILayer(pg.GraphicsObject):
    """
    Отрисовывает тысячи ROI с помощью QPainter.
    Умеет менять группы на лету.
    """
    def __init__(self, max_x, max_y):
        super().__init__()
        self.max_x = max_x
        self.max_y = max_y
        self.hidden_idx = None
        
        self.current_group: ROIGroup | None = None
        self.BUCKET_W = 1000
        self._buckets = {}
        
        self._pen = QPen(Qt.GlobalColor.white, 1)
        self._brush = QBrush(Qt.GlobalColor.transparent)

    def set_group(self, group: ROIGroup):
        """Меняет активный массив ROI и перестраивает пространственный хэш"""
        self.current_group = group
        self.hidden_idx = None
        
        # Настраиваем цвета слоя
        c = group.base_color
        self._pen = QPen(c, 1)
        self._pen.setCosmetic(True)
        
        bg_c = QColor(c)
        bg_c.setAlpha(50)  # Полупрозрачная заливка
        self._brush = QBrush(bg_c)
        
        self.build_index()
        self.update()

    def build_index(self):
        self._buckets.clear()
        if not self.current_group: return
        for r in self.current_group.rois:
            self._add_to_index(r)

    def _add_to_index(self, r):
        b0 = int(r.x) // self.BUCKET_W
        b1 = int(r.x + r.w) // self.BUCKET_W
        for b in range(b0, b1 + 1):
            if b not in self._buckets:
                self._buckets[b] = []
            self._buckets[b].append(r)

    def boundingRect(self):
        return QRectF(0, 0, self.max_x, self.max_y)

    def paint(self, p, *args):
        if not self.current_group: return
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        vb = self.getViewBox()
        if not vb: return
        vr = vb.viewRect()
        
        x_min, x_max = vr.left(), vr.right()
        b0 = int(x_min) // self.BUCKET_W
        b1 = int(x_max) // self.BUCKET_W
        
        visible_rects = []
        seen = set()
        
        for b in range(b0, b1 + 1):
            for r in self._buckets.get(b, []):
                if r.idx != self.hidden_idx and r.idx not in seen:
                    seen.add(r.idx)
                    if r.x <= x_max and r.x + r.w >= x_min and r.y <= vr.bottom() and r.y + r.h >= vr.top():
                        visible_rects.append(r.qrect)
        
        if visible_rects:
            p.setPen(self._pen)
            p.setBrush(self._brush)
            p.drawRects(visible_rects)

    def hide_roi(self, idx):
        self.hidden_idx = idx
        self.update()

    def update_roi_geometry(self, idx, x, y, w, h):
        if not self.current_group: return
        roi = self.current_group.rois[idx]
        
        b0 = int(roi.x) // self.BUCKET_W
        b1 = int(roi.x + roi.w) // self.BUCKET_W
        for b in range(b0, b1 + 1):
            if b in self._buckets and roi in self._buckets[b]:
                self._buckets[b].remove(roi)
                
        roi.x, roi.y, roi.w, roi.h = x, y, w, h
        self._add_to_index(roi)


# ------------------------------------------------------------------------------
# Контроллер логики
# ------------------------------------------------------------------------------
class ROIController:
    """Управляет кликами, активным выделением и переключением групп"""
    def __init__(self, plot_item, roi_layer, groups, status_label):
        self.pi = plot_item
        self.layer = roi_layer
        self.groups = groups
        self.label = status_label
        
        self.active_group = None
        self.active_item = None
        self.active_idx = None

    def switch_group(self, group_index):
        """Переключает активный массив ROI"""
        self.deselect()  # Сбрасываем выделение при переключении
        self.active_group = self.groups[group_index]
        self.layer.set_group(self.active_group)
        self.label.setText(f"Включена: {self.active_group.name} | Нет выделения")

    def on_click(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton or not self.active_group:
            return
            
        pos = self.pi.vb.mapSceneToView(ev.scenePos())
        px, py = pos.x(), pos.y()

        b = int(px) // self.layer.BUCKET_W
        hit = None
        for r in reversed(self.layer._buckets.get(b, [])):
            if r.contains(px, py):
                hit = r
                break

        if hit is None:
            self.deselect()
            return
            
        if self.active_idx != hit.idx:
            self.deselect()
            self.select(hit)

    def select(self, roi):
        self.active_idx = roi.idx
        self.layer.hide_roi(roi.idx)

        # Используем цвет выделения активной группы
        c_active = self.active_group.active_color
        c_hover = QColor(255, 255, 255) # Белый при наведении
        c_handle = QColor(c_active)
        c_handle.setAlpha(200)
        
        pen = pg.mkPen(color=c_active, width=2)
        hoverPen = pg.mkPen(color=c_hover, width=2)
        handlePen = pg.mkPen(color=c_handle, width=1)

        # Создаем наш кастомный ROI (8 ручек, без вращения)
        self.active_item = EightHandleROI(
            pos=[roi.x, roi.y], size=[roi.w, roi.h], 
            pen=pen, hoverPen=hoverPen, handlePen=handlePen
        )
        
        self.active_item.sigRegionChangeFinished.connect(self._sync_roi)
        self.pi.addItem(self.active_item)
        
        self._update_label(roi.idx, roi.x, roi.y, roi.w, roi.h)

    def deselect(self):
        if self.active_item is not None:
            self._sync_roi()
            self.pi.removeItem(self.active_item)
            self.active_item = None
            
        self.active_idx = None
        self.layer.hide_roi(None)
        if self.active_group:
            self.label.setText(f"Включена: {self.active_group.name} | Нет выделения")

    def _sync_roi(self):
        if self.active_item is None or self.active_idx is None: return
        p = self.active_item.pos()
        s = self.active_item.size()
        self.layer.update_roi_geometry(self.active_idx, p.x(), p.y(), s.x(), s.y())
        self._update_label(self.active_idx, p.x(), p.y(), s.x(), s.y())

    def _update_label(self, idx, x, y, w, h):
        self.label.setText(
            f"{self.active_group.name} | Выделен ROI #{idx} (x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f})"
        )

