import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# ==============================================================================
# МАТЕМАТИКА: АЛГОРИТМ ЧАЙКИНА
# ==============================================================================

def chaikin_smooth(pts, iterations=3):
    """ Алгоритм Чайкина для быстрого сглаживания полилиний (сплайнов) """
    if len(pts) < 3:
        return pts
    for _ in range(iterations):
        p0 = pts[:-1]
        p1 = pts[1:]
        q1 = 0.75 * p0 + 0.25 * p1
        q2 = 0.25 * p0 + 0.75 * p1
        
        new_pts = np.empty((len(pts) * 2 - 2, 2), dtype=np.float32)
        new_pts[0::2] = q1
        new_pts[1::2] = q2
        new_pts[0] = pts[0]
        new_pts[-1] = pts[-1]
        pts = new_pts
    return pts


# ==============================================================================
# КАСТОМНЫЕ ИНТЕРАКТИВНЫЕ ROI (ДЛЯ ВЫДЕЛЕНИЯ)
# ==============================================================================

class EightHandleROI(pg.ROI):
    """ Прямоугольный ROI с 8 ручками изменения размера """
    def __init__(self, pos, size, **kwargs):
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


class DraggablePointROI(pg.ROI):
    """ Точка, которая не вытягивается при зуме и не искажается """
    def __init__(self, pos, radius_px=8):
        super().__init__(pos, [0, 0], movable=True, resizable=False, rotatable=False)
        self.radius_px = radius_px

    def _get_local_radiuses(self):
        # Получаем размер 1 пикселя экрана в координатах сцены
        px, py = self.pixelSize()
        if px == 0 or py == 0:
            return 0, 0
        # Умножаем на нужный радиус, чтобы скомпенсировать растяжение
        return self.radius_px * px, self.radius_px * py

    def paint(self, p, opt, widget):
        rx, ry = self._get_local_radiuses()
        if rx == 0: return

        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(pg.mkPen('w', width=1, cosmetic=True)) # Белая рамка
        p.setBrush(pg.mkBrush(255, 0, 0, 200))          # Красная заливка
        
        # Рисуем эллипс, который при проекции на экран станет ровным кругом
        p.drawEllipse(QtCore.QRectF(-rx, -ry, rx * 2, ry * 2))

    def boundingRect(self):
        rx, ry = self._get_local_radiuses()
        return QtCore.QRectF(-rx, -ry, rx * 2, ry * 2)


class SmoothPolyLineROI(pg.PolyLineROI):
    """ Полилайн, который рисует плавную кривую Чайкина """
    def __init__(self, positions, **kwargs):
        kwargs['pen'] = pg.mkPen(None) # Скрываем прямые отрезки
        super().__init__(positions, **kwargs)
        
        self.smooth_path_item = QtWidgets.QGraphicsPathItem(self)
        self.smooth_path_item.setPen(pg.mkPen('r', width=2))
        self.smooth_path_item.setZValue(-1)
        
        self.sigRegionChanged.connect(self.update_smooth_path)
        self.update_smooth_path()

    def update_smooth_path(self):
        handles = self.getHandles()
        if len(handles) < 2: return
        
        pts = np.array([[h.pos().x(), h.pos().y()] for h in handles])
        smooth_pts = chaikin_smooth(pts, iterations=3)
        path = pg.arrayToQPath(smooth_pts[:, 0], smooth_pts[:, 1], connect='finite')
        self.smooth_path_item.setPath(path)


# ==============================================================================
# БЫСТРЫЕ СЛОИ (ДЛЯ НЕВЫДЕЛЕННЫХ ОБЪЕКТОВ)
# ==============================================================================

class SwitchableLayer(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self.data = {}
        self.hidden_indices = set()
        self._path = QtGui.QPainterPath()
        self._polygon = QtGui.QPolygonF()
        self.pen = None
        self.brush = None

    def set_data(self, data_dict):
        self.data = data_dict
        self.rebuild_path()

    def hide_item(self, idx):
        self.hidden_indices.add(idx)
        self.rebuild_path()

    def show_item(self, idx):
        self.hidden_indices.discard(idx)
        self.rebuild_path()

    def paint(self, p, *args):
        if not self._path.isEmpty():
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
            p.setPen(self.pen)
            if self.brush: p.fillPath(self._path, self.brush)
            p.drawPath(self._path)
        elif not self._polygon.isEmpty():
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            p.setPen(self.pen)
            p.drawPoints(self._polygon)

    def boundingRect(self):
        if not self._path.isEmpty(): return self._path.boundingRect()
        if not self._polygon.isEmpty(): return self._polygon.boundingRect()
        return QtCore.QRectF()


class FastRectLayer(SwitchableLayer):
    def __init__(self, pen_color, brush_color):
        super().__init__()
        self.pen = pg.mkPen(pen_color, width=1)
        self.brush = pg.mkBrush(brush_color)

    def rebuild_path(self):
        self._path = QtGui.QPainterPath()
        for idx, (x, y, w, h) in self.data.items():
            if idx not in self.hidden_indices:
                self._path.addRect(QtCore.QRectF(x, y, w, h))
        self.update()

    def check_hit(self, px, py, threshold):
        if not self.data: return None
        keys, rects = list(self.data.keys()), np.array(list(self.data.values()))
        hit_mask = (px >= rects[:, 0]) & (px <= rects[:, 0] + rects[:, 2]) & \
                   (py >= rects[:, 1]) & (py <= rects[:, 1] + rects[:, 3])
        hit_indices = np.where(hit_mask)[0]
        return keys[hit_indices[0]] if len(hit_indices) > 0 else None


class FastCurveLayer(SwitchableLayer):
    def __init__(self):
        super().__init__()
        self.pen = pg.mkPen((0, 200, 255, 180), width=1.5)

    def rebuild_path(self):
        if not self.data:
            self._path = QtGui.QPainterPath()
            self.update()
            return
        xs, ys = [], []
        nan_arr = np.array([np.nan], dtype=np.float32)
        for idx, pts in self.data.items():
            if idx in self.hidden_indices or len(pts) < 2: continue
            smooth_pts = chaikin_smooth(pts, iterations=3)
            xs.extend([smooth_pts[:, 0], nan_arr])
            ys.extend([smooth_pts[:, 1], nan_arr])

        if xs:
            self._path = pg.arrayToQPath(np.concatenate(xs), np.concatenate(ys), connect='finite')
        else:
            self._path = QtGui.QPainterPath()
        self.update()

    def check_hit(self, px, py, threshold):
        for idx, pts in self.data.items():
            if len(pts) < 2: continue
            min_x, min_y, max_x, max_y = pts[:,0].min(), pts[:,1].min(), pts[:,0].max(), pts[:,1].max()
            if px < min_x - threshold or px > max_x + threshold or py < min_y - threshold or py > max_y + threshold:
                continue

            p0, p1 = pts[:-1], pts[1:]
            dx, dy = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
            l2 = dx*dx + dy*dy
            l2[l2 == 0] = 1e-6
            t = np.clip(((px - p0[:, 0]) * dx + (py - p0[:, 1]) * dy) / l2, 0, 1)
            dist_sq = (px - (p0[:, 0] + t * dx))**2 + (py - (p0[:, 1] + t * dy))**2
            
            if np.min(dist_sq) < threshold**2:
                return idx
        return None


class FastPointLayer(SwitchableLayer):
    def __init__(self, size=12):
        super().__init__()
        self.pen = pg.mkPen((255, 255, 0, 220), width=size, cosmetic=True, cap=QtCore.Qt.PenCapStyle.RoundCap)

    def rebuild_path(self):
        pts = [QtCore.QPointF(x, y) for idx, (x, y) in self.data.items() if idx not in self.hidden_indices]
        self._polygon = QtGui.QPolygonF(pts)
        self.update()

    def check_hit(self, px, py, threshold):
        if not self.data: return None
        keys, pts = list(self.data.keys()), np.array(list(self.data.values()))
        dist_sq = (pts[:, 0] - px)**2 + (pts[:, 1] - py)**2
        min_idx_pos = np.argmin(dist_sq)
        if dist_sq[min_idx_pos] < threshold**2:
            return keys[min_idx_pos]
        return None


# ==============================================================================
# КОНТРОЛЛЕР СЦЕНЫ
# ==============================================================================

class SceneController(QtCore.QObject):
    def __init__(self, plot_widget):
        super().__init__()
        self.view = plot_widget.getViewBox()
        
        # 5 слоев по иерархии
        self.context_layer = FastRectLayer(pen_color=(0, 150, 255, 200), brush_color=(0, 100, 255, 40))
        self.rect_layer = FastRectLayer(pen_color=(0, 255, 0, 200), brush_color=(0, 255, 0, 40))
        self.curve_layer = FastCurveLayer()
        self.point_layer = FastPointLayer(size=12)

        self.context_layer.setZValue(5)
        self.rect_layer.setZValue(10)
        self.curve_layer.setZValue(20)
        self.point_layer.setZValue(30)

        self.view.addItem(self.context_layer)
        self.view.addItem(self.rect_layer)
        self.view.addItem(self.curve_layer)
        self.view.addItem(self.point_layer)

        self.active_rois = {} # Строго 1 выделенный элемент
        self.view.scene().sigMouseClicked.connect(self.on_click)

    def load_data(self, contexts, rects, curves, points):
        self.context_layer.set_data(contexts)
        self.rect_layer.set_data(rects)
        self.curve_layer.set_data(curves)
        self.point_layer.set_data(points)

    def get_hit_threshold(self):
        px_x, px_y = self.view.viewPixelSize()
        return max(px_x, px_y) * 6

    def on_click(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton: return

        # Игнорируем клик по уже активным ручкам ROI
        for item in self.view.scene().items(ev.scenePos()):
            if isinstance(item, pg.ROI) or (hasattr(item, 'parentItem') and isinstance(item.parentItem(), pg.ROI)):
                return

        pos = self.view.mapSceneToView(ev.scenePos())
        px, py, thresh = pos.x(), pos.y(), self.get_hit_threshold()

        # Иерархия клика (Точки -> Кривые -> Ректы -> Контексты)
        layer_type, hit_idx = None, None
        
        if (hit := self.point_layer.check_hit(px, py, thresh)) is not None:
            layer_type, hit_idx = 'pt', hit
        elif (hit := self.curve_layer.check_hit(px, py, thresh)) is not None:
            layer_type, hit_idx = 'curve', hit
        elif (hit := self.rect_layer.check_hit(px, py, thresh)) is not None:
            layer_type, hit_idx = 'rect', hit
        elif (hit := self.context_layer.check_hit(px, py, thresh)) is not None:
            layer_type, hit_idx = 'context', hit

        self.deselect_all()
        if hit_idx is not None:
            self.select_item(layer_type, hit_idx)

    def select_item(self, layer_type, idx):
        key = f"{layer_type}_{idx}"
        roi = None

        if layer_type == 'pt':
            self.point_layer.hide_item(idx)
            x, y = self.point_layer.data[idx]
            roi = DraggablePointROI([x, y], radius_px=8) # Точка при выделении становится чуть больше
            roi.setZValue(102)

        elif layer_type == 'curve':
            self.curve_layer.hide_item(idx)
            pts = self.curve_layer.data[idx]
            roi = SmoothPolyLineROI(pts, closed=False, hoverPen='w', handlePen=pg.mkPen('r', width=1))
            roi.setZValue(101)

        elif layer_type == 'rect':
            self.rect_layer.hide_item(idx)
            x, y, w, h = self.rect_layer.data[idx]
            roi = EightHandleROI([x, y], [w, h], pen=pg.mkPen('r', width=2), hoverPen='w')
            roi.setZValue(100)

        elif layer_type == 'context':
            self.context_layer.hide_item(idx)
            x, y, w, h = self.context_layer.data[idx]
            roi = EightHandleROI([x, y], [w, h], pen=pg.mkPen((255, 100, 100), width=3), hoverPen='w')
            roi.setZValue(99)

        if roi:
            self.view.addItem(roi)
            self.active_rois[key] = roi

    def deselect_all(self):
        for key, roi in list(self.active_rois.items()):
            layer_type, idx = key.split('_')
            idx = int(idx)

            if layer_type == 'context':
                self.context_layer.data[idx] = [roi.pos().x(), roi.pos().y(), roi.size().x(), roi.size().y()]
                self.context_layer.show_item(idx)
                
            elif layer_type == 'rect':
                self.rect_layer.data[idx] = [roi.pos().x(), roi.pos().y(), roi.size().x(), roi.size().y()]
                self.rect_layer.show_item(idx)
            
            elif layer_type == 'curve':
                roi_pos = roi.pos()
                new_pts = np.array([[h.pos().x() + roi_pos.x(), h.pos().y() + roi_pos.y()] for h in roi.getHandles()])
                self.curve_layer.data[idx] = new_pts
                self.curve_layer.show_item(idx)
            
            elif layer_type == 'pt':
                pos = roi.pos()
                self.point_layer.data[idx] = [pos.x(), pos.y()]
                self.point_layer.show_item(idx)

            self.view.removeItem(roi)
            
        self.active_rois.clear()


# ==============================================================================
# ЗАПУСК
# ==============================================================================

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph: 5 Layers + Chaikin + 8-Handles")
        self.resize(1200, 900)
        pg.setConfigOptions(imageAxisOrder='row-major')

        self.plot = pg.PlotWidget()
        self.setCentralWidget(self.plot)
        self.view = self.plot.getViewBox()

        # Фон
        img_data = np.random.normal(size=(200, 200))
        img_data = pg.gaussianFilter(img_data, (3, 3))
        self.img_item = pg.ImageItem(img_data)
        self.img_item.setZValue(-1)
        self.img_item.setLookupTable(pg.colormap.get('plasma').getLookupTable())
        self.view.addItem(self.img_item)

        print("Генерация данных (Контексты -> Зеленые Ректы -> Кривые и Точки)...")
        contexts, rects, curves, points = {}, {}, {}, {}
        c_idx, r_idx, v_idx, p_idx = 0, 0, 0, 0

        for _ in range(40):
            cx, cy = np.random.uniform(10, 150, 2)
            cw, ch = np.random.uniform(30, 50, 2)
            contexts[c_idx] = [cx, cy, cw, ch]

            for _ in range(np.random.randint(3, 7)):
                rw, rh = np.random.uniform(5, 12, 2)
                rx = np.random.uniform(cx, cx + cw - rw)
                ry = np.random.uniform(cy, cy + ch - rh)
                rects[r_idx] = [rx, ry, rw, rh]

                num_nodes = np.random.randint(4, 7)
                curve_x = np.linspace(rx, rx + rw, num_nodes)
                curve_y = ry + rh/2 + np.random.uniform(-rh/3, rh/3, num_nodes)
                curves[v_idx] = np.column_stack([curve_x, curve_y])

                px, py = np.random.uniform(rx, rx + rw), np.random.uniform(ry, ry + rh)
                points[p_idx] = [px, py]

                r_idx += 1; v_idx += 1; p_idx += 1
            c_idx += 1

        self.controller = SceneController(self.plot)
        self.controller.load_data(contexts, rects, curves, points)
        self.plot.autoRange()

if __name__ == '__main__':
    app = pg.mkQApp()
    win = App()
    win.show()
    sys.exit(app.exec())