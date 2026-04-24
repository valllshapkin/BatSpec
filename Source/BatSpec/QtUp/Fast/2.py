import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import colorsys

# ==============================================================================
# МАТЕМАТИКА И АЛГОРИТМЫ
# ==============================================================================

def chaikin_smooth(pts, iterations=3):
    if len(pts) < 3: return pts
    for _ in range(iterations):
        p0, p1 = pts[:-1], pts[1:]
        q1 = 0.75 * p0 + 0.25 * p1
        q2 = 0.25 * p0 + 0.75 * p1
        new_pts = np.empty((len(pts) * 2 - 2, 2), dtype=np.float32)
        new_pts[0::2], new_pts[1::2] = q1, q2
        new_pts[0], new_pts[-1] = pts[0], pts[-1]
        pts = new_pts
    return pts

# ==============================================================================
# ИЕРАРХИЧЕСКАЯ БАЗА ДАННЫХ (SCENE GRAPH)
# ==============================================================================

class DataStore:
    """ Единое хранилище всех объектов с поддержкой иерархии """
    def __init__(self):
        self.contexts = {} # id: {'geom': [x,y,w,h], 'hue': 0-1.0}
        self.rects = {}    # id: {'parent': ctx_id, 'geom': [x,y,w,h]}
        self.curves = {}   # id: {'parent': rect_id, 'geom': np.array}
        self.points = {}   # id: {'parent': rect_id, 'geom': [x,y]}
        self._next_id = 1
        self.display_mode = 'type' # 'type' или 'parent'

    def get_id(self):
        self._next_id += 1
        return self._next_id

    def add_context(self, geom):
        cid = self.get_id()
        self.contexts[cid] = {'geom': geom, 'hue': np.random.random()}
        return cid

    def add_rect(self, parent_id, geom):
        rid = self.get_id()
        self.rects[rid] = {'parent': parent_id, 'geom': geom}
        return rid

    def add_curve(self, parent_id, geom):
        vid = self.get_id()
        self.curves[vid] = {'parent': parent_id, 'geom': geom}
        return vid

    def add_point(self, parent_id, geom):
        pid = self.get_id()
        self.points[pid] = {'parent': parent_id, 'geom': geom}
        return pid

    def delete_item(self, layer_type, idx):
        """ Каскадное удаление """
        if layer_type == 'context' and idx in self.contexts:
            # Находим и удаляем всех детей
            children_rects = [r for r, d in self.rects.items() if d['parent'] == idx]
            for r in children_rects: self.delete_item('rect', r)
            del self.contexts[idx]
            
        elif layer_type == 'rect' and idx in self.rects:
            children_cv = [c for c, d in self.curves.items() if d['parent'] == idx]
            children_pt = [p for p, d in self.points.items() if d['parent'] == idx]
            for c in children_cv: del self.curves[c]
            for p in children_pt: del self.points[p]
            del self.rects[idx]
            
        elif layer_type == 'curve' and idx in self.curves:
            del self.curves[idx]
            
        elif layer_type == 'pt' and idx in self.points:
            del self.points[idx]

    def get_colors(self):
        """ Рассчитывает цвета элементов в зависимости от выбранного режима """
        colors = {'context': {}, 'rect': {}, 'curve': {}, 'pt': {}}
        
        for cid, ctx in self.contexts.items():
            if self.display_mode == 'type':
                colors['context'][cid] = (0, 150, 255) # Синий
                base_c = (0, 255, 0)
            else:
                r, g, b = [int(x*255) for x in colorsys.hsv_to_rgb(ctx['hue'], 0.8, 1.0)]
                colors['context'][cid] = (r, g, b)
                base_c = (r, g, b)

            # Красим детей
            for rid, r_data in self.rects.items():
                if r_data['parent'] == cid:
                    colors['rect'][rid] = (0, 255, 0) if self.display_mode == 'type' else base_c
                    
                    for cv_id, cv_data in self.curves.items():
                        if cv_data['parent'] == rid:
                            colors['curve'][cv_id] = (0, 200, 255) if self.display_mode == 'type' else base_c
                    
                    for pt_id, pt_data in self.points.items():
                        if pt_data['parent'] == rid:
                            colors['pt'][pt_id] = (255, 255, 0) if self.display_mode == 'type' else base_c
        return colors

# ==============================================================================
# КАСТОМНЫЕ ИНТЕРАКТИВНЫЕ ROI
# ==============================================================================

class EightHandleROI(pg.ROI):
    def __init__(self, pos, size, pen):
        super().__init__(pos, size, movable=True, resizable=True, rotatable=False, pen=pen, hoverPen='w')
        for x, y in [(0,0), (1,1), (0,1), (1,0), (0.5,0), (0.5,1), (0,0.5), (1,0.5)]:
            self.addScaleHandle([x, y], [1-x, 1-y])

class DraggablePointROI(pg.ROI):
    def __init__(self, pos, pen):
        super().__init__(pos, [0, 0], movable=True, resizable=False, rotatable=False)
        self.pen = pen

    def paint(self, p, opt, widget):
        px, py = self.pixelSize()
        rx, ry = 8 * px, 8 * py
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(pg.mkPen('w', width=1, cosmetic=True))
        c = self.pen.color()
        p.setBrush(pg.mkBrush(c.red(), c.green(), c.blue(), 200))
        p.drawEllipse(QtCore.QRectF(-rx, -ry, rx * 2, ry * 2))

    def boundingRect(self):
        px, py = self.pixelSize()
        rx, ry = 8 * px, 8 * py
        return QtCore.QRectF(-rx, -ry, rx * 2, ry * 2)

class SmoothPolyLineROI(pg.PolyLineROI):
    def __init__(self, positions, pen):
        super().__init__(positions, closed=False, pen=pg.mkPen(None), hoverPen='w')
        self.smooth_path_item = QtWidgets.QGraphicsPathItem(self)
        self.smooth_path_item.setPen(pen)
        self.smooth_path_item.setZValue(-1)
        self.sigRegionChanged.connect(self.update_smooth)
        self.update_smooth()

    def update_smooth(self):
        handles = self.getHandles()
        if len(handles) < 2: return
        pts = np.array([[h.pos().x(), h.pos().y()] for h in handles])
        smooth_pts = chaikin_smooth(pts, iterations=3)
        self.smooth_path_item.setPath(pg.arrayToQPath(smooth_pts[:, 0], smooth_pts[:, 1], connect='finite'))

# ==============================================================================
# БЫСТРЫЕ СЛОИ (С ГРУППИРОВКОЙ ПО ЦВЕТУ)
# ==============================================================================

class FastColorLayer(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self.data = {}
        self.colors = {}
        self.hidden_indices = set()
        self.paths_by_color = {} # {(r,g,b): QPainterPath}

    def update_data(self, data, colors):
        self.data = data
        self.colors = colors
        self.rebuild()

    def hide_item(self, idx):
        self.hidden_indices.add(idx)
        self.rebuild()

    def show_item(self, idx):
        self.hidden_indices.discard(idx)
        self.rebuild()

    def rebuild(self): pass

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        for color, path in self.paths_by_color.items():
            if path.isEmpty(): continue
            pen = pg.mkPen(color, width=self.pen_width)
            brush = pg.mkBrush(*color, self.alpha) if hasattr(self, 'alpha') else None
            p.setPen(pen)
            if brush: p.fillPath(path, brush)
            p.drawPath(path)

    def boundingRect(self):
        br = QtCore.QRectF()
        for path in self.paths_by_color.values():
            br = br.united(path.boundingRect())
        return br

class FastRectLayer(FastColorLayer):
    def __init__(self, alpha=40, pen_width=1):
        super().__init__()
        self.alpha = alpha
        self.pen_width = pen_width

    def rebuild(self):
        self.paths_by_color = {}
        for idx, (x, y, w, h) in self.data.items():
            if idx in self.hidden_indices: continue
            c = self.colors.get(idx, (255,255,255))
            if c not in self.paths_by_color: self.paths_by_color[c] = QtGui.QPainterPath()
            self.paths_by_color[c].addRect(QtCore.QRectF(x, y, w, h))
        self.update()

class FastCurveLayer(FastColorLayer):
    def __init__(self):
        super().__init__()
        self.pen_width = 1.5

    def rebuild(self):
        self.paths_by_color = {}
        xs_ys_by_color = {}
        nan_arr = np.array([np.nan], dtype=np.float32)

        for idx, pts in self.data.items():
            if idx in self.hidden_indices or len(pts) < 2: continue
            c = self.colors.get(idx, (255,255,255))
            if c not in xs_ys_by_color: xs_ys_by_color[c] = ([], [])
            
            s_pts = chaikin_smooth(pts, iterations=2)
            xs_ys_by_color[c][0].extend([s_pts[:, 0], nan_arr])
            xs_ys_by_color[c][1].extend([s_pts[:, 1], nan_arr])

        for c, (xs, ys) in xs_ys_by_color.items():
            self.paths_by_color[c] = pg.arrayToQPath(np.concatenate(xs), np.concatenate(ys), connect='finite')
        self.update()

class FastPointLayer(FastColorLayer):
    def __init__(self):
        super().__init__()

    def rebuild(self):
        self.paths_by_color = {}
        for idx, (x, y) in self.data.items():
            if idx in self.hidden_indices: continue
            c = self.colors.get(idx, (255,255,255))
            if c not in self.paths_by_color: self.paths_by_color[c] = QtGui.QPolygonF()
            self.paths_by_color[c].append(QtCore.QPointF(x, y))
        self.update()

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for color, poly in self.paths_by_color.items():
            if poly.isEmpty(): continue
            p.setPen(pg.mkPen(color, width=12, cosmetic=True, cap=QtCore.Qt.PenCapStyle.RoundCap))
            p.drawPoints(poly)

# ==============================================================================
# КОНТРОЛЛЕР СЦЕНЫ
# ==============================================================================

class SceneController(QtCore.QObject):
    # Сигнал для синхронизации с деревом
    itemSelected = QtCore.Signal(str, int) 
    dataChanged = QtCore.Signal()

    def __init__(self, plot_widget, store):
        super().__init__()
        self.view = plot_widget.getViewBox()
        self.store = store
        
        self.layers = {
            'context': FastRectLayer(alpha=30, pen_width=1),
            'rect': FastRectLayer(alpha=50, pen_width=1),
            'curve': FastCurveLayer(),
            'pt': FastPointLayer()
        }
        self.layers['context'].setZValue(5)
        self.layers['rect'].setZValue(10)
        self.layers['curve'].setZValue(20)
        self.layers['pt'].setZValue(30)

        for l in self.layers.values(): self.view.addItem(l)

        self.active_roi = None
        self.active_type = None
        self.active_idx = None
        
        self.view.scene().sigMouseClicked.connect(self.on_click)

    def refresh_all(self):
        colors = self.store.get_colors()
        self.layers['context'].update_data({k: v['geom'] for k,v in self.store.contexts.items()}, colors['context'])
        self.layers['rect'].update_data({k: v['geom'] for k,v in self.store.rects.items()}, colors['rect'])
        self.layers['curve'].update_data({k: v['geom'] for k,v in self.store.curves.items()}, colors['curve'])
        self.layers['pt'].update_data({k: v['geom'] for k,v in self.store.points.items()}, colors['pt'])
        self.dataChanged.emit()

    def get_hit_threshold(self):
        px, py = self.view.viewPixelSize()
        return max(px, py) * 6

    def on_click(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton: return
        for item in self.view.scene().items(ev.scenePos()):
            if isinstance(item, pg.ROI) or (hasattr(item, 'parentItem') and isinstance(item.parentItem(), pg.ROI)): return

        pos = self.view.mapSceneToView(ev.scenePos())
        px, py, thresh = pos.x(), pos.y(), self.get_hit_threshold()

        # Хиттесты
        hit_type, hit_idx = None, None
        
        # Ручные проверки для каждого типа (т.к. геометрия хранится чуть по-разному)
        # Точки
        pts = np.array([d['geom'] for d in self.store.points.values()]) if self.store.points else []
        if len(pts) > 0:
            keys = list(self.store.points.keys())
            ds = (pts[:,0]-px)**2 + (pts[:,1]-py)**2
            if np.min(ds) < thresh**2: hit_type, hit_idx = 'pt', keys[np.argmin(ds)]
            
        # Кривые
        if not hit_idx:
            for k, v in self.store.curves.items():
                pts = v['geom']
                if len(pts) < 2: continue
                p0, p1 = pts[:-1], pts[1:]
                l2 = (p1[:,0]-p0[:,0])**2 + (p1[:,1]-p0[:,1])**2
                l2[l2==0]=1e-6
                t = np.clip(((px-p0[:,0])*(p1[:,0]-p0[:,0]) + (py-p0[:,1])*(p1[:,1]-p0[:,1]))/l2, 0, 1)
                if np.min((px - (p0[:,0]+t*(p1[:,0]-p0[:,0])))**2 + (py - (p0[:,1]+t*(p1[:,1]-p0[:,1])))**2) < thresh**2:
                    hit_type, hit_idx = 'curve', k; break

        # Ректы
        if not hit_idx:
            for ltype, data_dict in [('rect', self.store.rects), ('context', self.store.contexts)]:
                rects = np.array([d['geom'] for d in data_dict.values()]) if data_dict else []
                if len(rects) > 0:
                    mask = (px>=rects[:,0])&(px<=rects[:,0]+rects[:,2])&(py>=rects[:,1])&(py<=rects[:,1]+rects[:,3])
                    idx = np.where(mask)[0]
                    if len(idx)>0: hit_type, hit_idx = ltype, list(data_dict.keys())[idx[0]]; break

        self.select_item(hit_type, hit_idx)

    def select_item(self, layer_type, idx):
        self.deselect()
        if not layer_type or not idx: 
            self.itemSelected.emit("", -1) # Снять выделение в дереве
            return

        self.active_type = layer_type
        self.active_idx = idx
        self.layers[layer_type].hide_item(idx)

        colors = self.store.get_colors()
        c = colors[layer_type].get(idx, (255,0,0))
        pen = pg.mkPen(c, width=3)

        if layer_type == 'pt':
            geom = self.store.points[idx]['geom']
            self.active_roi = DraggablePointROI(geom, pen)
        elif layer_type == 'curve':
            geom = self.store.curves[idx]['geom']
            self.active_roi = SmoothPolyLineROI(geom, pen)
        else:
            geom = self.store.rects[idx]['geom'] if layer_type=='rect' else self.store.contexts[idx]['geom']
            self.active_roi = EightHandleROI([geom[0], geom[1]], [geom[2], geom[3]], pen)

        self.active_roi.setZValue(100)
        self.view.addItem(self.active_roi)
        self.itemSelected.emit(layer_type, idx) # Сигнал для дерева

    def deselect(self):
        if not self.active_roi: return
        t, idx = self.active_type, self.active_idx

        # Сохраняем новые координаты
        if t in ['context', 'rect']:
            geom = [self.active_roi.pos().x(), self.active_roi.pos().y(), self.active_roi.size().x(), self.active_roi.size().y()]
            if t == 'context': self.store.contexts[idx]['geom'] = geom
            else: self.store.rects[idx]['geom'] = geom
        elif t == 'curve':
            geom = np.array([[h.pos().x() + self.active_roi.pos().x(), h.pos().y() + self.active_roi.pos().y()] for h in self.active_roi.getHandles()])
            self.store.curves[idx]['geom'] = geom
        elif t == 'pt':
            self.store.points[idx]['geom'] = [self.active_roi.pos().x(), self.active_roi.pos().y()]

        self.layers[t].show_item(idx)
        self.view.removeItem(self.active_roi)
        self.active_roi = None
        self.active_type = None
        self.active_idx = None


# ==============================================================================
# MAIN GUI (TREE + TOOLBAR)
# ==============================================================================

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph: Hierarchical Editor")
        self.resize(1300, 800)

        pg.setConfigOptions(imageAxisOrder='row-major')

        # Layout
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Левая панель
        left_panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(left_panel)
        splitter.addWidget(left_panel)

        # Toolbar
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Цвет: По Типу", "Цвет: По Родителю"])
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        layout.addWidget(self.mode_combo)

        btn_layout = QtWidgets.QGridLayout()
        self.btn_ctx = QtWidgets.QPushButton("+ Синий (Контекст)")
        self.btn_rect = QtWidgets.QPushButton("+ Зеленый (Рект)")
        self.btn_curve = QtWidgets.QPushButton("+ Кривая")
        self.btn_pt = QtWidgets.QPushButton("+ Точка")
        self.btn_del = QtWidgets.QPushButton("УДАЛИТЬ")
        self.btn_del.setStyleSheet("background-color: #ffcccc")

        btn_layout.addWidget(self.btn_ctx, 0, 0)
        btn_layout.addWidget(self.btn_rect, 0, 1)
        btn_layout.addWidget(self.btn_curve, 1, 0)
        btn_layout.addWidget(self.btn_pt, 1, 1)
        layout.addLayout(btn_layout)
        layout.addWidget(self.btn_del)

        # Tree Widget
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabel("Иерархия объектов")
        layout.addWidget(self.tree)

        # Правая панель (График)
        self.plot = pg.PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setSizes([300, 1000])

        # Инициализация ядра
        self.store = DataStore()
        self.controller = SceneController(self.plot, self.store)
        
        # Сигналы
        self.controller.itemSelected.connect(self.sync_tree_selection)
        self.controller.dataChanged.connect(self.rebuild_tree)
        self.tree.itemSelectionChanged.connect(self.sync_plot_selection)
        
        self.btn_ctx.clicked.connect(lambda: self.create_item('context'))
        self.btn_rect.clicked.connect(lambda: self.create_item('rect'))
        self.btn_curve.clicked.connect(lambda: self.create_item('curve'))
        self.btn_pt.clicked.connect(lambda: self.create_item('pt'))
        self.btn_del.clicked.connect(self.delete_selected)

        self.generate_demo_data()
        self.controller.refresh_all()
        self.plot.autoRange()

    def generate_demo_data(self):
        for _ in range(5):
            cx, cy = np.random.uniform(0, 100, 2)
            cid = self.store.add_context([cx, cy, 40, 40])
            for _ in range(3):
                rx, ry = np.random.uniform(cx, cx+20, 2)
                rid = self.store.add_rect(cid, [rx, ry, 15, 15])
                x = np.linspace(rx, rx+10, 4)
                self.store.add_curve(rid, np.column_stack([x, ry+5+np.sin(x)]))
                self.store.add_point(rid, [rx+5, ry+5])

    def rebuild_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        
        # Собираем иерархию
        for cid, ctx in self.store.contexts.items():
            c_item = QtWidgets.QTreeWidgetItem(self.tree, [f"Context #{cid}"])
            c_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ('context', cid))
            
            for rid, rdata in self.store.rects.items():
                if rdata['parent'] == cid:
                    r_item = QtWidgets.QTreeWidgetItem(c_item, [f"Rect #{rid}"])
                    r_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ('rect', rid))
                    
                    for vid, vdata in self.store.curves.items():
                        if vdata['parent'] == rid:
                            v_item = QtWidgets.QTreeWidgetItem(r_item, [f"Curve #{vid}"])
                            v_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ('curve', vid))
                    
                    for pid, pdata in self.store.points.items():
                        if pdata['parent'] == rid:
                            p_item = QtWidgets.QTreeWidgetItem(r_item, [f"Point #{pid}"])
                            p_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ('pt', pid))
                            
        self.tree.expandAll()
        self.tree.blockSignals(False)

    def sync_tree_selection(self, ltype, idx):
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        if ltype and idx != -1:
            it = QtWidgets.QTreeWidgetItemIterator(self.tree)
            while it.value():
                data = it.value().data(0, QtCore.Qt.ItemDataRole.UserRole)
                if data == (ltype, idx):
                    it.value().setSelected(True)
                    self.tree.scrollToItem(it.value())
                    break
                it += 1
        self.tree.blockSignals(False)

    def sync_plot_selection(self):
        sel = self.tree.selectedItems()
        if sel:
            ltype, idx = sel[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
            self.controller.select_item(ltype, idx)
        else:
            self.controller.select_item(None, None)

    def change_mode(self, idx):
        self.store.display_mode = 'type' if idx == 0 else 'parent'
        self.controller.refresh_all()
        # Если что-то выделено - перевыделим, чтобы обновить цвет рамки
        if self.controller.active_type:
            self.controller.select_item(self.controller.active_type, self.controller.active_idx)

    def create_item(self, ltype):
        sel = self.tree.selectedItems()
        sel_data = sel[0].data(0, QtCore.Qt.ItemDataRole.UserRole) if sel else (None, None)
        ptype, pidx = sel_data

        if ltype == 'context':
            # Спавним в центре экрана
            vr = self.plot.getViewBox().viewRect()
            self.store.add_context([vr.center().x()-10, vr.center().y()-10, 20, 20])
        elif ltype == 'rect':
            if ptype != 'context': return QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите Контекст (синий) для добавления Ректа")
            geom = self.store.contexts[pidx]['geom']
            self.store.add_rect(pidx, [geom[0]+2, geom[1]+2, 10, 10])
        elif ltype in ['curve', 'pt']:
            if ptype != 'rect': return QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите Зеленый Рект для добавления")
            geom = self.store.rects[pidx]['geom']
            if ltype == 'curve':
                x = np.linspace(geom[0], geom[0]+geom[2], 4)
                self.store.add_curve(pidx, np.column_stack([x, np.full_like(x, geom[1]+geom[3]/2)]))
            else:
                self.store.add_point(pidx, [geom[0]+geom[2]/2, geom[1]+geom[3]/2])
        
        self.controller.refresh_all()

    def delete_selected(self):
        if not self.controller.active_type: return
        self.store.delete_item(self.controller.active_type, self.controller.active_idx)
        self.controller.deselect() # Убираем с экрана
        self.controller.refresh_all()

if __name__ == '__main__':
    app = pg.mkQApp()
    win = App()
    win.show()
    sys.exit(app.exec())