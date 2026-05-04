import numpy as np
import pyqtgraph as pg
import zlib
from pyqtgraph.Qt import QtCore, QtGui
from PySide6.QtCore import Qt

from BatAnnotation.QtModels import QtRecording
from BatSpec.App.中Annotation.中Spectrogram.CustomROIs import PointROI, SmoothPolyLineROI
from BatSpec.App.中Annotation.中Spectrogram.Utils import point_line_distance, chaikin_smooth

class FastAnnotationLayer(pg.GraphicsObject):
    def __init__(self, recording: QtRecording):
        super().__init__()
        self.recording = recording
        self.active_model = None
        self.active_sub_type = None 
        
        self._color_cache = {}
        
        self.pen_pt = pg.mkPen((255, 255, 0), width=2)
        self.brush_pt = pg.mkBrush(255, 255, 0, 150)
        self.pen_curve = pg.mkPen((255, 100, 255), width=2)

        self.recording.changed.connect(self.update)

    def get_seq_colors(self, seq_id: str):
        if seq_id not in self._color_cache:
            h = zlib.adler32(seq_id.encode('utf-8')) % 360
            color = QtGui.QColor.fromHsv(h, 200, 255)
            
            seq_pen = pg.mkPen(color, width=1)
            color.setAlpha(30)
            seq_brush = pg.mkBrush(color)
            
            color.setAlpha(80)
            call_brush = pg.mkBrush(color)
            
            self._color_cache[seq_id] = (seq_pen, seq_brush, call_brush)
        return self._color_cache[seq_id]

    def set_active(self, model, sub_type):
        self.active_model = model
        self.active_sub_type = sub_type
        self.update()

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        px = self.pixelWidth() * 4 if self.pixelWidth() else 1
        py = self.pixelHeight() * 4 if self.pixelHeight() else 1

        for seq in self.recording.sequences:
            seq_id = seq.sequence_id if seq.sequence_id else "default"
            seq_pen, seq_brush, call_brush = self.get_seq_colors(seq_id)
            
            if seq != self.active_model or self.active_sub_type:
                p.setPen(seq_pen)
                p.setBrush(seq_brush)
                p.drawRect(QtCore.QRectF(seq.t_start_ms, seq.f_min_khz, seq.t_end_ms - seq.t_start_ms, seq.f_max_khz - seq.f_min_khz))
                
            for call in seq.calls:
                is_active_call = (call == self.active_model and not self.active_sub_type)
                if not is_active_call:
                    p.setPen(seq_pen)
                    p.setBrush(call_brush)
                    p.drawRect(QtCore.QRectF(call.t_start_ms, call.f_min_khz, call.t_end_ms - call.t_start_ms, call.f_max_khz - call.f_min_khz))
                    
                p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
                
                if call.peak_khz is not None and call.peak_ms is not None:
                    if not (call == self.active_model and self.active_sub_type == "fmaxe"):
                        p.setPen(self.pen_pt)
                        p.setBrush(self.brush_pt)
                        p.drawEllipse(QtCore.QRectF(call.peak_ms - px, call.peak_khz - py, px * 2, py * 2))

                if call.signal_curves and "main" in call.signal_curves:
                    if not (call == self.active_model and self.active_sub_type == "curve"):
                        pts = call.signal_curves["main"]
                        if len(pts) >= 2:
                            smooth_pts = chaikin_smooth(pts, 2)
                            path = pg.arrayToQPath(smooth_pts[:, 0], smooth_pts[:, 1], connect='finite')
                            p.setPen(self.pen_curve)
                            p.setBrush(QtCore.Qt.BrushStyle.NoBrush) 
                            p.drawPath(path)
                            
                p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 999999, 300)


class SpectrogramWidget(pg.PlotWidget):
    itemClicked = QtCore.Signal(str, object) 
    dataModified = QtCore.Signal() 

    def __init__(self, recording: QtRecording):
        super().__init__()
        self.recording = recording
        self.view = self.getViewBox()
        
        self.setLabel('bottom', 'Time', units='ms')
        self.setLabel('left', 'Frequency', units='kHz')
        
        self.setup_dummy_background()
        
        self.fast_layer = FastAnnotationLayer(self.recording)
        self.view.addItem(self.fast_layer)
        
        self.active_roi = None
        self.active_model = None
        self.active_type = None
        
        self._updating_from_code = False 
        self._updating_from_roi = False  
        
        self._prev_roi_pos = None
        self._prev_roi_size = None
        
        self.scene().sigMouseClicked.connect(self.on_mouse_click)

    def setup_dummy_background(self):
        img = pg.ImageItem()
        noise = np.random.normal(size=(10000, 300), loc=50, scale=20).astype(np.uint8)
        img.setImage(noise)
        img.setRect(QtCore.QRectF(0, 0, 10000, 300))
        img.setColorMap(pg.colormap.get('magma'))
        self.view.addItem(img)

    def set_selection(self, typ: str, model):
        self._updating_from_code = True
        
        if self.active_model and hasattr(self.active_model, 'changed'):
            try: self.active_model.changed.disconnect(self.on_model_changed_externally)
            except Exception: pass

        if self.active_roi:
            try: self.active_roi.sigRegionChanged.disconnect()
            except Exception: pass
            self.active_roi.hide() 
            self.view.removeItem(self.active_roi)
            self.active_roi.deleteLater() 
            self.active_roi = None
            
        sub_type = typ if typ in ["fmaxe", "curve"] else None
        self.fast_layer.set_active(model, sub_type)
        self.active_model = model
        self.active_type = typ
        
        if not model:
            self._updating_from_code = False
            return

        self.active_model.changed.connect(self.on_model_changed_externally)

        if typ == "rec":
            max_time_ms = (self.active_model.duration_s * 1000) if self.active_model.duration_s else 10000
            max_freq_khz = (self.active_model.sample_rate_hz / 2000) if self.active_model.sample_rate_hz else 150
            self.view.setRange(xRange=[0, max_time_ms], yRange=[0, max_freq_khz], padding=0.0)
            
        elif typ in ["seq", "call"]:
            seq = model.parent() if typ == "call" else model
            seq_id = seq.sequence_id if seq and hasattr(seq, 'sequence_id') else "default"
            h = zlib.adler32(seq_id.encode('utf-8')) % 360
            
            color = QtGui.QColor.fromHsv(h, 255, 255)
            pen = pg.mkPen(color, width=2)
            
            pos = (model.t_start_ms, model.f_min_khz)
            size = (max(1, model.t_end_ms - model.t_start_ms), max(1, model.f_max_khz - model.f_min_khz))
            
            self.active_roi = pg.ROI(pos, size, pen=pen, hoverPen=pg.mkPen('w', width=3), rotatable=False, resizable=True, movable=True)
            for sx, sy in [(0,0), (1,1), (0,1), (1,0), (0.5,0), (0.5,1), (0,0.5), (1,0.5)]:
                self.active_roi.addScaleHandle([sx, sy], [1-sx, 1-sy])
            
        elif typ == "fmaxe":
            cx = model.peak_ms if model.peak_ms else model.t_start_ms + (model.t_end_ms - model.t_start_ms)/2
            cy = model.peak_khz if model.peak_khz else model.f_min_khz + (model.f_max_khz - model.f_min_khz)/2
            self.active_roi = PointROI([cx, cy], pg.mkPen('y', width=2), pg.mkBrush(255, 255, 0, 150))

        elif typ == "curve":
            pts = model.signal_curves.get("main", []) if model.signal_curves else []
            if not pts:
                pts = [[model.t_start_ms, model.f_min_khz + (model.f_max_khz - model.f_min_khz)/2],
                       [model.t_end_ms, model.f_min_khz + (model.f_max_khz - model.f_min_khz)/2]]
            self.active_roi = SmoothPolyLineROI(pts, pg.mkPen((255, 100, 255), width=3))

        if self.active_roi:
            self._prev_roi_pos = self.active_roi.pos()
            self._prev_roi_size = self.active_roi.size()
            
            self.active_roi.setZValue(100)
            self.active_roi.sigRegionChanged.connect(self.on_roi_changed)
            self.view.addItem(self.active_roi)
            
            if typ != "rec":
                self.view.autoRange(items=[self.active_roi], padding=0.2)
                
        self._updating_from_code = False

    def on_model_changed_externally(self):
        if self._updating_from_roi or self._updating_from_code or not self.active_roi: return
        
        self._updating_from_code = True
        if self.active_type in ["seq", "call"]:
            self.active_roi.setPos((self.active_model.t_start_ms, self.active_model.f_min_khz))
            self.active_roi.setSize((self.active_model.t_end_ms - self.active_model.t_start_ms, self.active_model.f_max_khz - self.active_model.f_min_khz))
            self._prev_roi_pos = self.active_roi.pos()
            self._prev_roi_size = self.active_roi.size()
            
        elif self.active_type == "fmaxe":
            if self.active_model.peak_ms and self.active_model.peak_khz:
                self.active_roi.setPos((self.active_model.peak_ms, self.active_model.peak_khz))
        self._updating_from_code = False

    def on_roi_changed(self):
        if self._updating_from_code or not self.active_model: return
        self._updating_from_roi = True 
        
        if self.active_type in ["seq", "call"]:
            pos = self.active_roi.pos()
            size = self.active_roi.size()
            dx = pos.x() - self._prev_roi_pos.x()
            dy = pos.y() - self._prev_roi_pos.y()
            is_translating = (size.x() == self._prev_roi_size.x() and size.y() == self._prev_roi_size.y())
            
            self.active_model.t_start_ms = pos.x()
            self.active_model.f_min_khz = pos.y()
            self.active_model.t_end_ms = pos.x() + size.x()
            self.active_model.f_max_khz = pos.y() + size.y()
            
            if is_translating and (dx != 0 or dy != 0):
                if self.active_type == "seq":
                    for call in self.active_model.calls:
                        self._shift_call(call, dx, dy, shift_bounds=True)
                elif self.active_type == "call":
                    self._shift_call(self.active_model, dx, dy, shift_bounds=False)
                    
            self._prev_roi_pos = pos
            self._prev_roi_size = size
            
        elif self.active_type == "fmaxe":
            pos = self.active_roi.pos()
            self.active_model.peak_ms = pos.x()
            self.active_model.peak_khz = pos.y()
            
        elif self.active_type == "curve":
            pts = self.active_roi.get_raw_points()
            if not self.active_model.signal_curves:
                self.active_model.signal_curves = {}
            new_curves = dict(self.active_model.signal_curves)
            new_curves["main"] = pts
            self.active_model.signal_curves = new_curves

        self.dataModified.emit() 
        self._updating_from_roi = False
        self.fast_layer.update()

    def _shift_call(self, call, dx, dy, shift_bounds=True):
        if shift_bounds:
            call.t_start_ms += dx
            call.t_end_ms += dx
            call.f_min_khz += dy
            call.f_max_khz += dy
            
        if call.peak_ms is not None: call.peak_ms += dx
        if call.peak_khz is not None: call.peak_khz += dy
            
        if call.signal_curves and "main" in call.signal_curves:
            new_curves = dict(call.signal_curves)
            new_curves["main"] = [[pt[0] + dx, pt[1] + dy] for pt in new_curves["main"]]
            call.signal_curves = new_curves

    def on_mouse_click(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton: return
        
        for item in self.scene().items(ev.scenePos()):
            if type(item).__name__ == "Handle":
                return

        scene_pos = ev.scenePos()
        sx, sy = scene_pos.x(), scene_pos.y()
        px, py = self.view.mapSceneToView(scene_pos).x(), self.view.mapSceneToView(scene_pos).y()

        THRESH2_POINT = 36 
        THRESH2_LINE = 25 

        for seq in self.recording.sequences:
            for call in seq.calls:
                if call.peak_khz is not None and call.peak_ms is not None:
                    pt_scene = self.view.mapViewToScene(QtCore.QPointF(call.peak_ms, call.peak_khz))
                    if pt_scene:
                        if (pt_scene.x() - sx)**2 + (pt_scene.y() - sy)**2 <= THRESH2_POINT: 
                            return self.itemClicked.emit("fmaxe", call)

        for seq in self.recording.sequences:
            for call in seq.calls:
                if call.signal_curves and "main" in call.signal_curves:
                    pts = call.signal_curves["main"]
                    for i in range(len(pts)-1):
                        p1 = self.view.mapViewToScene(QtCore.QPointF(*pts[i]))
                        p2 = self.view.mapViewToScene(QtCore.QPointF(*pts[i+1]))
                        if p1 and p2:
                            d2 = point_line_distance(sx, sy, p1.x(), p1.y(), p2.x(), p2.y())
                            if d2 <= THRESH2_LINE:
                                return self.itemClicked.emit("curve", call)

        for seq in self.recording.sequences:
            for call in seq.calls:
                if (call.t_start_ms <= px <= call.t_end_ms) and (call.f_min_khz <= py <= call.f_max_khz):
                    return self.itemClicked.emit("call", call)

        for seq in self.recording.sequences:
            if (seq.t_start_ms <= px <= seq.t_end_ms) and (seq.f_min_khz <= py <= seq.f_max_khz):
                return self.itemClicked.emit("seq", seq)

        self.itemClicked.emit("rec", self.recording)
