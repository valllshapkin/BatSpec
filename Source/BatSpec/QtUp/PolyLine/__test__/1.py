import sys
import random
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QButtonGroup, QGroupBox, QSplitter
)

# Импорты из модуля выше
from BatSpec.QtUp.PolyLine import LineData, LineGroup, FastLineLayer, LineController, LineStatusLabel


class LineCreatorTool(pg.GraphicsObject):
    def __init__(self, plot_item: pg.PlotItem, on_create_callback) -> None:
        super().__init__()
        self.pi = plot_item
        self.on_create = on_create_callback
        self.active = False
        self.drawing = False
        self.current_points = []

        # Инструмент рисования будет использовать быстрый CurveItem (без AA)
        self.preview_curve = pg.PlotCurveItem(pen=pg.mkPen('y', width=2), antialias=False)
        self.pi.addItem(self.preview_curve)

    def boundingRect(self):
        return self.pi.vb.viewRect()

    def paint(self, *args): pass

    def mouseDragEvent(self, ev):
        if not self.active or ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        
        ev.accept()
        pos = self.pi.vb.mapSceneToView(ev.scenePos())
        
        if ev.isStart():
            self.drawing = True
            self.current_points = [(pos.x(), pos.y())]
            self.preview_curve.setData(x=[pos.x()], y=[pos.y()])
            
        elif ev.isFinish() and self.drawing:
            self.drawing = False
            self.preview_curve.setData(x=[], y=[])
            if len(self.current_points) > 1:
                self.on_create(self.current_points)
            self.current_points = []
            
        elif self.drawing:
            last_p = self.current_points[-1]
            # Добавляем точку, только если курсор сместился (оптимизация)
            if (pos.x() - last_p[0])**2 + (pos.y() - last_p[1])**2 > 10.0:
                self.current_points.append((pos.x(), pos.y()))
                pts = np.array(self.current_points)
                self.preview_curve.setData(x=pts[:, 0], y=pts[:, 1])


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Редактор кривых (Экстремальная Оптимизация)")
        self.resize(1200, 800)

        # ── Фоновые данные ──
        self.plot_widget = pg.PlotWidget(background='#111')
        self.plot_item = self.plot_widget.getPlotItem()
        img = pg.ImageItem(np.random.normal(size=(5000, 600)))
        self.plot_item.addItem(img)

        # ── Генерация ДЕСЯТИ ТЫСЯЧ тестовых линий ──
        lines = []
        for i in range(1_000):
            num_pts = random.randint(5, 20)
            x_start = random.uniform(0, 4800)
            y_start = random.uniform(0, 500)
            pts = [(x_start, y_start)]
            for _ in range(num_pts - 1):
                pts.append((pts[-1][0] + random.uniform(5, 100), pts[-1][1] + random.uniform(-30, 30)))
            lines.append(LineData(i, pts))

        self.group = LineGroup("Линии сигналов", lines, pg.mkColor('g'), pg.mkColor('y'))

        # ── UI ──
        main_splitter = QSplitter(Qt.Vertical)
        self.setCentralWidget(main_splitter)
        main_splitter.addWidget(self.plot_widget)

        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        main_splitter.addWidget(bottom_widget)
        main_splitter.setSizes([600, 200])

        tools_group = QGroupBox("Инструменты")
        tools_layout = QVBoxLayout(tools_group)
        
        self.btn_select = QPushButton("🖱️ Выделение")
        self.btn_create = QPushButton("✏️ Рисовать")
        self.btn_erase  = QPushButton("🗑️ Ластик")
        
        self.tool_btns = QButtonGroup(self)
        for i, btn in enumerate([self.btn_select, self.btn_create, self.btn_erase]):
            btn.setCheckable(True)
            self.tool_btns.addButton(btn, i)
            tools_layout.addWidget(btn)
            
        self.btn_select.setChecked(True)
        bottom_layout.addWidget(tools_group)

        # ── Инициализация Контроллеров ──
        self.line_layer = FastLineLayer()
        self.plot_item.addItem(self.line_layer)
        
        self.controller = LineController(self.plot_item, self.line_layer)
        self.controller.set_groups([self.group])

        self.status_label = LineStatusLabel(self.controller)
        bottom_layout.addWidget(self.status_label, stretch=1)

        self.creator_tool = LineCreatorTool(self.plot_item, self._on_line_created)
        self.plot_item.addItem(self.creator_tool)

        self.plot_item.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self.tool_btns.idClicked.connect(self._on_tool_changed)
        
        self.plot_item.setXRange(0, 2000, padding=0)
        self.plot_item.setYRange(0, 600, padding=0)

    def _on_tool_changed(self, tool_id: int) -> None:
        self.creator_tool.active = (tool_id == 1)
        is_panning = (tool_id == 0)
        self.plot_item.getViewBox().setMouseEnabled(x=is_panning, y=is_panning)
        
        cursors = [Qt.ArrowCursor, Qt.CrossCursor, Qt.ForbiddenCursor]
        self.plot_widget.setCursor(cursors[tool_id])
        if tool_id != 0:
            self.controller.deselect()

    def _on_scene_clicked(self, ev) -> None:
        tool_id = self.tool_btns.checkedId()
        if tool_id == 0:
            self.controller.on_click(ev)
        elif tool_id == 2 and ev.button() == Qt.LeftButton:
            self._erase_line_at(ev.scenePos())

    def _on_line_created(self, points: list[tuple[float, float]]) -> None:
        lines = self.group.lines
        new_idx = max([l.idx for l in lines], default=-1) + 1
        
        # Разрежаем точки: используем каждую 3-ю + последнюю, чтобы не было слишком много узлов
        sparse_pts = points[::3]
        if points[-1] not in sparse_pts:
            sparse_pts.append(points[-1])
            
        new_line = LineData(new_idx, sparse_pts)
        lines.append(new_line)
        self.line_layer.add_to_bucket(new_line)
        self.line_layer.rebuild_path()
        self.controller.signals.lines_changed.emit(self.group.name)

    def _erase_line_at(self, scene_pos) -> None:
        pos = self.plot_item.vb.mapSceneToView(scene_pos)
        px, py = pos.x(), pos.y()
        
        vr = self.plot_item.vb.viewRect()
        sx = self.plot_item.vb.width() / vr.width()
        sy = self.plot_item.vb.height() / vr.height()

        b = int(px) // self.line_layer.BUCKET_W
        hit = None
        for line in reversed(self.line_layer._buckets.get(b, [])):
            if line.distance_to(px, py, sx, sy) < 10.0:
                hit = line
                break
                
        if hit:
            if self.controller.active_idx == hit.idx:
                self.controller.deselect()
            self.group.lines.remove(hit)
            self.line_layer.remove_from_bucket(hit)
            self.line_layer.rebuild_path()
            self.controller.signals.lines_changed.emit(self.group.name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())