import logging
from threading import Thread
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QTransform

class AdaptiveImageItem(pg.ImageItem):
    MIN_COLS = 64
    _sigMipmapReady = Signal()

    def __init__(self):
        super().__init__()
        self.setOpts(axisOrder='row-major')
        cmap = pg.colormap.get('viridis')
        self.setLookupTable(cmap.getLookupTable())

        self._mipmap: list[np.ndarray] = []
        self._x_range = (0.0, 1.0)
        self._y_range = (0.0, 1.0)
        self._levels = (0.0, 1.0)
        self._plot_item: pg.PlotItem | None = None
        self._updating = False
        self._sigMipmapReady.connect(self._doUpdate)

    def attachTo(self, plot_item: pg.PlotItem) -> None:
        self._plot_item = plot_item
        plot_item.getViewBox().sigRangeChanged.connect(self._doUpdate)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if ax == 0: return self._x_range
        elif ax == 1: return self._y_range
        return super().dataBounds(ax, frac, orthoRange)

    def setFullData(self, arr: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
        self._x_range = x_range
        self._y_range = y_range
        data_min = float(np.min(arr))
        data_max = float(np.max(arr))
        self._levels = (data_min, data_max if data_max > data_min else data_min + 1.0)
        self._mipmap = []
        Thread(target=self._buildMipmap, args=(arr,), daemon=True).start()

    def _buildMipmap(self, arr: np.ndarray) -> None:
        data = np.ascontiguousarray(arr, dtype=np.float32)
        levels = [data]
        cur = data
        while cur.shape[1] // 2 >= self.MIN_COLS:
            nf, nt = cur.shape
            nt2 = nt // 2
            down = cur[:, :nt2 * 2].reshape(nf, nt2, 2).mean(axis=2).astype(np.float32)
            levels.append(down)
            cur = down
            
        self._mipmap = levels
        self._sigMipmapReady.emit()

    @Slot()
    def _doUpdate(self) -> None:
        if not self._mipmap or self._plot_item is None or self._updating: return
        self._updating = True
        try: self._renderTile()
        except Exception: logging.exception("AdaptiveImageItem._renderTile error")
        finally: self._updating = False

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

        vis_rows, vis_cols = r1 - r0, c1 - c0
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
        bh, bw = max(1, ch // tile_h), max(1, cw // tile_w)

        if bh > 1 or bw > 1:
            out_h, out_w = ch // bh, cw // bw
            sub = crop[:bh * out_h, :bw * out_w].reshape(out_h, bh, out_w, bw).mean(axis=(1, 3)).astype(np.float32)
        else:
            sub = crop

        self.setImage(sub, autoLevels=False, levels=self._levels)

        tr = QTransform()
        tr.translate(t_min + c0_a * t_per_col, f_min + r0 * f_per_row)
        tr.scale(bw * s * t_per_col, bh * f_per_row)
        self.setTransform(tr)
