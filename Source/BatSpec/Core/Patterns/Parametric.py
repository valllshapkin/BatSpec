import numpy as np
from scipy.interpolate import splprep, splev
from dataclasses import dataclass
from typing import List, Tuple, Optional
from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.Physical.Units import UREG, PintUnit
import MultiArray as ma

@dataclass
class Trace:
    t: np.ndarray | List  # Время (сек)
    f: np.ndarray | List  # Частота (Гц)
    i: np.ndarray | List  # Интенсивность (амплитуда)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.f = np.asarray(self.f, dtype=float)
        self.i = np.asarray(self.i, dtype=float)

class Call:
    """Акустический сигнал (писки мышей), состоящий из параметрических кривых (Trace)."""
    def __init__(self, traces: List[Trace] | None = None):
        self.traces = traces if traces else []

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        if not self.traces: return 0.0, 0.1, 0.0, 1000.0
        return (
            min(np.min(tr.t) for tr in self.traces), max(np.max(tr.t) for tr in self.traces),
            min(np.min(tr.f) for tr in self.traces), max(np.max(tr.f) for tr in self.traces)
        )

    def toSpecFunc(
        self, 
        sigma_t: float, 
        sigma_f: float, 
        dt: float, 
        df: float,
        padding_s: float = 0.05,
        padding_hz: float = 5000.0,
        ctx: Optional[ma.ArrayContext] = None
    ) -> SpecFunc:
        """Рендерит векторные кривые в растровую спектрограмму (SpecFunc)."""
        t_min, t_max, f_min, f_max = self.get_bounding_box()
        t_arr = np.arange(t_min - padding_s, t_max + padding_s, dt)
        f_arr = np.arange(max(0, f_min - padding_hz), f_max + padding_hz, df)
        
        matrix = np.zeros((len(t_arr), len(f_arr)), dtype=np.float32)

        rad_t, rad_f = max(1, int(3 * sigma_t / dt)), max(1, int(3 * sigma_f / df))
        PT, PF = np.meshgrid(np.arange(-rad_t, rad_t + 1) * dt, np.arange(-rad_f, rad_f + 1) * df, indexing='ij')
        base_patch = np.exp(-0.5 * ((PT / sigma_t)**2 + (PF / sigma_f)**2))

        for trace in self.traces:
            if len(trace.t) < 2: continue
            k = min(3, len(trace.t) - 1)
            tck, _ = splprep([trace.t, trace.f, trace.i], s=0, k=k)
            u_dense = np.linspace(0, 1, max(1000, int((t_arr[-1] - t_arr[0]) / dt) * 2))
            t_dense, f_dense, i_dense = splev(u_dense, tck)

            for t_val, f_val, i_val in zip(t_dense, f_dense, i_dense):
                if i_val <= 0: continue
                idx_t, idx_f = int(round((t_val - t_arr[0]) / dt)), int(round((f_val - f_arr[0]) / df))

                mt_start, mt_end = max(0, idx_t - rad_t), min(matrix.shape[0], idx_t + rad_t + 1)
                mf_start, mf_end = max(0, idx_f - rad_f), min(matrix.shape[1], idx_f + rad_f + 1)
                if mt_start >= mt_end or mf_start >= mf_end: continue

                pt_start, pf_start = mt_start - (idx_t - rad_t), mf_start - (idx_f - rad_f)
                scaled_patch = base_patch[pt_start:pt_start+(mt_end-mt_start), pf_start:pf_start+(mf_end-mf_start)] * i_val
                matrix[mt_start:mt_end, mf_start:mf_end] = np.maximum(matrix[mt_start:mt_end, mf_start:mf_end], scaled_patch)

        if ctx is None: ctx = ma.ArrayContext(ma.Framework.NUMPY, ma.DeviceType.CPU, None)
        return SpecFunc(ma.convert_to(matrix, ctx), ma.convert_to(t_arr, ctx), ma.convert_to(f_arr, ctx), UREG.dimensionless)