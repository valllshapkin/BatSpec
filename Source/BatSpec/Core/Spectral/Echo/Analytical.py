import numpy as np
from scipy.signal import fftconvolve
import MultiArray as ma
from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.Physical.Units import UREG

class CausalEchoModel:
    """Строго каузальная модель эха (пик всегда на t=0). Подходит для деконволюции."""
    def __init__(self, decay_rate_base: float, f_ref: float, freq_exp: float, time_power: float = 0.0, max_duration: float = 10.0):
        self.decay_rate_base = decay_rate_base
        self.f_ref = f_ref
        self.freq_exp = freq_exp
        self.time_power = time_power
        self.max_duration = max_duration

    def get(self, freq_axis: np.ndarray, dt: float, threshold: float = 1e-4) -> SpecFunc:
        rates = self.decay_rate_base * (np.abs(freq_axis) / self.f_ref) ** self.freq_exp
        with np.errstate(divide='ignore'):
            durations = np.where(rates > 0, -np.log(threshold) / rates, self.max_duration)
        
        max_t = min(np.max(durations), self.max_duration)
        time_axis = np.arange(0, max_t, dt, dtype=np.float32)
        
        t_grid = time_axis[np.newaxis, :]  # Shape: (1, T)
        rates_grid = rates[:, np.newaxis]  # Shape: (F, 1) - под новую размерность (T, H)
        
        power_decay = 1.0 / ((t_grid + 1.0) ** self.time_power)
        exp_decay = np.exp(-rates_grid * t_grid)
        
        matrix = (power_decay * exp_decay).astype(np.float32)
        # В новой архитектуре матрица имеет форму (T, F), поэтому транспонируем
        return SpecFunc(matrix=ma.convert_to(matrix.T, ma.ArrayContext.from_array(time_axis)), 
                        time=time_axis, freq=freq_axis, unit=UREG.dimensionless)

def removeEchoWiener(spec: SpecFunc, model: CausalEchoModel, eps_factor: float = 1e-2) -> SpecFunc:
    """Слепая очистка эха через фильтр Винера (Wiener Deconvolution)."""
    ctx = spec.context
    mat_np = ma.to_numpy(spec._matrx_a) # Shape (T, F)
    f_axis = ma.to_numpy(spec._sec___a)
    
    _, dt_val = spec.dt
    sr = 1.0 / dt_val
    T_len, F_len = mat_np.shape
    
    new_matrix = np.zeros_like(mat_np)
    
    for i in range(F_len):
        freq_val = f_axis[i]
        signal_1d = mat_np[:, i]
        
        # Генерируем 1D ядро эха для конкретной частоты
        raw_kernel = ma.to_numpy(model.get(np.array([freq_val]), dt_val)._matrx_a)[:, 0]
        K = len(raw_kernel)
        M = T_len + K - 1 
        
        H = np.fft.rfft(raw_kernel, n=M)
        X = np.fft.rfft(signal_1d, n=M)
        
        eps = eps_factor * np.max(np.abs(H))
        H_inv = np.conj(H) / (np.abs(H)**2 + eps**2)
        
        signal_restored = np.fft.irfft(X * H_inv, n=M)[:T_len]
        new_matrix[:, i] = np.clip(signal_restored, 0, None)
        
    return SpecFunc(ma.convert_to(new_matrix, ctx), spec._first_a, spec._sec___a, spec._matrx_u)

def removeEchoRichardsonLucy(spec: SpecFunc, model: CausalEchoModel, iterations: int = 15) -> SpecFunc:
    """Очистка эха через деконволюцию Ричардсона-Люси (строго положительные значения)."""
    ctx = spec.context
    mat_np = ma.to_numpy(spec._matrx_a) # Shape (T, F)
    f_axis = ma.to_numpy(spec._sec___a)
    _, dt_val = spec.dt
    T_len, F_len = mat_np.shape
    
    new_matrix = np.zeros_like(mat_np)
    
    for i in range(F_len):
        signal_1d = np.clip(mat_np[:, i], 1e-24, None)
        raw_kernel = ma.to_numpy(model.get(np.array([f_axis[i]]), dt_val)._matrx_a)[:, 0]
        
        kernel_rev = raw_kernel[::-1]
        estimate = np.copy(signal_1d)
        
        for _ in range(iterations):
            blurred = fftconvolve(estimate, raw_kernel, mode='full')[:T_len]
            blurred[blurred == 0] = 1e-24
            ratio = signal_1d / blurred
            correction = fftconvolve(ratio, kernel_rev, mode='full')[-T_len:]
            estimate = estimate * correction
            
        new_matrix[:, i] = estimate
        
    return SpecFunc(ma.convert_to(new_matrix, ctx), spec._first_a, spec._sec___a, spec._matrx_u)