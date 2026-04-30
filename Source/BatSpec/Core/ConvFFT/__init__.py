from matplotlib import pyplot as plt
import torch
import torch.fft
from typing import Union

# --- Внедряем наш новый универсальный фреймворк ---
import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType

from BatSpec.Core.Functions import SpecFunc, TimeFunc
from BatSpec.Core.Physical.Units import UREG, unit_mul

# =====================================================================
# НИЗКОУРОВНЕВОЕ ТЕНЗОРНОЕ ЯДРО (Для быстрых ML-алгоритмов)
# =====================================================================

def tensor_conv1d_time(sig_v: torch.Tensor, ker_v: torch.Tensor, anchor_idx: int) -> torch.Tensor:
    """1D Свертка по оси времени (с учетом якоря)."""
    T_sig, T_ker = sig_v.shape[-1], ker_v.shape[-1]
    N = T_sig + T_ker - 1
    Sig_f = torch.fft.rfft(sig_v, n=N, dim=-1)
    Ker_f = torch.fft.rfft(ker_v, n=N, dim=-1)
    out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
    return out[..., anchor_idx : anchor_idx + T_sig]

def tensor_conv2d_ft(sig_v: torch.Tensor, ker_v: torch.Tensor, anchor_f: int, anchor_t: int) -> torch.Tensor:
    """2D Свертка по частоте и времени (с учетом якорей)."""
    F_sig, T_sig = sig_v.shape[-2], sig_v.shape[-1]
    F_ker, T_ker = ker_v.shape[-2], ker_v.shape[-1]
    N_f, N_t = F_sig + F_ker - 1, T_sig + T_ker - 1
    Sig_f = torch.fft.rfft2(sig_v, s=(N_f, N_t), dim=(-2, -1))
    Ker_f = torch.fft.rfft2(ker_v, s=(N_f, N_t), dim=(-2, -1))
    out = torch.fft.irfft2(Sig_f * Ker_f, s=(N_f, N_t), dim=(-2, -1))
    return out[..., anchor_f : anchor_f + F_sig, anchor_t : anchor_t + T_sig]

def tensor_match1d_time(sig_v: torch.Tensor, ker_v: torch.Tensor) -> torch.Tensor:
    """1D кросс-корреляция по времени (с автоматическим центрированием)."""
    flipped_ker = torch.flip(ker_v, dims=(-1,))
    anchor_idx = (ker_v.shape[-1] - 1) // 2
    return tensor_conv1d_time(sig_v, flipped_ker, anchor_idx)

def tensor_match2d_ft(sig_v: torch.Tensor, ker_v: torch.Tensor) -> torch.Tensor:
    """2D кросс-корреляция по частоте и времени (с автоматическим центрированием)."""
    flipped_ker = torch.flip(ker_v, dims=(-2, -1))
    anchor_f = (ker_v.shape[-2] - 1) // 2
    anchor_t = (ker_v.shape[-1] - 1) // 2
    return tensor_conv2d_ft(sig_v, flipped_ker, anchor_f, anchor_t)


# =====================================================================
# ВЫСОКОУРОВНЕВЫЙ DSP API (С единицами измерения и авто-контекстом)
# =====================================================================

def _get_zero_index(time_axis: torch.Tensor) -> int:
    """Находит индекс якоря: точку на оси времени, максимально близкую к 0."""
    return int(torch.argmin(torch.abs(time_axis)))

def _to_pt(func: Union[TimeFunc, SpecFunc]) -> Union[TimeFunc, SpecFunc]:
    ctx = func.context
    pt_ctx = ArrayContext(Framework.TORCH, DeviceType.GPU if ctx.isGPU() else DeviceType.CPU, None)
    return func.to_context(pt_ctx)

def conv_T_T(sig: TimeFunc, ker: TimeFunc) -> TimeFunc:
    orig_ctx = sig.context
    sig_pt, ker_pt = _to_pt(sig), _to_pt(ker)
    sig_v, sig_u = sig_pt.values
    ker_v, ker_u = ker_pt.values
    
    Z = _get_zero_index(ker_pt.time[0])
    out = tensor_conv1d_time(sig_v, ker_v, Z)
    
    dt_val, dt_unit = ker_pt.dt
    out = out * float(dt_val)
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    return TimeFunc(values=(out, new_unit), axis=sig_pt.time[0]).to_context(orig_ctx)

def conv_S_T(sig: SpecFunc, ker: TimeFunc) -> SpecFunc:
    orig_ctx = sig.context
    sig_pt, ker_pt = _to_pt(sig), _to_pt(ker)
    sig_v, sig_u = sig_pt.values
    ker_v, ker_u = ker_pt.values
    
    Z = _get_zero_index(ker_pt.time[0])
    out = tensor_conv1d_time(sig_v, ker_v.unsqueeze(-2), Z)
    
    dt_val, dt_unit = ker_pt.dt
    out = out * float(dt_val)
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    return SpecFunc(matrix=(out, new_unit), freq=sig_pt.freq[0], time=sig_pt.time[0]).to_context(orig_ctx)

def conv_T_S(sig: TimeFunc, ker: SpecFunc) -> SpecFunc:
    orig_ctx = sig.context
    sig_pt, ker_pt = _to_pt(sig), _to_pt(ker)
    sig_v, sig_u = sig_pt.values
    ker_v, ker_u = ker_pt.values
    
    Z = _get_zero_index(ker_pt.time[0])
    out = tensor_conv1d_time(sig_v.unsqueeze(-2), ker_v, Z)
    
    dt_val, dt_unit = ker_pt.dt
    out = out * float(dt_val)
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    return SpecFunc(matrix=(out, new_unit), freq=ker_pt.freq[0], time=sig_pt.time[0]).to_context(orig_ctx)

def conv_S_S(sig: SpecFunc, ker: SpecFunc) -> SpecFunc:
    orig_ctx = sig.context
    sig_pt, ker_pt = _to_pt(sig), _to_pt(ker)
    sig_v, sig_u = sig_pt.values
    ker_v, ker_u = ker_pt.values
    
    assert sig_v.shape[-2] == ker_v.shape[-2], "Размерности частот сигнала и ядра должны совпадать"
    
    Z = _get_zero_index(ker_pt.time[0])
    out = tensor_conv1d_time(sig_v, ker_v, Z)
    
    dt_val, dt_unit = ker_pt.dt
    out = out * float(dt_val)
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    return SpecFunc(matrix=(out, new_unit), freq=sig_pt.freq[0], time=sig_pt.time[0]).to_context(orig_ctx)


def convolveTime(a: Union[TimeFunc, SpecFunc], b: Union[TimeFunc, SpecFunc]) -> Union[TimeFunc, SpecFunc]:
    """Универсальная функция свёртки. a - сигнал, b - ядро(фильтр). Определяет тип операции автоматически."""
    if isinstance(a, TimeFunc) and isinstance(b, TimeFunc): return conv_T_T(a, b)
    elif isinstance(a, SpecFunc) and isinstance(b, TimeFunc): return conv_S_T(a, b)
    elif isinstance(a, TimeFunc) and isinstance(b, SpecFunc): return conv_T_S(a, b)
    elif isinstance(a, SpecFunc) and isinstance(b, SpecFunc): return conv_S_S(a, b)
    raise TypeError(f"Неподдерживаемые типы для свёртки: {type(a)} и {type(b)}")
    
def convolveFT(sig: SpecFunc, ker: SpecFunc) -> SpecFunc:
    """ИСТИННАЯ 2D СВЁРТКА: 2D Спектрограмма * 2D Окно -> 2D Спектрограмма."""
    orig_ctx = sig.context
    sig_pt, ker_pt = _to_pt(sig), _to_pt(ker)
    sig_v, sig_u = sig_pt.values
    ker_v, ker_u = ker_pt.values
    
    Z_f = _get_zero_index(ker_pt.freq[0])
    Z_t = _get_zero_index(ker_pt.time[0])
    
    out = tensor_conv2d_ft(sig_v, ker_v, Z_f, Z_t)
    
    df_val, df_unit = ker_pt.df
    dt_val, dt_unit = ker_pt.dt
    out = out * float(df_val * dt_val)
    new_unit = unit_mul(sig_u, ker_u, df_unit, dt_unit)
    
    return SpecFunc(matrix=(out, new_unit), freq=sig_pt.freq[0], time=sig_pt.time[0]).to_context(orig_ctx)


# =====================================================================
# ТЕСТИРОВАНИЕ
# =====================================================================
if __name__ == "__main__":
    TIME_STEPS = 1000
    FREQ_BINS = 64
    
    t_axis = torch.linspace(-5.0, 5.0, TIME_STEPS)
    f_axis = torch.linspace(0.0, 100.0, FREQ_BINS)
    
    unit = UREG.dimensionless
    extent_S = [t_axis[0].item(), t_axis[-1].item(), f_axis[0].item(), f_axis[-1].item()]

    # 1. ТЕСТ: T * T
    print("\n--- 1. ТЕСТ (T * T): Свертка Лоренциан ---")
    y_vals = (1.0 / (t_axis**2 + 1.0)).unsqueeze(0)
    sig_T = TimeFunc(values=(y_vals, unit), axis=t_axis)
    ker_T = TimeFunc(values=(y_vals, unit), axis=t_axis)
    res_T_T = convolveTime(sig_T, ker_T)
    y_num = res_T_T.values[0][0]
    y_ana = (2 * torch.pi) / (t_axis**2 + 4.0)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_axis.numpy(), y_ana.numpy(), 'k--', lw=3, label="Аналитика $2\pi / (t^2+4)$")
    plt.plot(t_axis.numpy(), y_num.numpy(), 'r-', alpha=0.7, lw=2, label="Численно T * T")
    plt.plot(t_axis.numpy(), y_vals[0].numpy(), 'b-', alpha=0.3, label="Исходный сигнал")
    plt.title("1. T * T (1D Свертка)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()

    # 2. ТЕСТ: S * T
    print("--- 2. ТЕСТ (S * T): Сглаживание шумной спектрограммы ---")
    F_grid, T_grid = torch.meshgrid(f_axis, t_axis, indexing='ij')
    clean_chirp = torch.exp(-((F_grid - (T_grid + 5)*10)**2) / 20.0)
    noise = torch.rand_like(clean_chirp) * 0.5
    S_noisy_vals = (clean_chirp + noise).unsqueeze(0)
    T_gauss_vals = torch.exp(-(t_axis**2) / (2 * 0.2**2)).unsqueeze(0)
    
    sig_S = SpecFunc(matrix=(S_noisy_vals, unit), freq=f_axis, time=t_axis)
    ker_T_gauss = TimeFunc(values=(T_gauss_vals, unit), axis=t_axis)
    res_S_T = convolveTime(sig_S, ker_T_gauss)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(S_noisy_vals[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='viridis')
    axs[0].set_title("Сигнал S (Шумный чирп)")
    axs[1].plot(t_axis.numpy(), T_gauss_vals[0].numpy(), 'g')
    axs[1].set_title("Ядро T (Гауссиана)")
    axs[1].set_xlim(-1, 1); axs[1].grid(True)
    axs[2].imshow(res_S_T.values[0][0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='viridis')
    axs[2].set_title("Результат S * T (Сглажено)")
    plt.tight_layout()

    # 3. ТЕСТ: T * S
    print("--- 3. ТЕСТ (T * S): Размещение шаблонов по спайкам ---")
    T_spikes_vals = torch.zeros_like(t_axis)
    T_spikes_vals[torch.argmin(torch.abs(t_axis - (-3.0)))] = 1.0
    T_spikes_vals[torch.argmin(torch.abs(t_axis - 0.0))] = 0.5
    T_spikes_vals[torch.argmin(torch.abs(t_axis - 2.0))] = 1.2
    T_spikes_vals = T_spikes_vals.unsqueeze(0)
    S_blob_vals = (torch.exp(-((F_grid - 60)**2) / 100.0) * torch.exp(-(T_grid**2) / 0.1)).unsqueeze(0)
    
    sig_T_spikes = TimeFunc(values=(T_spikes_vals, unit), axis=t_axis)
    ker_S_blob = SpecFunc(matrix=(S_blob_vals, unit), freq=f_axis, time=t_axis)
    res_T_S = convolveTime(sig_T_spikes, ker_S_blob)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(t_axis.numpy(), T_spikes_vals[0].numpy(), 'r')
    axs[0].set_title("Активации T (Спайки)")
    axs[0].grid(True)
    axs[1].imshow(S_blob_vals[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='magma')
    axs[1].set_title("Ядро S (Шаблон)")
    axs[2].imshow(res_T_S.values[0][0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='magma')
    axs[2].set_title("Результат T * S")
    plt.tight_layout()

    # 4. ТЕСТ: S * S
    print("--- 4. ТЕСТ (S * S): Физичное эхо (частотно-зависимое) ---")
    S_clean_vals = clean_chirp.unsqueeze(0)
    echo_mask = (T_grid >= 0).float()
    decay_rates = 1.0 + (F_grid / 20.0) 
    S_echo_vals = (torch.exp(-T_grid * decay_rates) * echo_mask).unsqueeze(0)
    
    sig_S_clean = SpecFunc(matrix=(S_clean_vals, unit), freq=f_axis, time=t_axis)
    ker_S_echo = SpecFunc(matrix=(S_echo_vals, unit), freq=f_axis, time=t_axis)
    res_S_S = convolveTime(sig_S_clean, ker_S_echo)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(S_clean_vals[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='plasma')
    axs[0].set_title("Сигнал S (Оригинал)")
    axs[1].imshow(S_echo_vals[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='plasma')
    axs[1].set_title("Профиль эха S (ВЧ затухают быстрее)")
    axs[1].set_xlim(-1, 3)
    axs[2].imshow(res_S_S.values[0][0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='plasma')
    axs[2].set_title("Результат S * S (Шлейф эха)")
    plt.tight_layout()

    # 5. ТЕСТ: convolveFT
    print("\n--- 5. ТЕСТ (convolveFT): 2D Размытие спектрограммы ---")
    S_points = torch.zeros(1, FREQ_BINS, TIME_STEPS)
    S_points[0, 15, 200] = 1.0
    S_points[0, 45, 500] = 0.8
    S_points[0, 30, 800] = 1.2
    sig_S_2d = SpecFunc(matrix=(S_points, unit), freq=f_axis, time=t_axis)
    
    f_ker_axis = torch.linspace(-30.0, 30.0, 31)
    t_ker_axis = torch.linspace(-1.0, 1.0, 51)
    F_ker_grid, T_ker_grid = torch.meshgrid(f_ker_axis, t_ker_axis, indexing='ij')
    S_gauss_2d = torch.exp(- (F_ker_grid**2 / 100.0) - (T_ker_grid**2 / 0.05)).unsqueeze(0)
    ker_S_2d = SpecFunc(matrix=(S_gauss_2d, unit), freq=f_ker_axis, time=t_ker_axis)
    
    res_FT = convolveFT(sig_S_2d, ker_S_2d)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(S_points[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='inferno')
    axs[0].set_title("Сигнал S (Одиночные точки)")
    
    ker_extent = [t_ker_axis[0].item(), t_ker_axis[-1].item(), f_ker_axis[0].item(), f_ker_axis[-1].item()]
    axs[1].imshow(S_gauss_2d[0].numpy(), aspect='auto', origin='lower', extent=ker_extent, cmap='inferno')
    axs[1].set_title("Ядро S (2D Гауссиана)")
    axs[1].axhline(0, color='w', alpha=0.5, linestyle='--')
    axs[1].axvline(0, color='w', alpha=0.5, linestyle='--')
    
    axs[2].imshow(res_FT.values[0][0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='inferno')
    axs[2].set_title("Результат convolveFT (2D Свертка)")
    plt.tight_layout()
    plt.show()