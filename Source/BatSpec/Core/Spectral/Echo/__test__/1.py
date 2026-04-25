import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# --- Корректные импорты из вашего проекта ---
from BatSpec.Core.Physical.Units import UREG
from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.ConvFFT import convolveTime

# =====================================================================
# 1. МОДУЛИ ГЕНЕРАЦИИ ЭХА
# =====================================================================
def _get_zero_index(time_axis: torch.Tensor) -> int:
    return int(torch.argmin(torch.abs(time_axis)))

def init_derivatives_window(ref_spec: SpecFunc, duration_ms: float, anchor_pct: float) -> SpecFunc:
    dt_val, dt_unit = ref_spec.dt
    f_axis, f_unit = ref_spec.freq
    
    duration_s = duration_ms / 1000.0
    N_time = max(2, int(round(duration_s / dt_val)))
    anchor_idx = int(round(N_time * (anchor_pct / 100.0)))
    anchor_idx = max(0, min(N_time - 1, anchor_idx))
    
    t_axis = (torch.arange(N_time, dtype=f_axis.dtype, device=f_axis.device) - anchor_idx) * dt_val
    N_freq = len(f_axis)
    D = torch.zeros((1, N_freq, N_time), dtype=f_axis.dtype, device=f_axis.device)
    
    if anchor_idx > 0:
        D[..., :anchor_idx] = 50.0 
        
    if anchor_idx < N_time:
        base_decay = -5.0
        freq_penalty = (f_axis / (f_axis.max() + 1e-6)).unsqueeze(1) 
        decay_rates = base_decay - (15.0 * freq_penalty) 
        D[..., anchor_idx:] = decay_rates
    
    return SpecFunc(matrix=(D, UREG.hertz), freq=f_axis, time=t_axis)

def build_causal_echo_window(derivatives_spec: SpecFunc) -> SpecFunc:
    D, d_unit = derivatives_spec.values
    t_axis, t_unit = derivatives_spec.time
    dt_val, _ = derivatives_spec.dt
    
    Z = _get_zero_index(t_axis)
    D_left = D[..., :Z]
    D_right = D[..., Z:]
    
    if D_right.shape[-1] > 0:
        X_right = torch.cumsum(D_right, dim=-1) * dt_val
        X_right = X_right - X_right[..., 0:1]
    else:
        X_right = torch.empty((*D.shape[:-1], 0), dtype=D.dtype, device=D.device)
        
    if D_left.shape[-1] > 0:
        D_left_rev = torch.flip(D_left, dims=[-1])
        X_left_rev = torch.cumsum(D_left_rev, dim=-1) * (-dt_val)
        X_left = torch.flip(X_left_rev, dims=[-1])
    else:
        X_left = torch.empty((*D.shape[:-1], 0), dtype=D.dtype, device=D.device)
        
    X = torch.cat([X_left, X_right], dim=-1)
    echo_matrix = torch.exp(X)
    
    # ФИЗИЧЕСКАЯ НОРМИРОВКА (Остается!)
    sum_vals = echo_matrix.sum(dim=-1, keepdim=True)
    echo_matrix = echo_matrix / (sum_vals + 1e-8)
    
    return SpecFunc(
        matrix=(echo_matrix, UREG.dimensionless), 
        freq=derivatives_spec.freq[0], 
        time=t_axis
    )

# =====================================================================
# 2. НОВАЯ КОМБИНИРОВАННАЯ ФУНКЦИЯ ПОТЕРЬ
# =====================================================================
def spectral_loss(pred, target, eps=1e-3):
    """
    Комбинированная функция потерь (Linear + Log).
    Стандарт де-факто для факторизации спектрограмм.
    """
    # 1. Линейный MSE (держит амплитуды в узде, помогает учить правильное эхо)
    loss_linear = F.mse_loss(pred, target)
    
    # 2. Логарифмический MSE (не дает импульсам занулиться, вытягивает структуру)
    pred_log = torch.log(pred + eps)
    target_log = torch.log(target + eps)
    loss_log = F.mse_loss(pred_log, target_log)
    
    # Складываем их (коэффициент 0.1 для логарифма балансирует их влияние)
    return loss_linear + 0.1 * loss_log

# =====================================================================
# 3. ОБУЧАЮЩИЙ СКРИПТ (Фрагмент)
# =====================================================================
if __name__ == "__main__":
    TIME_STEPS = 400
    FREQ_BINS = 64
    t_axis = torch.linspace(0.0, 2.0, TIME_STEPS)
    f_axis = torch.linspace(0.0, 100.0, FREQ_BINS)
    unit = UREG.dimensionless

    # --- 1. ГЕНЕРАЦИЯ GROUND TRUTH ---
    S_true_vals = torch.zeros((1, FREQ_BINS, TIME_STEPS))
    S_true_vals[0, 15:20, 50] = 5.0   
    S_true_vals[0, 30:40, 150] = 3.0  
    S_true_vals[0, 50:60, 250] = 4.0  
    S_true = SpecFunc((S_true_vals, unit), f_axis, t_axis)

    P_true_spec = init_derivatives_window(S_true, duration_ms=400.0, anchor_pct=10.0)
    P_true_vals, _ = P_true_spec.values
    Z_true = _get_zero_index(P_true_spec.time[0])
    freq_penalty = (f_axis / f_axis.max()).unsqueeze(1)
    P_true_vals[..., Z_true:] = -5.0 - (20.0 * freq_penalty)
    P_true_spec.values = (P_true_vals, unit)
    
    E_true = build_causal_echo_window(P_true_spec)

    Target = convolveTime(S_true, E_true)
    
    # Убираем всякие Softplus для таргета, берем чистые данные
    Target_tensor = Target.values[0].detach() 

    # --- 2. ИНИЦИАЛИЗАЦИЯ ПАРАМЕТРОВ ---
    # ИСПРАВЛЕНИЕ: Инициализируем сигнал СЛУЧАЙНЫМ ШУМОМ. 
    # Это классика NMF/Deconvolution. Шум ломает симметрию и дает градиентам цель.
    # Инициализация - случайный шум, чтобы логарифму было за что зацепиться
    Sc_tensor = (torch.rand_like(S_true_vals) * 0.1).requires_grad_(True)
    
    P_learn_spec = init_derivatives_window(S_true, duration_ms=400.0, anchor_pct=10.0)
    P_learn_vals = P_learn_spec.values[0].clone().detach()
    Z_learn = _get_zero_index(P_learn_spec.time[0])
    P_learn_vals[..., :Z_learn] = 20.0
    P_learn_vals[..., Z_learn:] = -3.0
    P_tensor = P_learn_vals.requires_grad_(True)

    # ОПТИМИЗАТОР
    optimizer = optim.Adam([
        {'params': Sc_tensor, 'lr': 0.05}, 
        {'params': P_tensor, 'lr': 0.05}   
    ])

    EPOCHS = 500 
    LAMBDA_L1 = 1e-4 
    loss_history = []

    print("Начинаем обучение (Combined Spectral Loss + Random Init + ReLU)...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        P_learn_spec.values = (P_tensor, unit)
        Ec = build_causal_echo_window(P_learn_spec)

        # Оставляем ReLU! Логарифмический лосс вытащит нужные пики из нуля.
        Sc_positive = F.relu(Sc_tensor)
        Sc = SpecFunc((Sc_positive, unit), f_axis, t_axis)

        Pred = convolveTime(Sc, Ec)
        Pred_tensor = Pred.values[0]

        # ИСПОЛЬЗУЕМ КОМБИНИРОВАННЫЙ ЛОСС
        loss_recon = spectral_loss(Pred_tensor, Target_tensor, eps=1e-3)
        
        loss_l1 = torch.abs(Sc_positive).mean()

        loss = loss_recon + LAMBDA_L1 * loss_l1
        loss.backward()
        
        optimizer.step()
        loss_history.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:04d}/{EPOCHS} | Total: {loss.item():.6f} | L1: {loss_l1.item():.6f}")

    # --- 4. ВИЗУАЛИЗАЦИЯ ---
    Sc_final = F.relu(Sc_tensor).detach().numpy()[0]
    Ec_final = Ec.values[0][0].detach().numpy()
    Pred_final = Pred.values[0][0].detach().numpy()
    
    Target_np = Target_tensor[0].numpy()
    E_true_np = E_true.values[0][0].numpy()
    S_true_np = S_true_vals[0].numpy()

    extent_S = [t_axis[0].item(), t_axis[-1].item(), f_axis[0].item(), f_axis[-1].item()]
    t_echo = P_learn_spec.time[0].numpy()
    extent_E = [t_echo[0], t_echo[-1], f_axis[0].item(), f_axis[-1].item()]

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), dpi=100)

    # ИСПРАВЛЕНИЕ ГРАФИКИ: Жестко фиксируем vmin и vmax, чтобы matplotlib не обманывал глаза!
    vmax_S = S_true_np.max()
    vmax_E = E_true_np.max()
    vmax_T = Target_np.max()

    axs[0, 0].imshow(S_true_np, aspect='auto', origin='lower', extent=extent_S, cmap='inferno', vmin=0, vmax=vmax_S); axs[0, 0].set_title("Ground Truth S")
    axs[0, 1].imshow(E_true_np, aspect='auto', origin='lower', extent=extent_E, cmap='magma', vmin=0, vmax=vmax_E); axs[0, 1].set_title("Ground Truth Echo")
    axs[0, 2].imshow(Target_np, aspect='auto', origin='lower', extent=extent_S, cmap='viridis', vmin=0, vmax=vmax_T); axs[0, 2].set_title("Target = S * E")
    
    axs[1, 0].imshow(Sc_final, aspect='auto', origin='lower', extent=extent_S, cmap='inferno', vmin=0, vmax=vmax_S); axs[1, 0].set_title("Learned Signal (Sc)")
    axs[1, 1].imshow(Ec_final, aspect='auto', origin='lower', extent=extent_E, cmap='magma', vmin=0, vmax=vmax_E); axs[1, 1].set_title("Learned Echo")
    axs[1, 2].imshow(Pred_final, aspect='auto', origin='lower', extent=extent_S, cmap='viridis', vmin=0, vmax=vmax_T); axs[1, 2].set_title("Predicted = Sc * Ec")
    
    axs[2, 0].plot(loss_history, 'b', lw=2); axs[2, 0].set_title("Training Loss (MSE + L1)"); axs[2, 0].set_xlabel("Epoch"); axs[2, 0].set_yscale('log'); axs[2, 0].grid(True)
    
    # Проверка производных на тех бинах, где ЕСТЬ сигнал
    f_idx_high = 55 
    axs[2, 1].plot(t_echo, P_true_vals[0, f_idx_high].numpy(), 'k--', label="True P"); 
    axs[2, 1].plot(t_echo, P_tensor[0, f_idx_high].detach().numpy(), 'r', label="Learned P"); 
    axs[2, 1].set_title(f"Производные (ВЧ, bin={f_idx_high})"); axs[2, 1].axvline(0, color='gray', linestyle='--'); axs[2, 1].legend(); axs[2, 1].grid(True)
    
    f_idx_low = 17
    axs[2, 2].plot(t_echo, P_true_vals[0, f_idx_low].numpy(), 'k--', label="True P"); 
    axs[2, 2].plot(t_echo, P_tensor[0, f_idx_low].detach().numpy(), 'b', label="Learned P"); 
    axs[2, 2].set_title(f"Производные (НЧ, bin={f_idx_low})"); axs[2, 2].axvline(0, color='gray', linestyle='--'); axs[2, 2].legend(); axs[2, 2].grid(True)
    
    plt.tight_layout()
    plt.show()