import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Any
import warnings

# --- Импорты из вашего проекта ---
from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.Physical.Units import UREG
from BatSpec.Core.ConvFFT import convolveTime

# =====================================================================
# Вспомогательные функции (Межфреймворковый мост)
# =====================================================================
def _get_fw_info(tensor: Any) -> Tuple[str, Any]:
    """Определяет фреймворк и устройство тензора."""
    type_str = str(type(tensor)).lower()
    dev = getattr(tensor, 'device', None)
    if 'torch' in type_str: return 'torch', dev
    if 'tensorflow' in type_str: return 'tensorflow', dev
    return 'numpy', dev


# =====================================================================
# Физические генераторы Эха
# =====================================================================
def _get_zero_index(time_axis: torch.Tensor) -> int:
    """Находит индекс якоря (t=0) на оси времени."""
    return int(torch.argmin(torch.abs(time_axis)))

def init_derivatives_window(ref_spec: SpecFunc, duration_ms: float, anchor_pct: float) -> SpecFunc:
    """Инициализирует окно стартовых производных для эха."""
    dt_val, _ = ref_spec.dt
    f_axis, _ = ref_spec.freq
    duration_s = duration_ms / 1000.0
    N_time = max(2, int(round(duration_s / dt_val)))
    anchor_idx = int(round(N_time * (anchor_pct / 100.0)))
    anchor_idx = max(0, min(N_time - 1, anchor_idx))
    
    t_axis = (torch.arange(N_time, dtype=f_axis.dtype, device=f_axis.device) - anchor_idx) * dt_val
    N_freq = len(f_axis)
    D = torch.zeros((1, N_freq, N_time), dtype=f_axis.dtype, device=f_axis.device)
    
    # Базовые стартовые значения (позже они будут обучаться)
    if anchor_idx > 0: 
        D[..., :anchor_idx] = 50.0 
    if anchor_idx < N_time:
        base_decay = -5.0
        freq_penalty = (f_axis / (f_axis.max() + 1e-6)).unsqueeze(1) 
        D[..., anchor_idx:] = base_decay - (15.0 * freq_penalty) 
    
    return SpecFunc(matrix=(D, UREG.hertz), freq=f_axis, time=t_axis)

def build_causal_echo_window(derivatives_spec: SpecFunc) -> SpecFunc:
    """Строит физически корректное окно эха (нормированное по энергии) из производных."""
    D, _ = derivatives_spec.values
    t_axis, _ = derivatives_spec.time
    dt_val, _ = derivatives_spec.dt
    
    Z = _get_zero_index(t_axis)
    D_left, D_right = D[..., :Z], D[..., Z:]
    
    # Интегрирование вправо
    if D_right.shape[-1] > 0:
        X_right = torch.cumsum(D_right, dim=-1) * dt_val
        X_right = X_right - X_right[..., 0:1]
    else: 
        X_right = torch.empty((*D.shape[:-1], 0), dtype=D.dtype, device=D.device)
        
    # Интегрирование влево
    if D_left.shape[-1] > 0:
        X_left = torch.flip(torch.cumsum(torch.flip(D_left, dims=[-1]), dim=-1) * (-dt_val), dims=[-1])
    else: 
        X_left = torch.empty((*D.shape[:-1], 0), dtype=D.dtype, device=D.device)
        
    # Сборка экспоненты
    echo_matrix = torch.exp(torch.cat([X_left, X_right], dim=-1))
    
    # Нормировка энергии (сумма=1 для каждой частоты)
    echo_matrix = echo_matrix / (echo_matrix.sum(dim=-1, keepdim=True) + 1e-8) 
    
    return SpecFunc(matrix=(echo_matrix, UREG.dimensionless), freq=derivatives_spec.freq[0], time=t_axis)

def spectral_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Комбинированная функция потерь: Линейный MSE (амплитуды) + Логарифмический MSE (структура)."""
    loss_linear = F.mse_loss(pred, target)
    loss_log = F.mse_loss(torch.log(pred + eps), torch.log(target + eps))
    return loss_linear + 0.1 * loss_log


# =====================================================================
# ОСНОВНАЯ ФУНКЦИЯ ДЕКОНВОЛЮЦИИ
# =====================================================================
def deconvolveEcho(
    spec: SpecFunc,
    echo_duration_ms: float = 400.0,
    anchor_pct: float = 10.0,
    num_iterations: int = 1000,
    lr_signal: float = 0.1,
    lr_echo: float = 0.1,
    lambda_sparsity: float = 5e-4,
    lambda_smooth_echo: float = 1e-3,
    device: Optional[Any] = None,
    verbose: bool = True
) -> Tuple[SpecFunc, SpecFunc]:
    """
    Слепая деконволюция для очистки спектрограммы от эха (Physics-Informed AI).
    
    Модель: S_observed ≈ S_clean * E_echo 
    Алгоритм обучается как сигналам S_clean, так и параметрам затухания среды E_echo,
    соблюдая строгие физические законы (каузальность, сохранение энергии, монотонность затухания).
    
    Args:
        spec: Исходная зашумленная/искаженная спектрограмма.
        echo_duration_ms: Длина окна эха в миллисекундах.
        anchor_pct: Процент от начала окна, где находится якорь (0 мс). Например, 10.0.
        num_iterations: Количество эпох оптимизации (рекомендуется 500-1000).
        lr_signal: Скорость обучения для чистого сигнала.
        lr_echo: Скорость обучения для профиля среды (производных).
        lambda_sparsity: Сила L1-регуляризации (чем выше, тем чище фон, но могут пропасть тихие звуки).
        lambda_smooth_echo: Штраф за резкое изменение свойств эха между соседними частотами.
        device: Устройство для вычислений (по умолчанию берется из входного тензора).
        verbose: Показывать ли прогресс-бар.
        
    Returns:
        Tuple[SpecFunc, SpecFunc]: Очищенный сигнал (S_clean) и профиль среды (E_echo).
    """
    
    # 1. Межфреймворковый мост: определяем исходный формат и переходим в Torch
    fw_target, dev_target = _get_fw_info(spec.values[0])
    target_device = device if device is not None else dev_target
    spec_torch = spec.to_framework('torch', device=target_device)
    
    S_unit = spec_torch.values[1]
    f_axis = spec_torch.freq[0]
    t_axis = spec_torch.time[0]
    
    # Целевой сигнал (наблюдаемый спектр), с защитой от отрицательных артефактов
    Target_tensor = F.relu(spec_torch.values[0].detach())
    
    # 2. Инициализация обучаемых параметров (Sc и Производных P)
    # Сигнал инициируем случайным шумом для поиска структуры
    Sc_tensor = (torch.rand_like(Target_tensor) * 0.1).requires_grad_(True)
    
    # Производные эха
    P_learn_spec = init_derivatives_window(spec_torch, duration_ms=echo_duration_ms, anchor_pct=anchor_pct)
    P_learn_vals = P_learn_spec.values[0].clone().detach()
    Z_learn = _get_zero_index(P_learn_spec.time[0])
    
    # Базовая стартовая гипотеза: быстрая атака, медленное затухание
    P_learn_vals[..., :Z_learn] = 20.0
    P_learn_vals[..., Z_learn:] = -3.0
    P_tensor = P_learn_vals.requires_grad_(True)

    # 3. Настройка оптимизатора
    optimizer = torch.optim.Adam([
        {'params': Sc_tensor, 'lr': lr_signal}, 
        {'params': P_tensor, 'lr': lr_echo}   
    ])

    # Подключаем tqdm, если нужен прогресс-бар
    if verbose:
        try:
            from tqdm.auto import tqdm
            progress_bar = tqdm(range(num_iterations), desc="Очистка эха (Physics-Informed)")
        except ImportError:
            warnings.warn("Модуль tqdm не установлен. Вывод прогресса будет упрощенным.")
            progress_bar = range(num_iterations)
    else:
        progress_bar = range(num_iterations)
        
    # 4. Основной цикл обучения
    for i in progress_bar:
        optimizer.zero_grad()

        # Создаем окно эха из текущих производных
        P_learn_spec.values = (P_tensor, UREG.hertz)
        Ec = build_causal_echo_window(P_learn_spec)

        # Сигнал должен быть строго положительным
        Sc_positive = F.relu(Sc_tensor)
        Sc = SpecFunc((Sc_positive, S_unit), f_axis, t_axis)

        # Свертка (Предсказание)
        Pred = convolveTime(Sc, Ec)
        Pred_tensor = Pred.values[0]

        # Вычисление функции потерь
        loss_recon = spectral_loss(Pred_tensor, Target_tensor, eps=1e-3)
        loss_l1 = torch.abs(Sc_positive).mean()
        loss_p_smooth = torch.diff(P_tensor, dim=1).pow(2).mean()

        total_loss = loss_recon + (lambda_sparsity * loss_l1) + (lambda_smooth_echo * loss_p_smooth)
        
        # Шаг оптимизации
        total_loss.backward()
        optimizer.step()

        # =========================================================
        # ЖЕСТКОЕ ФИЗИЧЕСКОЕ ПРОЕЦИРОВАНИЕ
        # =========================================================
        with torch.no_grad():
            # 1. До якоря (атака): производная обязана быть строго положительной
            P_tensor[..., :Z_learn].clamp_(min=0.1)
            
            # 2. После якоря (затухание): производная обязана быть строго отрицательной
            P_tensor[..., Z_learn:].clamp_(max=-0.1)
        # =========================================================

        # Обновление прогресс-бара
        if verbose and hasattr(progress_bar, 'set_postfix') and ((i + 1) % 10 == 0 or i == num_iterations - 1):
            progress_bar.set_postfix(
                loss=f"{total_loss.item():.4f}", 
                recon=f"{loss_recon.item():.4f}",
                smooth=f"{loss_p_smooth.item():.4f}"
            )

    # 5. Сборка финальных результатов
    # Итоговый очищенный сигнал
    Sc_final_vals = F.relu(Sc_tensor).detach()
    clean_spec_torch = SpecFunc((Sc_final_vals, S_unit), f_axis, t_axis)
    
    # Итоговый профиль эха среды
    P_learn_spec.values = (P_tensor.detach(), UREG.hertz)
    echo_profile_torch = build_causal_echo_window(P_learn_spec)

    # 6. Возврат в исходный фреймворк (Numpy / TF / Torch CPU)
    clean_spec = clean_spec_torch.to_framework(fw_target, dev_target)
    echo_profile = echo_profile_torch.to_framework(fw_target, dev_target)

    return clean_spec, echo_profile