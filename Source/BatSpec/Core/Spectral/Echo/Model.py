import torch
from matplotlib import pyplot as plt

from BatSpec.Core.Physical.Units import UREG
from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.ConvFFT import convolveTime

def _get_zero_index(time_axis: torch.Tensor) -> int:
    """Находит индекс якоря: точку на оси времени, максимально близкую к 0."""
    return int(torch.argmin(torch.abs(time_axis)))

def init_derivatives_window(
    ref_spec: SpecFunc, 
    duration_ms: float, 
    anchor_pct: float
) -> SpecFunc:
    """
    Создает стартовое окно производных (D) для генерации казуального эха.
    
    Args:
        ref_spec: Исходная спектрограмма (отсюда берем dt, оси частот).
        duration_ms: Желаемая длительность окна эха в миллисекундах.
        anchor_pct: Процент от начала окна, где будет находиться якорь (время = 0).
                    Например, 20% означает, что атака займет 20% времени, а затухание 80%.
                    
    Returns:
        SpecFunc: Спектрограмма, содержащая скорости роста/затухания (производные).
    """
    dt_val, dt_unit = ref_spec.dt
    f_axis, f_unit = ref_spec.freq
    
    # 1. Рассчитываем размерности по оси времени
    duration_s = duration_ms / 1000.0
    N_time = max(2, int(round(duration_s / dt_val)))
    
    # 2. Находим индекс якоря
    anchor_idx = int(round(N_time * (anchor_pct / 100.0)))
    # Защита от выхода за границы
    anchor_idx = max(0, min(N_time - 1, anchor_idx))
    
    # 3. Создаем новую ось времени так, чтобы в anchor_idx был 0
    t_axis = (torch.arange(N_time, dtype=f_axis.dtype, device=f_axis.device) - anchor_idx) * dt_val
    
    # 4. Инициализируем матрицу производных (батч 1, частоты, время)
    N_freq = len(f_axis)
    D = torch.zeros((1, N_freq, N_time), dtype=f_axis.dtype, device=f_axis.device)
    
    # 5. Заполняем базовыми физичными значениями
    # До нуля (атака): производная положительная (резко нарастает)
    if anchor_idx > 0:
        D[..., :anchor_idx] = 50.0 
        
    # После нуля (затухание): производная отрицательная.
    # Сделаем так, чтобы высокие частоты затухали быстрее (физичная реверберация).
    if anchor_idx < N_time:
        base_decay = -5.0
        # Нормируем частоты от 0 до 1
        freq_penalty = (f_axis / (f_axis.max() + 1e-6)).unsqueeze(1) 
        # Высокие частоты получат дополнительный минус к производной (быстрее затухнут)
        decay_rates = base_decay - (15.0 * freq_penalty) 
        D[..., anchor_idx:] = decay_rates
    
    # Единицы измерения производной от экспоненциального показателя 1/с (Hz)
    return SpecFunc(matrix=(D, UREG.hertz), freq=f_axis, time=t_axis)


def build_causal_echo_window(derivatives_spec: SpecFunc) -> SpecFunc:
    """
    Дифференцируемая функция генерации эха из окна производных.
    Градиенты свободно протекают через эту функцию в исходные производные.
    
    Args:
        derivatives_spec: SpecFunc с производными (положительные до 0, отрицательные после).
        
    Returns:
        SpecFunc: Окно реверберации со значениями от 0 до 1. Максимум = 1 в точке 0.
    """
    D, d_unit = derivatives_spec.values
    t_axis, t_unit = derivatives_spec.time
    dt_val, _ = derivatives_spec.dt
    
    Z = _get_zero_index(t_axis)
    
    # Разделяем матрицу на "левую" (до якоря) и "правую" (от якоря и дальше) части
    D_left = D[..., :Z]
    D_right = D[..., Z:]
    
    # 1. ИНТЕГРИРУЕМ ВПРАВО (t >= 0)
    if D_right.shape[-1] > 0:
        # Прямая кумулятивная сумма
        X_right = torch.cumsum(D_right, dim=-1) * dt_val
        # Сдвигаем так, чтобы в точке Z (первый элемент X_right) значение было равно 0
        X_right = X_right - X_right[..., 0:1]
    else:
        X_right = torch.empty((*D.shape[:-1], 0), dtype=D.dtype, device=D.device)
        
    # 2. ИНТЕГРИРУЕМ ВЛЕВО (t < 0)
    if D_left.shape[-1] > 0:
        # Переворачиваем ось времени, чтобы двигаться от Z назад
        D_left_rev = torch.flip(D_left, dims=[-1])
        # При движении назад шаг времени равен -dt_val
        X_left_rev = torch.cumsum(D_left_rev, dim=-1) * (-dt_val)
        # Переворачиваем обратно в нормальный ход времени
        X_left = torch.flip(X_left_rev, dims=[-1])
    else:
        X_left = torch.empty((*D.shape[:-1], 0), dtype=D.dtype, device=D.device)
        
    # 3. СОБИРАЕМ ИНТЕГРАЛ X
    X = torch.cat([X_left, X_right], dim=-1)
    
    # 4. ПЕРЕХОД К ФОРМЕ ЭХА (Exp)
    echo_matrix = torch.exp(X)
    
    # 5. НОРМАЛИЗАЦИЯ
    # Строго по ТЗ нормируем на 1 (хотя в точке 0 интеграл и так равен 0, а exp(0)=1,
    # это защитит от смещений, если нейросеть выдаст шум).
    max_vals = echo_matrix.max(dim=-1, keepdim=True).values
    echo_matrix = echo_matrix / (max_vals + 1e-8)
    
    # Возвращаем безразмерное окно, оси остаются теми же
    return SpecFunc(
        matrix=(echo_matrix, UREG.dimensionless), 
        freq=derivatives_spec.freq[0], 
        time=t_axis
    )


# =====================================================================
# ТЕСТИРОВАНИЕ И ПРОВЕРКА ГРАДИЕНТОВ
# =====================================================================
if __name__ == "__main__":
    print("--- Тест: Генерация дифференцируемого профиля эха ---")
    
    # 1. Создаем фиктивную референсную спектрограмму (от нее возьмется только сетка)
    time_ref = torch.linspace(0, 2.0, 400) # 2 секунды, dt = 0.005
    freq_ref = torch.linspace(0, 100, 64)  # 64 бина частот
    mat_ref = torch.zeros(1, 64, 400)
    
    ref_spec = SpecFunc(matrix=(mat_ref, UREG.dimensionless), freq=freq_ref, time=time_ref)
    
    # 2. Инициализируем окно производных (длина 300 мс, якорь на 10%)
    D_spec = init_derivatives_window(ref_spec, duration_ms=300.0, anchor_pct=10.0)
    
    # ВАЖНО: Включаем градиенты у тензора значений производных!
    # Имитируем ситуацию, когда D - это выход нейросети (или nn.Parameter)
    D_tensor, u = D_spec.values
    D_tensor.requires_grad = True
    
    # Обновляем объект с тензором, у которого requires_grad=True
    D_spec.values = (D_tensor, u)
    
    # 3. Прогоняем через нашу дифференцируемую функцию
    Echo_spec = build_causal_echo_window(D_spec)
    Echo_tensor, _ = Echo_spec.values
    
    # 4. Проверим протекание градиентов (Backward pass)
    # Вычислим какую-нибудь "loss" (например, сумму значений)
    loss = Echo_tensor.sum()
    loss.backward()
    
    print(f"Loss (сумма окна): {loss.item():.4f}")
    print(f"Градиенты дошли до производных D_tensor? {'Да' if D_tensor.grad is not None else 'Нет'}")
    print(f"Сумма абсолютных градиентов: {D_tensor.grad.abs().sum().item():.4f}")

    # 5. Отрисовка результатов
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    t_axis = D_spec.time[0].detach().numpy()
    f_axis = D_spec.freq[0].detach().numpy()
    extent = [t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]]
    
    # Рисуем инициализированные производные
    im0 = axs[0].imshow(D_tensor[0].detach().numpy(), aspect='auto', origin='lower', extent=extent, cmap='RdBu_r')
    axs[0].set_title("Производные D (До 0: >0, После 0: <0)")
    axs[0].axvline(0, color='w', linestyle='--', alpha=0.5)
    fig.colorbar(im0, ax=axs[0])
    
    # Рисуем итоговое окно эха
    im1 = axs[1].imshow(Echo_tensor[0].detach().numpy(), aspect='auto', origin='lower', extent=extent, cmap='magma')
    axs[1].set_title("Окно эха (Exp(Интеграл D))")
    axs[1].axvline(0, color='w', linestyle='--', alpha=0.5)
    fig.colorbar(im1, ax=axs[1])
    
    plt.tight_layout()
    plt.show()
    
    # 6. Покажем профиль (срез) по низкой и высокой частоте
    plt.figure(figsize=(8, 4))
    plt.plot(t_axis, Echo_tensor[0, 0, :].detach().numpy(), label=f'Низкая частота ({f_axis[0]:.1f} Hz)')
    plt.plot(t_axis, Echo_tensor[0, -1, :].detach().numpy(), label=f'Высокая частота ({f_axis[-1]:.1f} Hz)')
    plt.axvline(0, color='k', linestyle='--', alpha=0.3, label='Время 0 (Якорь)')
    plt.title("Срез казуального окна эха во времени")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда окна")
    plt.legend()
    plt.grid(True)
    plt.show()