

**Ты можешь создавать файлы.**
Контент файлов ты должен оборачивать в специальную маркдаун конструкцию.
``````{{ ext }} path="{{ path }}" encoding="{{ encoding }}"
{{ content }}
``````

**Внимание!**
Количество ` должно быть ровно 6.
СТРОГО СЛЕДИ за этим, так как эта конструкция парсится программно.

**Пример:**
``````py path="HelloWorld.py" encoding="utf-8"
print("Hello world!")

``````

**Пояснения к формату:**
*   `ext`: стандартный MarkDown формат (например, `py`, `html`, `css`, `js`).
*   `path`:  абсолютный или относительный путь к файлу (например, `./src/components/Button.js`).
*   `encoding`: кодировка файла.
*   `content`: **полный и готовый к использованию код** файла.

---
**КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:**

1.  **ПРАВИЛО ПЕРЕНОСА СТРОКИ ПОСЛЕ БЛОКА:** После каждого закрывающего блока `````` **ВСЕГДА** должен быть как минимум один перенос строки.

2.  **ПРАВИЛО ЗАВЕРШАЮЩЕГО ПЕРЕНОСА СТРОКИ В КОНТЕНТЕ:** Содержимое (`content`) **КАЖДОГО** файла **ОБЯЗАТЕЛЬНО** должно заканчиваться как минимум одним переносом строки.

3.  **ПРАВИЛО ВЫБОРА КОДИРОВКИ (ИСПРАВЛЕНО):**
    *   Для файлов PowerShell (`.psd1`, `.psm1`) используй кодировку **`utf-16`**. Это заставит Python добавить необходимый BOM.
    *   Для файлов .ps1 используй кодировку `utf_8_sig`. Она также добавляет BOM, что является хорошей практикой для PowerShell.
    *   Для большинства других текстовых файлов (`.md`, `.json`, `.py`, `.js` и т.д.) используй кодировку `utf-8`.
---

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/ConvFFT/__init__.py" encoding="utf-8"
from matplotlib import pyplot as plt
import torch
import torch.fft
from typing import Union

from BatSpec.Core.Functions import SpecFunc, TimeFunc
from BatSpec.Core.Spectral.Some import matchPearsonFT
from BatSpec.Core.Physical.Units import UREG, unit_mul

def _get_zero_index(time_axis: torch.Tensor) -> int:
    """Находит индекс якоря: точку на оси времени, максимально близкую к 0."""
    return int(torch.argmin(torch.abs(time_axis)))

def conv_T_T(sig: TimeFunc, ker: TimeFunc) -> TimeFunc:
    """1D Сигнал * 1D Окно -> 1D Сигнал"""
    sig_v, sig_u = sig.values
    ker_v, ker_u = ker.values
    
    T_sig = sig_v.shape[-1]
    T_ker = ker_v.shape[-1]
    N = T_sig + T_ker - 1
    
    # FFT
    Sig_f = torch.fft.rfft(sig_v, n=N, dim=-1)
    Ker_f = torch.fft.rfft(ker_v, n=N, dim=-1)
    out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
    
    # Якорь берем из ядра
    Z = _get_zero_index(ker.time[0])
    out = out[..., Z : Z + T_sig]
    
    # Физическое интегрирование (умножаем на dt ядра)
    dt_val, dt_unit = ker.dt
    out = out * dt_val
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    return TimeFunc(values=(out, new_unit), axis=sig.time[0])

def conv_S_T(sig: SpecFunc, ker: TimeFunc) -> SpecFunc:
    """2D Спектрограмма * 1D Окно -> 2D Спектрограмма (Фильтрация спектрограммы во времени)"""
    sig_v, sig_u = sig.values
    ker_v, ker_u = ker.values
    
    T_sig = sig_v.shape[-1]
    T_ker = ker_v.shape[-1]
    N = T_sig + T_ker - 1
    
    # Добавляем фиктивную ось частот для ядра: (..., T) -> (..., 1, T)
    ker_v_expanded = ker_v.unsqueeze(-2)
    
    Sig_f = torch.fft.rfft(sig_v, n=N, dim=-1)
    Ker_f = torch.fft.rfft(ker_v_expanded, n=N, dim=-1)
    
    # Broadcasting: (..., F, N_freq) * (..., 1, N_freq)
    out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
    
    Z = _get_zero_index(ker.time[0])
    out = out[..., Z : Z + T_sig]
    
    dt_val, dt_unit = ker.dt
    out = out * dt_val
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    return SpecFunc(matrix=(out, new_unit), freq=sig.freq[0], time=sig.time[0])

def conv_T_S(sig: TimeFunc, ker: SpecFunc) -> SpecFunc:
    """1D Сигнал * 2D Окно -> 2D Спектрограмма (Генерация: Активации * Шаблоны)"""
    sig_v, sig_u = sig.values
    ker_v, ker_u = ker.values
    
    T_sig = sig_v.shape[-1]
    T_ker = ker_v.shape[-1]
    N = T_sig + T_ker - 1
    
    # Добавляем фиктивную ось частот для сигнала: (..., T) -> (..., 1, T)
    sig_v_expanded = sig_v.unsqueeze(-2)
    
    Sig_f = torch.fft.rfft(sig_v_expanded, n=N, dim=-1)
    Ker_f = torch.fft.rfft(ker_v, n=N, dim=-1)
    
    # Broadcasting: (..., 1, N_freq) * (..., F, N_freq)
    out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
    
    Z = _get_zero_index(ker.time[0])
    out = out[..., Z : Z + T_sig]
    
    dt_val, dt_unit = ker.dt
    out = out * dt_val
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    # Ось времени берем от сигнала (так как он задает длину), а частоты - от ядра (шаблона)
    return SpecFunc(matrix=(out, new_unit), freq=ker.freq[0], time=sig.time[0])

def conv_S_S(sig: SpecFunc, ker: SpecFunc) -> SpecFunc:
    """
    2D Спектрограмма * 2D Окно -> 2D Спектрограмма
    Внимание: это 1D-свертка по оси времени для каждой частоты независимо!
    (Модель реверберации: поэлементное умножение по частоте, свертка по времени).
    """
    sig_v, sig_u = sig.values
    ker_v, ker_u = ker.values
    
    # Проверка, что частотные оси совпадают по размеру
    assert sig_v.shape[-2] == ker_v.shape[-2], "Размерности частот сигнала и ядра должны совпадать"
    
    T_sig = sig_v.shape[-1]
    T_ker = ker_v.shape[-1]
    N = T_sig + T_ker - 1
    
    Sig_f = torch.fft.rfft(sig_v, n=N, dim=-1)
    Ker_f = torch.fft.rfft(ker_v, n=N, dim=-1)
    
    # Элементное умножение по F и N_freq
    out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
    
    Z = _get_zero_index(ker.time[0])
    out = out[..., Z : Z + T_sig]
    
    dt_val, dt_unit = ker.dt
    out = out * dt_val
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    return SpecFunc(matrix=(out, new_unit), freq=sig.freq[0], time=sig.time[0])

# =====================================================================
# Удобный диспетчер (Overloading)
# =====================================================================
def convolveTime(a: Union[TimeFunc, SpecFunc], b: Union[TimeFunc, SpecFunc]) -> Union[TimeFunc, SpecFunc]:
    """
    Универсальная функция свёртки. a - сигнал, b - ядро(фильтр).
    Определяет тип операции автоматически.
    """
    if isinstance(a, TimeFunc) and isinstance(b, TimeFunc):
        return conv_T_T(a, b)
    elif isinstance(a, SpecFunc) and isinstance(b, TimeFunc):
        return conv_S_T(a, b)
    elif isinstance(a, TimeFunc) and isinstance(b, SpecFunc):
        return conv_T_S(a, b)
    elif isinstance(a, SpecFunc) and isinstance(b, SpecFunc):
        return conv_S_S(a, b)
    else:
        raise TypeError(f"Неподдерживаемые типы для свёртки: {type(a)} и {type(b)}")
    
def convolveFT(sig: SpecFunc, ker: SpecFunc) -> SpecFunc:
    """
    ИСТИННАЯ 2D СВЁРТКА: 2D Спектрограмма * 2D Окно -> 2D Спектрограмма.
    Свертка происходит одновременно по осям Времени и Частоты (через 2D FFT).
    """
    sig_v, sig_u = sig.values
    ker_v, ker_u = ker.values
    
    # Размеры по частоте (F) и времени (T)
    F_sig, T_sig = sig_v.shape[-2], sig_v.shape[-1]
    F_ker, T_ker = ker_v.shape[-2], ker_v.shape[-1]
    
    # Размеры для 2D FFT (по теореме о свертке)
    N_f = F_sig + F_ker - 1
    N_t = T_sig + T_ker - 1
    
    # Применяем 2D вещественное БПФ
    # s=(N_f, N_t) указывает размерности, dim=(-2, -1) указывает оси
    Sig_fft = torch.fft.rfft2(sig_v, s=(N_f, N_t), dim=(-2, -1))
    Ker_fft = torch.fft.rfft2(ker_v, s=(N_f, N_t), dim=(-2, -1))
    
    # Умножаем в частотной (Фурье) области и возвращаем в пространственно-временную
    out = torch.fft.irfft2(Sig_fft * Ker_fft, s=(N_f, N_t), dim=(-2, -1))
    
    # Находим физические якоря (индексы нулей) в ядре по ОБЕИМ осям
    Z_f = _get_zero_index(ker.freq[0])
    Z_t = _get_zero_index(ker.time[0])
    
    # Отрезаем рамки: размер выходного сигнала равен размеру входного сигнала
    out = out[..., Z_f : Z_f + F_sig, Z_t : Z_t + T_sig]
    
    # Физическое интегрирование: двойной интеграл df dt
    df_val, df_unit = ker.df
    dt_val, dt_unit = ker.dt
    
    out = out * (df_val * dt_val)
    new_unit = unit_mul(sig_u, ker_u, df_unit, dt_unit)
    
    # Возвращаем новую спектрограмму, сохраняя оси исходного сигнала
    return SpecFunc(matrix=(out, new_unit), freq=sig.freq[0], time=sig.time[0])


if __name__ == "__main__":
    # --- ОБЩИЕ ПАРАМЕТРЫ ---
    TIME_STEPS = 1000
    FREQ_BINS = 64
    
    t_axis = torch.linspace(-5.0, 5.0, TIME_STEPS)
    f_axis = torch.linspace(0.0, 100.0, FREQ_BINS)
    
    unit = UREG.dimensionless
    extent_S = [t_axis[0].item(), t_axis[-1].item(), f_axis[0].item(), f_axis[-1].item()]

    # =================================================================
    # 1. ТЕСТ: T * T (Аналитическая свертка Лоренциан)
    # =================================================================
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


    # =================================================================
    # 2. ТЕСТ: S * T (Фильтрация спектрограммы во времени)
    # =================================================================
    print("--- 2. ТЕСТ (S * T): Сглаживание шумной спектрограммы ---")
    
    # Сигнал S: Диагональная линия (чирп) + сильный шум
    F_grid, T_grid = torch.meshgrid(f_axis, t_axis, indexing='ij')
    clean_chirp = torch.exp(-((F_grid - (T_grid + 5)*10)**2) / 20.0)
    noise = torch.rand_like(clean_chirp) * 0.5
    S_noisy_vals = (clean_chirp + noise).unsqueeze(0)
    
    # Ядро T: Гауссово окно сглаживания (ширина ~0.2 сек)
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


    # =================================================================
    # 3. ТЕСТ: T * S (Генеративная модель)
    # =================================================================
    print("--- 3. ТЕСТ (T * S): Размещение шаблонов по спайкам ---")
    
    # Сигнал T: Три спайка в разное время (-3, 0, 2)
    T_spikes_vals = torch.zeros_like(t_axis)
    T_spikes_vals[torch.argmin(torch.abs(t_axis - (-3.0)))] = 1.0
    T_spikes_vals[torch.argmin(torch.abs(t_axis - 0.0))] = 0.5   # Тихий спайк
    T_spikes_vals[torch.argmin(torch.abs(t_axis - 2.0))] = 1.2   # Громкий спайк
    T_spikes_vals = T_spikes_vals.unsqueeze(0)
    
    # Шаблон S: Капля/Радио-импульс на частоте 60 Гц
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


    # =================================================================
    # 4. ТЕСТ: S * S (Реверберация среды)
    # =================================================================
    print("--- 4. ТЕСТ (S * S): Физичное эхо (частотно-зависимое) ---")
    
    # Сигнал S: Тот же чирп, но чистый
    S_clean_vals = clean_chirp.unsqueeze(0)
    
    # Эхо S: Экспоненциальное затухание. 
    # Физика: Высокие частоты (F) затухают быстрее (t * F/20). Эхо только для t >= 0.
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
    axs[1].set_xlim(-1, 3) # Приблизим окно эха
    axs[2].imshow(res_S_S.values[0][0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='plasma')
    axs[2].set_title("Результат S * S (Шлейф эха)")
    plt.tight_layout()

    # =================================================================
    # 5. ТЕСТ: convolveFT (Истинная 2D свёртка S * S)
    # =================================================================
    print("\n--- 5. ТЕСТ (convolveFT): 2D Размытие спектрограммы ---")
    
    # 1. Сигнал S: Три точки-импульса в координатах (F, T)
    S_points = torch.zeros(1, FREQ_BINS, TIME_STEPS)
    # Ставим точки в случайных местах
    S_points[0, 15, 200] = 1.0  # Низкая частота, начало
    S_points[0, 45, 500] = 0.8  # Высокая частота, середина
    S_points[0, 30, 800] = 1.2  # Средняя частота, конец
    
    sig_S_2d = SpecFunc(matrix=(S_points, unit), freq=f_axis, time=t_axis)
    
    # 2. Ядро S: 2D Гауссово пятно
    # Создаем ОТДЕЛЬНЫЕ маленькие оси для ядра, центрированные вокруг 0!
    f_ker_axis = torch.linspace(-30.0, 30.0, 31) # от -30 до 30 Гц
    t_ker_axis = torch.linspace(-1.0, 1.0, 51)   # от -1 до 1 сек
    
    F_ker_grid, T_ker_grid = torch.meshgrid(f_ker_axis, t_ker_axis, indexing='ij')
    # Формула 2D Гауссианы
    S_gauss_2d = torch.exp(- (F_ker_grid**2 / 100.0) - (T_ker_grid**2 / 0.05)).unsqueeze(0)
    
    ker_S_2d = SpecFunc(matrix=(S_gauss_2d, unit), freq=f_ker_axis, time=t_ker_axis)
    
    # 3. Выполняем 2D свёртку
    res_FT = convolveFT(sig_S_2d, ker_S_2d)
    
    # 4. Отрисовка
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    axs[0].imshow(S_points[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='inferno')
    axs[0].set_title("Сигнал S (Одиночные точки)")
    
    # Ядро показываем в его собственных осях
    ker_extent = [t_ker_axis[0].item(), t_ker_axis[-1].item(), f_ker_axis[0].item(), f_ker_axis[-1].item()]
    axs[1].imshow(S_gauss_2d[0].numpy(), aspect='auto', origin='lower', extent=ker_extent, cmap='inferno')
    axs[1].set_title("Ядро S (2D Гауссиана центрированная в 0, 0)")
    axs[1].axhline(0, color='w', alpha=0.5, linestyle='--')
    axs[1].axvline(0, color='w', alpha=0.5, linestyle='--')
    
    axs[2].imshow(res_FT.values[0][0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='inferno')
    axs[2].set_title("Результат convolveFT (2D Свертка)")
    
    plt.tight_layout()

    
    # =================================================================
    # 6. ТЕСТ: matchPearsonFT (Поиск шаблона NCC)
    # =================================================================
    print("\n--- 6. ТЕСТ (matchPearsonFT): Поиск шаблона Пирсона в шуме ---")
    
    # Создаем базовые оси для паттерна (центрированные)
    f_patt_axis = torch.linspace(-20.0, 20.0, 21)
    t_patt_axis = torch.linspace(-0.5, 0.5, 51)
    F_patt_grid, T_patt_grid = torch.meshgrid(f_patt_axis, t_patt_axis, indexing='ij')
    
    # Шаблон: Сложная фигура (Крест)
    pattern_vals = torch.exp(-(F_patt_grid**2)/5.0) * torch.exp(-(T_patt_grid**2)/0.5) + \
                   torch.exp(-(F_patt_grid**2)/200.0) * torch.exp(-(T_patt_grid**2)/0.005)
    pattern_vals = pattern_vals.unsqueeze(0)
    
    # Исходный сигнал: чистый белый ШУМ [0, 1]
    S_search_vals = torch.rand(1, FREQ_BINS, TIME_STEPS) / 1000
    
    # Вживляем паттерн в сигнал в случайное место (индексы 40 по частоте, 700 по времени)
    f_insert, t_insert = 40, 700
    F_p, T_p = pattern_vals.shape[-2], pattern_vals.shape[-1]
    
    # Добавляем паттерн (накладываем поверх шума)
    S_search_vals[0, f_insert : f_insert + F_p, t_insert : t_insert + T_p] += pattern_vals[0]
    
    # Создаем классы
    sig_search = SpecFunc(matrix=(S_search_vals, unit), freq=f_axis, time=t_axis)
    ker_pattern = SpecFunc(matrix=(pattern_vals, unit), freq=f_patt_axis, time=t_patt_axis)
    
    # ВЫПОЛНЯЕМ ПОИСК
    res_pearson = matchPearsonFT(sig_search, ker_pattern)
    map_vals = res_pearson.values[0][0].numpy()
    
    # Отрисовка
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    axs[0].imshow(S_search_vals[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='gray')
    axs[0].set_title("Сигнал S (Шум + Вживленный крест)")
    
    patt_extent = [t_patt_axis[0].item(), t_patt_axis[-1].item(), f_patt_axis[0].item(), f_patt_axis[-1].item()]
    axs[1].imshow(pattern_vals[0].numpy(), aspect='auto', origin='lower', extent=patt_extent, cmap='magma')
    axs[1].set_title("Искомый шаблон (Ядро)")
    
    im = axs[2].imshow(map_vals, aspect='auto', origin='lower', extent=extent_S, cmap='RdBu_r', vmin=-1, vmax=1)
    axs[2].set_title("Карта Пирсона R [-1, 1]")
    fig.colorbar(im, ax=axs[2])
    
    plt.tight_layout()
    plt.show()
    
    # Проверяем максимум Пирсона
    max_R = map_vals.max()
    print(f"Максимальный коэффициент Пирсона: {max_R:.4f}")
    if max_R > 0.9:
        print("Патерн успешно найден!")
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/Saprse/second.py" encoding="utf-8"
from typing import Tuple, Optional, Callable
from tqdm.auto import tqdm

# --- Внедряем наш универсальный фреймворк ---
import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType

from BatSpec.Core.Functions import SpecFunc, TimeFunc
from BatSpec.Core.Physical.Units import UREG

# Попытка импорта PyTorch (обязателен для градиентной оптимизации паттернов)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Работают строго в PyTorch)
# =====================================================================

def _functional_conv1d_fft_2dkernel(signal: 'torch.Tensor', kernel: 'torch.Tensor') -> 'torch.Tensor':
    """
    Вычисляет 1D свертку 2D сигнала и 2D ядра ТОЛЬКО по временной оси (последней).
    Ось частот должна совпадать.
    """
    assert signal.shape[-2] == kernel.shape[-2], "Размерность частот сигнала и ядра должна совпадать"
    
    T_sig, T_ker = signal.shape[-1], kernel.shape[-1]
    N_t = T_sig + T_ker - 1

    Sig_f = torch.fft.rfft(signal, n=N_t, dim=-1)
    Ker_f = torch.fft.rfft(kernel, n=N_t, dim=-1)
    
    out_f = Sig_f * Ker_f
    out = torch.fft.irfft(out_f, n=N_t, dim=-1)

    pad_t_start = (T_ker - 1) // 2
    return out[..., pad_t_start : pad_t_start + T_sig]

def _functional_match1d_fft_2dkernel(signal: 'torch.Tensor', kernel: 'torch.Tensor') -> 'torch.Tensor':
    """
    Вычисляет 1D кросс-корреляцию 2D сигнала и 2D ядра по временной оси.
    """
    # Для корреляции через свертку нужно отзеркалить ядро по оси свертки
    flipped_kernel = torch.flip(kernel, dims=(-1,))
    return _functional_conv1d_fft_2dkernel(signal, flipped_kernel)

def _total_variation_loss(img: 'torch.Tensor') -> 'torch.Tensor':
    """Заставляет паттерн быть гладким, штрафуя резкие перепады."""
    tv_h = torch.sum(torch.abs(img[..., :-1, :] - img[..., 1:, :]))
    tv_w = torch.sum(torch.abs(img[..., :, :-1] - img[..., :, 1:]))
    return tv_h + tv_w


# =====================================================================
# ОСНОВНАЯ ФУНКЦИЯ ПОИСКА ПАТТЕРНА
# =====================================================================

def discover_best_pattern_with_callback(
    spec: SpecFunc,
    window_duration_ms: float = 100.0,
    anchor_freq_percent: float = 50.0,
    anchor_time_percent: float = 50.0,
    epochs: int = 500,
    learning_rate: float = 0.05,
    top_k_peaks: int = 20,
    lambda_tv: float = 0.01,
    lambda_center: float = 0.1,
    callback: Optional[Callable[[SpecFunc, TimeFunc], None]] = None,
    callback_interval_percent: Optional[float] = 2.0
) -> Tuple[SpecFunc, TimeFunc]:
    """
    Находит один "идеальный" 2D паттерн, который наилучшим образом описывает 
    повторяющиеся события в спектрограмме, используя 1D свертку по времени.
    Автоматически переводит данные в PyTorch для обучения и возвращает в исходном фреймворке.

    Args:
        spec: Входная спектрограмма (Может быть в любом фреймворке: NumPy, TF, JAX и т.д.).
        window_duration_ms: Длительность искомого паттерна в миллисекундах.
        anchor_freq_percent: Ожидаемый центр масс по частоте (в % от высоты).
        anchor_time_percent: Ожидаемый центр масс по времени (в % от ширины).
        epochs: Количество эпох обучения.
        learning_rate: Скорость обучения.
        top_k_peaks: Сколько самых сильных откликов использовать для расчета лосса.
        lambda_tv: Вес регуляризации гладкости (Total Variation).
        lambda_center: Общий вес штрафа за отклонение от якорей.
        callback: Функция для визуализации прогресса.
        callback_interval_percent: Как часто вызывать callback (в процентах от эпох).

    Returns:
        Кортеж из (найденный_паттерн: SpecFunc, 1D_временная_карта_корреляций: TimeFunc)
        в том же фреймворке, в котором была передана спектрограмма.
    """
    if not HAS_TORCH:
        raise RuntimeError("Для градиентного поиска паттерна требуется установленный PyTorch.")

    # --- 1. Бесшовная миграция данных в PyTorch ---
    orig_ctx = spec.context
    
    dev_enum = DeviceType.GPU if orig_ctx.isGPU() and torch.cuda.is_available() else DeviceType.CPU
    pt_ctx = ArrayContext(Framework.TORCH, dev_enum, None)
    
    # Переводим спектрограмму в тензоры PyTorch на нужном девайсе
    spec_pt = spec.to_context(pt_ctx)
    
    S, S_unit = spec_pt.values
    freq_axis, _ = spec_pt.freq
    time_axis, _ = spec_pt.time
    
    device = S.device
    dtype = S.dtype
    target_spec = S.unsqueeze(0) if S.dim() == 2 else S
    
    # --- 2. Инициализация параметров ---
    dt_val, _ = spec_pt.dt
    dt_val = float(dt_val)
    sr = 1.0 / dt_val
    num_freqs, _ = target_spec.shape[-2], target_spec.shape[-1]
    win_width = max(3, int(round((window_duration_ms / 1000.0) * sr)))
    
    W_raw = nn.Parameter(torch.randn(1, num_freqs, win_width, device=device, dtype=dtype))
    optimizer = torch.optim.Adam([W_raw], lr=learning_rate)
    
    # Предрасчет для лосса центрирования
    target_center_freq = (num_freqs - 1) * (anchor_freq_percent / 100.0)
    target_center_time = (win_width - 1) * (anchor_time_percent / 100.0)
    freq_indices = torch.arange(num_freqs, device=device, dtype=dtype)
    time_indices = torch.arange(win_width, device=device, dtype=dtype)
    
    print(f"Поиск паттерна: окно {num_freqs}x{win_width}. Целевой центр (F,T): ({target_center_freq:.1f}, {target_center_time:.1f})")
    
    # --- Локальный Callback (с авто-переводом в исходный фреймворк) ---
    def _invoke_callback(current_W_raw: torch.Tensor, correlation_timeline: torch.Tensor):
        with torch.no_grad():
            W_norm = current_W_raw - current_W_raw.mean()
            W_norm /= (torch.norm(W_norm) + 1e-8)
            
            pattern_time_axis = torch.arange(win_width, device=device, dtype=dtype) * dt_val
            
            # Собираем функции в PyTorch
            found_pattern_spec = SpecFunc(
                matrix=(W_norm.squeeze(0), S_unit), 
                freq=freq_axis, 
                time=pattern_time_axis
            )
            
            correlation_timeline_func = TimeFunc(
                values=(correlation_timeline.squeeze(0), UREG.dimensionless),
                axis=time_axis
            )
            
            # Возвращаем их в тот фреймворк (NumPy/TF/и тд), из которого пришел пользователь
            found_pattern_spec = found_pattern_spec.to_context(orig_ctx)
            correlation_timeline_func = correlation_timeline_func.to_context(orig_ctx)
            
            if callback:
                callback(found_pattern_spec, correlation_timeline_func)
                
        return found_pattern_spec, correlation_timeline_func

    # --- 3. Цикл обучения (Pure PyTorch) ---
    pbar = tqdm(range(epochs), desc="Поиск паттерна")
    latest_results = None
    eps = 1e-8

    for epoch in pbar:
        optimizer.zero_grad()
        
        # --- ШАГ A: Дифференцируемая нормализация паттерна ---
        W_zero_mean = W_raw - W_raw.mean()
        W = W_zero_mean / (torch.norm(W_zero_mean) + eps)
        
        # --- ШАГ B: Дифференцируемый ZNCC (с 1D сверткой по времени) ---
        numerator = _functional_match1d_fft_2dkernel(target_spec, W)
        
        U = torch.ones_like(W); N = W.numel()
        Sum_S = _functional_match1d_fft_2dkernel(target_spec, U)
        Sum_S2 = _functional_match1d_fft_2dkernel(target_spec**2, U)
        
        local_variance = (Sum_S2 - (Sum_S**2) / N).clamp_(min=0.0)
        denominator = torch.sqrt(local_variance) + eps
        
        # Карта корреляций все еще 2D (F, T)
        zncc_map_2d = numerator / denominator
        # Схлопываем по частоте, чтобы получить 1D временную функцию
        correlation_timeline = zncc_map_2d.sum(dim=-2)
        
        # --- ШАГ C: Вычисление Loss-функции ---
        # 1. Лосс "заметности" для 1D временной функции
        top_peaks, _ = torch.topk(correlation_timeline.flatten(), k=top_k_peaks)
        loss_match = -torch.mean(top_peaks) 
        
        # 2. Лосс гладкости (TV Loss)
        loss_tv = lambda_tv * _total_variation_loss(W_raw)
        
        # 3. Лосс центрирования по двум осям
        energy_density = W_raw**2
        total_mass = energy_density.sum() + eps
        
        # Центр масс по времени
        time_density = energy_density.sum(dim=-2)
        com_time = (time_density * time_indices).sum() / total_mass
        loss_center_time = F.mse_loss(com_time, torch.full_like(com_time, target_center_time))
        
        # Центр масс по частоте
        freq_density = energy_density.sum(dim=-1)
        com_freq = (freq_density * freq_indices).sum() / total_mass
        loss_center_freq = F.mse_loss(com_freq, torch.full_like(com_freq, target_center_freq))
        
        loss_center = lambda_center * (loss_center_time + loss_center_freq)
        
        total_loss = loss_match + loss_tv + loss_center
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(Loss=f"{total_loss.item():.3f}", Match=f"{-loss_match.item():.3f}", Cntr=f"{loss_center.item():.3f}")

        is_last_epoch = (epoch == epochs - 1)
        if callback_interval_percent > 0 and ((epoch + 1) % int(epochs * callback_interval_percent / 100.0) == 0 or is_last_epoch):
            latest_results = _invoke_callback(W_raw, correlation_timeline)

    if latest_results is None:
        # Резервный сбор результатов, если callback не срабатывал
        with torch.no_grad():
            W_zero_mean = W_raw - W_raw.mean()
            W = W_zero_mean / (torch.norm(W_zero_mean) + eps)
            numerator = _functional_match1d_fft_2dkernel(target_spec, W)
            U = torch.ones_like(W); N = W.numel()
            Sum_S = _functional_match1d_fft_2dkernel(target_spec, U)
            Sum_S2 = _functional_match1d_fft_2dkernel(target_spec**2, U)
            local_variance = (Sum_S2 - (Sum_S**2) / N).clamp_(min=0.0)
            denominator = torch.sqrt(local_variance) + eps
            final_zncc_map_2d = numerator / denominator
            final_timeline = final_zncc_map_2d.sum(dim=-2)
            latest_results = _invoke_callback(W_raw, final_timeline)

    return latest_results


# =====================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ И ТЕСТИРОВАНИЕ
# =====================================================================
if __name__ == "__main__":
    if not HAS_TORCH:
        print("Для запуска теста требуется PyTorch")
        exit()
        
    F_BINS, TIME_STEPS = 128, 4000
    
    # Создаем данные сразу в контексте NumPy, чтобы проверить магию бесшовной конвертации
    import numpy as np
    spec_tensor = np.random.randn(F_BINS, TIME_STEPS).astype(np.float32) * 0.1

    pattern_width = 150
    y_coords = np.arange(F_BINS)[:, np.newaxis]
    x_coords = np.arange(pattern_width)[np.newaxis, :]
    center_freq = F_BINS * 0.7 
    chirp = np.exp(-((y_coords - (center_freq + 20 * np.sin(x_coords * 0.15)))**2) / 25)
    
    spec_tensor[:, 500:500+pattern_width] += chirp * 5.0
    spec_tensor[:, 1800:1800+pattern_width] += chirp * 3.0
    spec_tensor[:, 3000:3000+pattern_width] += chirp * 8.0

    freq_axis = np.arange(F_BINS).astype(np.float32)
    time_axis = np.arange(TIME_STEPS).astype(np.float32) * 0.01

    ctx = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    test_spec = SpecFunc(
        matrix=(ma.convert_to(spec_tensor, ctx), UREG.dimensionless), 
        freq=ma.convert_to(freq_axis, ctx), 
        time=ma.convert_to(time_axis, ctx)
    )

    def plot_callback(found_pattern: SpecFunc, correlation_timeline: TimeFunc):
        try:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 1, figsize=(15, 7), gridspec_kw={'height_ratios': [2, 1]})
            
            # Поскольку функция возвращает объекты в исходном контексте (NumPy), 
            # мы можем смело брать их значения
            pattern_img, _ = found_pattern.values
            im = axs[0].imshow(pattern_img, aspect='auto', origin='lower', cmap='viridis')
            axs[0].set_title(f'Найденный паттерн (W) {pattern_img.shape}')
            fig.colorbar(im, ax=axs[0])
            
            timeline_data, _ = correlation_timeline.values
            t_axis, _ = correlation_timeline.time
            
            axs[1].plot(t_axis, timeline_data)
            axs[1].set_title('1D Временная карта корреляций')
            axs[1].grid(True)
            axs[1].set_xlabel('Время, с')
            axs[1].set_ylabel('Суммарная корреляция')
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib не установлен, визуализация пропущена.")

    print("\n--- Запуск поиска самого характерного паттерна (NumPy -> PyTorch -> NumPy) ---")
    found_pattern, final_corr_map = discover_best_pattern_with_callback(
        spec=test_spec,
        window_duration_ms=1500.0,
        anchor_freq_percent=70.0,
        anchor_time_percent=50.0,
        epochs=150,
        learning_rate=0.05,
        top_k_peaks=15,
        lambda_tv=0.001,
        lambda_center=0.2,
        callback=plot_callback,
        callback_interval_percent=100.0  # Вызвать колбек только в конце для тестов
    )
    
    print(f"Итоговый контекст паттерна: {found_pattern.context}")
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/Saprse/__init__.py" encoding="utf-8"
from typing import Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import NMF
from tqdm.auto import tqdm

from BatSpec.Core.Functions import SpecFunc, TimeFunc
from BatSpec.Core.Physical.Units import UREG

class _FFTSpecConvDecomposer(nn.Module):
    def __init__(self, num_patterns: int, num_freqs: int, win_width: int, seq_len: int, dtype: torch.dtype):
        super().__init__()
        self.num_patterns = num_patterns
        self.W = seq_len
        self.W_win = win_width
        self.basis = nn.Parameter(torch.abs(torch.randn(num_patterns, num_freqs, win_width, dtype=dtype)) * 0.1)
        self.seq = nn.Parameter(torch.abs(torch.randn(1, num_patterns, seq_len, dtype=dtype)) * 0.01)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        active_seq = F.relu(self.seq)
        fft_size = self.W + self.W_win - 1
        seq_f = torch.fft.rfft(active_seq, n=fft_size)
        basis_f = torch.fft.rfft(self.basis, n=fft_size)
        canvas_f = torch.einsum('b n f, n h f -> b h f', seq_f, basis_f)
        canvas = torch.fft.irfft(canvas_f, n=fft_size)
        return canvas[..., :self.W], active_seq


def extract_spec_patterns_with_callback(
    spec: SpecFunc,
    num_patterns: int = 6,
    num_learn: int = 12,
    window_duration_ms: float = 100.0,
    epochs: int = 2000,
    learning_rate: float = 0.05,
    lambda_l1: float = 0.15,
    lambda_ortho: float = 1.0,
    lambda_center: float = 0.1,  # НОВЫЙ ПАРАМЕТР: вес лосса центрирования
    device: Optional[torch.device] = None,
    callback: Optional[Callable[[SpecFunc, TimeFunc, SpecFunc, SpecFunc], None]] = None,
    callback_interval_percent: Optional[float] = 1.0
) -> Tuple[SpecFunc, TimeFunc, SpecFunc, SpecFunc]:
    
    # --- 1. Подготовка данных ---
    S_tensor, S_unit = spec.values
    freq_axis, _ = spec.freq
    time_axis, _ = spec.time
    
    if device is None:
        device = S_tensor.device
        
    dtype = S_tensor.dtype
    target_canvas = S_tensor.unsqueeze(0) if S_tensor.dim() == 2 else S_tensor
    target_canvas = target_canvas.to(device)
    
    dt_val, _ = spec.dt
    sr = 1.0 / dt_val
    num_freqs, seq_len = target_canvas.shape[1], target_canvas.shape[2]
    win_width = max(3, int(round((window_duration_ms / 1000.0) * sr)))
    
    model = _FFTSpecConvDecomposer(num_learn, num_freqs, win_width, seq_len, dtype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    identity = torch.eye(num_learn, dtype=dtype, device=device)

    # Предрасчет для лосса центрирования
    target_center = (win_width - 1) / 2.0
    time_indices = torch.arange(win_width, dtype=dtype, device=device)

    callback_epoch_interval = 0
    if callback and callback_interval_percent is not None and callback_interval_percent > 0 and epochs > 0:
        callback_epoch_interval = max(1, int(epochs * (callback_interval_percent / 100.0)))

    print(f"Запуск разложения: {num_patterns} паттернов (через {num_learn}). Окно: {win_width} сэмплов.")
        
    def _post_process_and_invoke_callback(
        current_model: _FFTSpecConvDecomposer, 
        reconstructed_canvas: torch.Tensor, 
        activations: torch.Tensor
    ):
        current_model.eval()
        with torch.no_grad():
            seq_np = activations[0].cpu().numpy()
            
            # --- Сжатие через NMF ---
            nmf = NMF(n_components=num_patterns, init='nndsvda', max_iter=500)
            nmf_projections = nmf.fit_transform(seq_np.T)
            nmf_timelines = torch.tensor(nmf_projections.T, dtype=dtype, device='cpu')

            nmf_weights = nmf.components_ 
            flat_bases_np = current_model.basis.view(num_learn, -1).cpu().numpy()
            nmf_bases_flat = np.dot(nmf_weights, flat_bases_np)
            
            nmf_bases = torch.tensor(
                nmf_bases_flat.reshape(num_patterns, num_freqs, win_width),
                dtype=dtype, device='cpu'
            )

            # --- Покомпонентная реконструкция (БЕЗ ручного сдвига, модель отцентровала всё сама) ---
            fft_size = seq_len + win_width - 1
            time_f = torch.fft.rfft(nmf_timelines, n=fft_size)  # (num_patterns, freq_bins)
            basis_f = torch.fft.rfft(nmf_bases, n=fft_size)     # (num_patterns, num_freqs, freq_bins)
            
            # Свертка (Умножение в частотной области: Таймлайн * Базис)
            canvas_components_f = torch.einsum('n f, n h f -> n h f', time_f, basis_f)
            
            # Возвращаемся во временную область, обрезаем хвост свертки
            components_canvas = torch.fft.irfft(canvas_components_f, n=fft_size)[..., :seq_len]
            
            # Общая реконструкция — сумма всех компонентов
            total_reconstructed = components_canvas.sum(dim=0) 
            
            # --- Упаковка в классы ---
            cpu_freq_axis, cpu_time_axis = freq_axis.cpu(), time_axis.cpu()
            basis_time_axis = torch.arange(win_width, dtype=dtype, device='cpu') * dt_val
            
            timelines_func = TimeFunc((nmf_timelines, UREG.dimensionless), cpu_time_axis)
            bases_spec = SpecFunc((nmf_bases, S_unit), cpu_freq_axis, basis_time_axis)
            reconstructed_spec = SpecFunc((total_reconstructed, S_unit), cpu_freq_axis, cpu_time_axis)
            
            # Слой компонентов
            components_spec = SpecFunc((components_canvas, S_unit), cpu_freq_axis, cpu_time_axis)
            
            if callback:
                callback(bases_spec, timelines_func, reconstructed_spec, components_spec)

        current_model.train()
        return bases_spec, timelines_func, reconstructed_spec, components_spec
            
    # --- 3. Цикл обучения ---
    pbar = tqdm(range(epochs), desc="Оптимизация базисов")
    latest_results = None

    for epoch in pbar:
        optimizer.zero_grad()
        
        with torch.no_grad():
            model.basis.data.clamp_(min=0.0) # Запрет на отрицательные значения
            norms = torch.norm(model.basis, p=2, dim=(1, 2), keepdim=True)
            model.basis.data.div_(norms + 1e-8)

        pred_canvas, active_seq = model.forward()
        
        # 1. Лосс реконструкции
        loss_recon = F.mse_loss(pred_canvas, target_canvas)
        
        # 2. Лосс спарсовости (редкости) таймлайнов
        seq_l2_norms = torch.sqrt(torch.sum(active_seq**2, dim=1) + 1e-8)
        loss_sparse = lambda_l1 * torch.mean(seq_l2_norms)
        
        # 3. Лосс ортогональности базисов (чтобы они не были похожи друг на друга)
        flat_basis = model.basis.view(num_learn, -1)
        gram_matrix = torch.mm(flat_basis, flat_basis.T)
        loss_ortho = lambda_ortho * F.mse_loss(gram_matrix, identity)

        # 4. НОВЫЙ ЛОСС: Центр масс (CoM) каждого базиса должен быть в центре окна
        # Считаем плотность энергии по времени: (num_learn, win_width)
        density = model.basis.sum(dim=1) 
        mass = density.sum(dim=1) + 1e-8
        # Считаем центр масс: sum(time_idx * density) / mass
        com = (density * time_indices).sum(dim=1) / mass
        # Штрафуем отклонение от target_center
        loss_center = lambda_center * F.mse_loss(com, torch.full_like(com, target_center))
        
        loss = loss_recon + loss_sparse + loss_ortho + loss_center
        loss.backward()
        optimizer.step()

        pbar.set_postfix(Loss=f"{loss.item():.3f}", Rec=f"{loss_recon.item():.3f}", Cntr=f"{loss_center.item():.3f}")

        is_last_epoch = (epoch == epochs - 1)
        if callback_epoch_interval > 0 and ((epoch + 1) % callback_epoch_interval == 0 or is_last_epoch):
            latest_results = _post_process_and_invoke_callback(model, pred_canvas, active_seq)

    if latest_results is None:
        latest_results = _post_process_and_invoke_callback(model, *model.forward())

    return latest_results


# =====================================================================
# НОВЫЙ БЛОК: Поиск самого характерного паттерна через ZNCC и градиентный спуск
# =====================================================================

def _functional_match2d_fft(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет 2D кросс-корреляцию через FFT. Ядро НЕ должно быть отзеркалено заранее.
    Функциональная версия для дифференцируемых вычислений.
    """
    flipped_kernel = torch.flip(kernel, dims=(-2, -1))
    
    F_sig, T_sig = signal.shape[-2], signal.shape[-1]
    F_ker, T_ker = flipped_kernel.shape[-2], flipped_kernel.shape[-1]
    
    N_f = F_sig + F_ker - 1
    N_t = T_sig + T_ker - 1

    Sig_f = torch.fft.rfft2(signal, s=(N_f, N_t), dim=(-2, -1))
    Ker_f = torch.fft.rfft2(flipped_kernel, s=(N_f, N_t), dim=(-2, -1))
    
    out_f = Sig_f * Ker_f
    out = torch.fft.irfft2(out_f, s=(N_f, N_t), dim=(-2, -1))

    pad_f_start = (F_ker - 1) // 2
    pad_t_start = (T_ker - 1) // 2
    
    return out[..., pad_f_start : pad_f_start + F_sig, pad_t_start : pad_t_start + T_sig]

def _total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Заставляет паттерн быть гладким, штрафуя резкие перепады."""
    tv_h = torch.sum(torch.abs(img[..., :-1, :] - img[..., 1:, :]))
    tv_w = torch.sum(torch.abs(img[..., :, :-1] - img[..., :, 1:]))
    return tv_h + tv_w

def discover_best_pattern_with_callback(
    spec: SpecFunc,
    window_duration_ms: float = 100.0,
    anchor_percent: float = 50.0,  ### ИЗМЕНЕНИЕ: Добавляем якорь положения ###
    epochs: int = 500,
    learning_rate: float = 0.05,
    top_k_peaks: int = 20,
    lambda_tv: float = 0.01,
    lambda_center: float = 0.1,  ### ИЗМЕНЕНИЕ: Вес для нового лосса ###
    device: Optional[torch.device] = None,
    callback: Optional[Callable[[SpecFunc, SpecFunc], None]] = None,
    callback_interval_percent: Optional[float] = 2.0
) -> Tuple[SpecFunc, SpecFunc]:
    """
    Находит один "идеальный" паттерн...
    
    Args:
        ...
        anchor_percent (float): Ожидаемое положение центра масс паттерна по времени
                                (в процентах от ширины окна). 50.0 - центр.
        ...
        lambda_center (float): Вес штрафа за отклонение от якоря.
        ...
    """
    # --- 1. Подготовка данных ---
    S, S_unit = spec.values
    freq_axis, _ = spec.freq
    time_axis, _ = spec.time
    
    if device is None: device = S.device
    dtype = S.dtype
    target_spec = S.unsqueeze(0) if S.dim() == 2 else S
    target_spec = target_spec.to(device, dtype)
    
    dt_val, _ = spec.dt
    sr = 1.0 / dt_val
    num_freqs, _ = target_spec.shape[-2], target_spec.shape[-1]
    win_width = max(3, int(round((window_duration_ms / 1000.0) * sr)))
    
    # --- 2. Инициализация ---
    W_raw = nn.Parameter(torch.randn(1, num_freqs, win_width, device=device, dtype=dtype))
    optimizer = torch.optim.Adam([W_raw], lr=learning_rate)
    
    ### ИЗМЕНЕНИЕ: Предрасчет для лосса центрирования ###
    target_center_index = (win_width - 1) * (anchor_percent / 100.0)
    time_indices = torch.arange(win_width, device=device, dtype=dtype)
    
    print(f"Поиск паттерна: окно {win_width} сэмплов. Целевой центр: {target_center_index:.1f}")
    
    callback_epoch_interval = 0
    if callback and callback_interval_percent is not None and callback_interval_percent > 0:
        callback_epoch_interval = max(1, int(epochs * (callback_interval_percent / 100.0)))

    def _invoke_callback(current_W_raw: torch.Tensor, zncc_map_tensor: torch.Tensor):
        with torch.no_grad():
            W_norm = current_W_raw - current_W_raw.mean()
            W_norm /= (torch.norm(W_norm) + 1e-8)
            
            # Упаковка в SpecFunc
            pattern_time_axis = torch.arange(win_width, device='cpu', dtype=dtype) * dt_val
            
            found_pattern_spec = SpecFunc(
                (W_norm.squeeze(0).cpu(), S_unit), 
                freq_axis.cpu(), 
                pattern_time_axis
            )
            
            correlation_map_spec = SpecFunc(
                (zncc_map_tensor.squeeze(0).cpu(), UREG.dimensionless),
                freq_axis.cpu(),
                time_axis.cpu()
            )
            
            if callback:
                callback(found_pattern_spec, correlation_map_spec)
        return found_pattern_spec, correlation_map_spec

    # --- 3. Цикл обучения ---

    pbar = tqdm(range(epochs), desc="Поиск паттерна")
    latest_results = None
    eps = 1e-8

    for epoch in pbar:
        optimizer.zero_grad()
        
        # --- ШАГ A: Нормализация паттерна (без изменений) ---
        W_zero_mean = W_raw - W_raw.mean()
        W = W_zero_mean / (torch.norm(W_zero_mean) + eps)
        
        # --- ШАГ B: ZNCC (без изменений) ---
        # ... (код для расчета zncc_map) ...
        numerator = _functional_match2d_fft(target_spec, W)
        U = torch.ones_like(W); N = W.numel()
        Sum_S = _functional_match2d_fft(target_spec, U)
        Sum_S2 = _functional_match2d_fft(target_spec**2, U)
        local_variance = (Sum_S2 - (Sum_S**2) / N).clamp_(min=0.0)
        denominator = torch.sqrt(local_variance) + eps
        zncc_map = numerator / denominator

        # --- ШАГ C: Вычисление Loss-функции ---
        # 1. Лосс "заметности" (без изменений)
        top_peaks, _ = torch.topk(zncc_map.flatten(), k=top_k_peaks)
        loss_match = -torch.mean(top_peaks) 
        
        # 2. Лосс гладкости (без изменений)
        loss_tv = lambda_tv * _total_variation_loss(W_raw)
        
        ### ИЗМЕНЕНИЕ: Добавляем лосс центрирования ###
        # 3. Лосс центра масс (Center of Mass Loss)
        # Мы хотим, чтобы энергия в W_raw была сконцентрирована вокруг target_center_index.
        # Используем квадрат W_raw, чтобы работать с "энергией" и сделать ее положительной.
        energy_density = W_raw**2
        
        # Плотность энергии по времени: (1, win_width)
        time_density = energy_density.sum(dim=-2)
        total_mass = time_density.sum() + eps
        
        # Считаем центр масс: sum(time_idx * density) / mass
        current_com = (time_density * time_indices).sum() / total_mass
        
        # Штрафуем отклонение от целевого центра
        loss_center = lambda_center * F.mse_loss(
            current_com, 
            torch.full_like(current_com, target_center_index)
        )
        
        total_loss = loss_match + loss_tv + loss_center
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(Loss=f"{total_loss.item():.3f}", Match=f"{-loss_match.item():.3f}", Cntr=f"{loss_center.item():.3f}")


        # Вызов callback для визуализации
        is_last_epoch = (epoch == epochs - 1)
        if callback_epoch_interval > 0 and ((epoch + 1) % callback_epoch_interval == 0 or is_last_epoch):
            latest_results = _invoke_callback(W_raw, zncc_map)

    # Гарантированный вызов в конце, если не было в цикле
    if latest_results is None:
        # Пересчитаем карту с финальными весами
        with torch.no_grad():
            W_zero_mean = W_raw - W_raw.mean()
            W = W_zero_mean / (torch.norm(W_zero_mean) + eps)
            numerator = _functional_match2d_fft(target_spec, W)
            U = torch.ones_like(W); N = W.numel()
            Sum_S = _functional_match2d_fft(target_spec, U)
            Sum_S2 = _functional_match2d_fft(target_spec**2, U)
            local_variance = (Sum_S2 - (Sum_S**2) / N).clamp_(min=0.0)
            denominator = torch.sqrt(local_variance) + eps
            final_zncc_map = numerator / denominator
            latest_results = _invoke_callback(W_raw, final_zncc_map)

    return latest_results

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
if __name__ == "__main__":
    # --- Сначала протестируем старый код, чтобы убедиться, что ничего не сломалось ---
    try:
        F_BINS, TIME_STEPS, TEMPLATES_COUNT, TEMPLATES_TIME = 64, 1000, 4, 21
        ANYBATCH = (10, )
        H = nn.Parameter(torch.rand(*ANYBATCH, TEMPLATES_COUNT, TIME_STEPS))
        templates_data = torch.rand(*ANYBATCH, TEMPLATES_COUNT, F_BINS, TEMPLATES_TIME)
        # и т.д. ... (ваш старый __main__ блок)
        print("Старый код __main__ работает.")
    except Exception as e:
        print(f"Ошибка в старом коде: {e}")
        
    # --- Теперь новый тест для discover_best_pattern ---
    F_BINS, TIME_STEPS = 64, 2000
    spec_tensor = torch.randn(F_BINS, TIME_STEPS) * 0.1 # Фоновый шум

    # Создадим "идеальный" паттерн (например, птичий чирк - синусоида)
    pattern_width = 100
    y_coords = torch.arange(F_BINS).unsqueeze(1)
    x_coords = torch.arange(pattern_width).unsqueeze(0)
    center_freq = F_BINS / 2
    chirp = torch.exp(-((y_coords - (center_freq + 15 * torch.sin(x_coords * 0.2)))**2) / 10)
    
    # Вставим этот паттерн в три места на спектрограмме
    spec_tensor[:, 200:200+pattern_width] += chirp * 5.0
    spec_tensor[:, 900:900+pattern_width] += chirp * 3.0
    spec_tensor[:, 1500:1500+pattern_width] += chirp * 8.0

    # Упакуем в ваши классы
    freq_axis = torch.arange(F_BINS)
    time_axis = torch.arange(TIME_STEPS) * 0.01 # dt = 10ms
    
    test_spec = SpecFunc((spec_tensor, UREG.dimensionless), freq_axis, time_axis)

    # Определим callback для визуализации
    def plot_callback(found_pattern, correlation_map):
        try:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            
            pattern_img, _ = found_pattern.values
            axs[0].imshow(pattern_img.numpy(), aspect='auto', origin='lower', cmap='viridis')
            axs[0].set_title('Найденный паттерн (W)')
            
            corr_img, _ = correlation_map.values
            axs[1].imshow(corr_img.numpy(), aspect='auto', origin='lower', cmap='coolwarm', vmin=-1, vmax=1)
            axs[1].set_title('Карта корреляций ZNCC')
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib не установлен, визуализация пропущена.")

    print("\n--- Запуск поиска самого характерного паттерна ---")
    found_pattern, final_corr_map = discover_best_pattern_with_callback(
        spec=test_spec,
        window_duration_ms=1000.0, # 100 сэмплов при dt=10ms
        epochs=100,
        learning_rate=0.05,
        top_k_peaks=10,
        lambda_tv=0.001,
        callback=plot_callback,
        callback_interval_percent=25.0
    )

    print("\nПоиск завершен!")
    print("Форма найденного паттерна:", found_pattern.values[0].shape)
    print("Форма итоговой карты корреляций:", final_corr_map.values[0].shape)
``````

