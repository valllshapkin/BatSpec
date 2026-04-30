from pathlib import Path
import numpy as np
from scipy.signal import convolve, correlate

ScriptDir = Path(__file__).parent

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW, TEST_BLHA_WINODW
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.Spectral import makeLogDB, DSPContext, makeRobustSpec
from BatSpec.Core.Spectral.Statistic import noiseZNormByFreq
from BatSpec.Core.Functions import TimeFunc
from BatSpec.Core.Physical.Units import UREG

from BatSpec.Visualize import update_spec2d, run_visualizer, update_function

# --- Импорты MultiArray ---
import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType


def discover_patterns_convnmf_classic(energy_func: TimeFunc, num_patterns: int = 2, pattern_time_ms: float = 10.0, num_iter: int = 50, sparsity: float = 0.5):
    """
    Классический алгоритм Convolutional NMF с использованием строгих 
    мультипликативных правил обновления (Smaragdis, 2004).
    Гарантированная математическая сходимость, строгая неотрицательность.
    """
    ctx_orig = energy_func.context
    
    # Алгоритм требует точной математики массивов, используем NumPy/SciPy
    V_np = ma.to_numpy(energy_func.values[0])
    if V_np.ndim > 1:
        V_np = V_np[0] # Берем первый батч/канал
        
    # Предотвращаем деление на ноль и отрицательные значения в исходных данных
    V = np.maximum(V_np, 1e-9)
    T = len(V)
    
    dt_val, dt_unit = energy_func.dt
    pattern_bins = int((pattern_time_ms / 1000.0) / float(dt_val))
    pattern_bins = max(3, pattern_bins)
    
    H_len = T - pattern_bins + 1
    if H_len <= 0:
        print("Сигнал слишком короткий для паттерна такой длины.")
        return

    # 1. Инициализация W (Паттерны) и H (Активации) случайными неотрицательными числами
    np.random.seed(42) # Фиксируем seed для воспроизводимости
    W = np.random.rand(num_patterns, pattern_bins) + 0.1
    H = np.random.rand(num_patterns, H_len) + 0.1
    
    print(f"Классический ConvNMF: {num_patterns} паттернов по {pattern_bins} бинов. Выполняем {num_iter} итераций...")

    # 2. Мультипликативные обновления (Lee & Seung -> Smaragdis)
    for iteration in range(num_iter):
        
        # Шаг A: Реконструкция V_hat = sum(W_k * H_k)
        V_hat = np.zeros(T)
        for k in range(num_patterns):
            # linear convolution: W * H
            V_hat += convolve(H[k], W[k], mode='full')
        V_hat = np.maximum(V_hat, 1e-12)
        
        # Шаг B: Обновление H (Активации)
        for k in range(num_patterns):
            # Числитель: кросс-корреляция Сигнала и Паттерна
            num = correlate(V, W[k], mode='valid')
            # Знаменатель: кросс-корреляция Реконструкции и Паттерна + штраф за разреженность (L1)
            den = correlate(V_hat, W[k], mode='valid') + sparsity
            
            # Мультипликативное обновление
            H[k] *= (num / np.maximum(den, 1e-12))
            
        # Обновляем V_hat после изменения H (повышает стабильность)
        V_hat = np.zeros(T)
        for k in range(num_patterns):
            V_hat += convolve(H[k], W[k], mode='full')
        V_hat = np.maximum(V_hat, 1e-12)

        # Шаг C: Обновление W (Паттерны)
        for k in range(num_patterns):
            # Числитель: кросс-корреляция Сигнала и Активаций
            num = correlate(V, H[k], mode='valid')
            # Знаменатель: кросс-корреляция Реконструкции и Активаций
            den = correlate(V_hat, H[k], mode='valid')
            
            # Мультипликативное обновление
            W[k] *= (num / np.maximum(den, 1e-12))
            
            # Решение проблемы неоднозначности масштаба (Scale Ambiguity)
            # Приравниваем норму паттерна к 1, а амплитуду переносим на H
            norm_factor = np.sum(W[k]) + 1e-12
            W[k] /= norm_factor
            H[k] *= norm_factor

    print("Классический ConvNMF завершен. Выгрузка в UI...")

    # 3. Распаковка в объекты системы BatSpec
    time_axis = energy_func.time[0]
    axis_ctx = ArrayContext(ctx_orig._framework, ctx_orig._device, None)
    
    # Реконструкция V_hat
    recon_func = TimeFunc(
        values=(ma.convert_to(V_hat, ctx_orig), energy_func.values[1]),
        axis=time_axis
    )
    update_function("Classic_ConvNMF_Reconstruction", recon_func)
    
    # Паттерны и Активации
    for k in range(num_patterns):
        # Паттерн W
        w_axis_np = np.arange(pattern_bins) * float(dt_val)
        w_func = TimeFunc(
            values=(ma.convert_to(W[k], ctx_orig), energy_func.values[1]),
            axis=ma.convert_to(w_axis_np, axis_ctx)
        )
        update_function(f"Classic_Pattern_{k}", w_func)
        
        # Активация H (добиваем нулями, чтобы график совпал с осью X оригинала)
        h_full = np.zeros(T)
        h_full[:H_len] = H[k]
        h_func = TimeFunc(
            values=(ma.convert_to(h_full, ctx_orig), UREG.dimensionless),
            axis=time_axis
        )
        update_function(f"Classic_Activation_{k}", h_func)


@run_visualizer
def main():
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    update_function("record", record)

    ctx_numpy_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    
    print("--- Основные вычисления ---")
    with DSPContext(ctx_numpy_cpu):
        spec_robust = makeRobustSpec(
            record,
            window=(TEST_HANN_WINODW, TEST_BLHA_WINODW),
            overlap=0.8,
            bins=300,
            shifts=(-1, 1)  
        ).to_context(ctx_numpy_cpu)
        
    update_spec2d("Robust", makeLogDB(spec_robust, add_one=False))

    zspec = noiseZNormByFreq(spec_robust)
    update_spec2d("zspec", makeLogDB(zspec, add_one=True))

    print("--- Интегрирование и фильтрация шума ---")
    energy_time = zspec.integrateOverFreq()
    
    val_a, val_u = energy_time.values
    val_clipped = ma.clamp_min(val_a, 0.0)
    
    energy_time.values = (val_clipped, val_u)
    update_function("Energy_Clipped", energy_time)

    print("--- Поиск паттернов ---")
    # Используем строгий алгоритм: 2 паттерна, ширина паттерна ~8 мс, 100 итераций.
    # Параметр sparsity определяет насколько "острыми" будут всплески активации.
    discover_patterns_convnmf_classic(energy_time, num_patterns=4, pattern_time_ms=100, num_iter=500, sparsity=1e+6)