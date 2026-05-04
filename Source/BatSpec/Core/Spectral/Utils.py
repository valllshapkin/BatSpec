import numpy as np
from typing import Optional
from BatSpec.Core.Functions import SpecFunc, TimeFunc
import MultiArray as ma

def padSpecFreq(spec: SpecFunc, new_f_start: float, new_f_end: float) -> SpecFunc:
    """Расширяет спектрограмму по частотной оси до новых границ, дополняя нулями."""
    ctx = spec.context
    mat_a = spec._matrx_a
    unit = spec._matrx_u
    t_arr = spec._first_a
    f_arr = spec._sec___a

    df_unit, df_val = spec.df
    
    orig_f_start = float(f_arr[0])
    orig_f_end = float(f_arr[-1])
    
    if new_f_start > orig_f_start or new_f_end < orig_f_end:
        raise ValueError(f"Новый диапазон [{new_f_start}, {new_f_end}] не охватывает [{orig_f_start}, {orig_f_end}]")

    num_new_f_bins = int(round((new_f_end - new_f_start) / df_val)) + 1
    new_f_arr = ma.linspace(new_f_start, new_f_end, num_new_f_bins, ctx)
    
    num_t_bins = mat_a.shape[0]
    new_mat = ma.zeros((num_t_bins, num_new_f_bins), ctx)

    start_idx = int(round((orig_f_start - new_f_start) / df_val))
    end_idx = start_idx + len(f_arr)
    
    # Копируем оригинальные данные в центр новой матрицы
    # В MultiArray прямое присваивание срезам пока лучше делать через NumPy, если нет встроенной поддержки 
    new_mat_np = ma.to_numpy(new_mat)
    new_mat_np[:, start_idx:end_idx] = ma.to_numpy(mat_a)
    new_mat = ma.convert_to(new_mat_np, ctx)

    return SpecFunc(new_mat, t_arr, new_f_arr, unit)

def stackSpecsMax(*specs: SpecFunc) -> SpecFunc:
    """Создает спектрограмму, где каждый пиксель - это максимум из всех переданных спектрограмм."""
    if not specs:
        raise ValueError("Нужно передать хотя бы одну спектрограмму.")
    if len(specs) == 1:
        return specs[0]

    ref_spec = specs[0]
    ctx = ref_spec.context
    stacked_np = np.stack([ma.to_numpy(s._matrx_a) for s in specs], axis=0)
    max_mat_np = np.max(stacked_np, axis=0)
    
    max_mat = ma.convert_to(max_mat_np, ctx)
    return SpecFunc(max_mat, ref_spec._first_a, ref_spec._sec___a, ref_spec._matrx_u)

def cropSpecRect(spec: SpecFunc, t_start: float, t_end: float, f_start: float, f_end: float) -> SpecFunc:
    """Вырезает прямоугольник из спектрограммы по физическим координатам."""
    ctx = spec.context
    t_arr, f_arr = spec._first_a, spec._sec___a
    _, dt = spec.dt
    _, df = spec.df

    # Индексы относительно оригинала
    idx_t_start = int(round((t_start - float(t_arr[0])) / dt))
    idx_t_end   = int(round((t_end - float(t_arr[0])) / dt))
    idx_f_start = int(round((f_start - float(f_arr[0])) / df))
    idx_f_end   = int(round((f_end - float(f_arr[0])) / df))

    # Вырезаем через NumPy для безопасности границ
    mat_np = ma.to_numpy(spec._matrx_a)
    N_t, N_f = max(0, idx_t_end - idx_t_start), max(0, idx_f_end - idx_f_start)
    
    pad_val = np.min(mat_np) if not np.iscomplexobj(mat_np) and np.min(mat_np) < 0 else 0
    new_mat_np = np.full((N_t, N_f), fill_value=pad_val, dtype=mat_np.dtype)

    orig_t_start = max(0, idx_t_start)
    orig_t_end   = min(len(t_arr), idx_t_end)
    orig_f_start = max(0, idx_f_start)
    orig_f_end   = min(len(f_arr), idx_f_end)

    new_t_start = max(0, -idx_t_start)
    new_t_end   = new_t_start + (orig_t_end - orig_t_start)
    new_f_start = max(0, -idx_f_start)
    new_f_end   = new_f_start + (orig_f_end - orig_f_start)

    if (orig_t_start < orig_t_end) and (orig_f_start < orig_f_end):
        new_mat_np[new_t_start:new_t_end, new_f_start:new_f_end] = mat_np[orig_t_start:orig_t_end, orig_f_start:orig_f_end]

    new_t_arr = float(t_arr[0]) + np.arange(idx_t_start, idx_t_end) * dt
    new_f_arr = float(f_arr[0]) + np.arange(idx_f_start, idx_f_end) * df

    return SpecFunc(
        matrix=ma.convert_to(new_mat_np, ctx),
        time=ma.convert_to(new_t_arr, ctx),
        freq=ma.convert_to(new_f_arr, ctx),
        unit=spec._matrx_u
    )

def extractPeakContext(spec: SpecFunc, duration_s: float = 0.100) -> SpecFunc:
    """Находит пик энергии и вырезает окно вокруг него."""
    time_func: TimeFunc = spec.integrateOverFreq()
    val_np = ma.to_numpy(time_func._value_a)
    t_np = ma.to_numpy(time_func._axis__a)
    
    peak_time = float(t_np[np.argmax(val_np)])
    return cropSpecRect(
        spec, 
        t_start=peak_time - duration_s / 2.0, 
        t_end=peak_time + duration_s / 2.0, 
        f_start=float(spec._sec___a[0]), 
        f_end=float(spec._sec___a[-1])
    )
