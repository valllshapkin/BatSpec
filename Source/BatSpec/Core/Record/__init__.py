from pathlib import Path
from typing import Optional
import numpy as np

from BatSpec.Core.Physical.Units import UREG, PintUnit
from BatSpec.Core.Functions import TimeFunc

def loadRecord(path: Path, unit: PintUnit = UREG.FS) -> 'TimeFunc':
    """
    Загружает аудиофайл как TimeFunc (mono, float64).
    По умолчанию возвращает объект на базе NumPy массивов.
    """
    import soundfile as sf  # type: ignore

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    # Читаем файл (numpy)
    data, samplerate = sf.read(str(path), dtype='float64', always_2d=True)

    # Приводим к mono
    if data.ndim == 2:
        if data.shape[1] > 1:
            data = np.mean(data, axis=1)        # stereo -> mono
        else:
            data = data.flatten()               # (N, 1) -> (N,)

    # Создаем массивы времени
    n_samples = len(data)
    time_arr = np.arange(n_samples, dtype=np.float64) / samplerate

    return TimeFunc(
        values=(data, unit),
        axis=time_arr
    )


def trimRecord(signal: 'TimeFunc', 
               t_start: float = 0.0, 
               t_end: Optional[float] = None) -> 'TimeFunc':
    """
    Обрезает запись по времени (в секундах). Поддерживает батчи.
    Работает с любым фреймворком (NumPy, PyTorch, TF).
    """
    if t_start < 0:
        raise ValueError("t_start не может быть отрицательным")

    time_array, time_unit = signal.time
    value_array, value_unit = signal.values

    if t_end is None:
        t_end = float(time_array[-1])

    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) должен быть больше t_start ({t_start})")

    # Создаем маску. Операторы >=, <= и & работают одинаково во всех фреймворках
    mask = (time_array >= t_start) & (time_array <= t_end)
    
    # Метод .any() поддерживается и в numpy, и в torch
    if not mask.any():
        raise ValueError(f"Интервал [{t_start}, {t_end}] не пересекается с сигналом")

    # Обрезаем массивы (слайсинг работает везде одинаково)
    new_time = time_array[mask]
    new_values = value_array[..., mask]

    return TimeFunc(
        values=(new_values, value_unit),
        axis=new_time
    )


def resampleRecord(signal: 'TimeFunc', new_sr: int) -> 'TimeFunc':
    """
    Изменяет частоту дискретизации сигнала (resampling).
    Автоматически сохраняет фреймворк и устройство исходного сигнала.
    """
    from scipy.signal import resample_poly  # type: ignore
    from math import gcd

    if new_sr <= 0:
        raise ValueError("new_sr должен быть положительным целым числом")

    orig_sr = signal.sr
    if orig_sr == new_sr:
        return signal

    # 1. Запоминаем исходный фреймворк и девайс
    orig_tensor = signal.values[0]
    type_str = str(type(orig_tensor)).lower()
    
    if 'torch' in type_str: fw_target = 'torch'
    elif 'tensorflow' in type_str: fw_target = 'tensorflow'
    else: fw_target = 'numpy'
    
    device_target = getattr(orig_tensor, 'device', None)

    # 2. Переводим сигнал в NumPy, так как scipy работает только с ним
    sig_np = signal.to_framework('numpy')
    v_np, V_unit = sig_np.values
    t_np, T_unit = sig_np.time

    g = gcd(orig_sr, new_sr)
    up = new_sr // g
    down = orig_sr // g

    # 3. Выполняем ресэмплинг (NumPy)
    new_v_np = resample_poly(v_np, up=up, down=down, axis=-1)

    new_length = new_v_np.shape[-1]
    new_dt = 1.0 / new_sr
    new_t_np = np.arange(new_length, dtype=t_np.dtype) * new_dt

    # 4. Собираем временный NumPy-объект
    sig_resampled = TimeFunc(
        values=(new_v_np, V_unit),
        axis=new_t_np
    )

    # 5. Возвращаем в исходный фреймворк и устройство!
    return sig_resampled.to_framework(fw_target, device=device_target)


def correctDC(f: TimeFunc) -> TimeFunc:
    """
    Удаляет смещение постоянного тока (DC offset), вычитая медиану.
    Фреймворк-агностично.
    """
    v, V = f.values
    t, _ = f.time
    
    type_str = str(type(v)).lower()

    # Считаем медиану в зависимости от фреймворка
    if 'torch' in type_str:
        import torch
        # В PyTorch median возвращает namedtuple (values, indices)
        med = torch.median(v, dim=-1, keepdim=True).values
    elif 'tensorflow' in type_str:
        # У TF нет простой функции median для тензоров без TFP, 
        # поэтому безопасно перекинем в numpy и обратно
        import tensorflow as tf
        med_np = np.median(v.numpy(), axis=-1, keepdims=True)
        med = tf.convert_to_tensor(med_np, dtype=v.dtype)
    else:
        # Для NumPy и всех остальных
        med = np.median(v, axis=-1, keepdims=True)
        
    corrected_values = v - med
    
    return TimeFunc(
        values=(corrected_values, V),
        axis=t # Передаем ось времени как есть
    )