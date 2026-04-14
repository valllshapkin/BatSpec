from Physics.Context import fw
from Physics.Function import TimeFunc
from Physics.Units import ureg, PintUnit

from pathlib import Path
import soundfile as sf
from typing import Any

type T = Any

def loadRecord(path: Path, unit: PintUnit = ureg.Pa) -> 'TimeFunc':
    import numpy as np
    """Load audio file as TimeFunc (mono, float64)."""
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    data, samplerate = sf.read(str(path), dtype='float64', always_2d=True)

    # Force mono
    if data.ndim == 2:
        if data.shape[1] > 1:
            data = np.mean(data, axis=1)        # stereo -> mono
        else:
            data = data.flatten()               # (N, 1) -> (N,)

    # Time axis in seconds
    n_samples = len(data)
    time_axis = np.arange(n_samples) / samplerate

    return TimeFunc(
        values=data,
        axis=time_axis,
        unit_values=unit,
    )

def trimRecord(signal: TimeFunc, t_start: float = 0.0, t_end: float | None = None) -> TimeFunc:
    """
    Обрезает запись по времени (в секундах).

    Параметры
    ----------
    signal : TimeFunc
        Исходный сигнал.
    t_start : float, optional
        Начальное время обрезки в секундах (по умолчанию 0.0).
    t_end : float | None, optional
        Конечное время обрезки в секундах. 
        Если None — обрезается до конца сигнала.

    Возвращает
    ----------
    TimeFunc
        Новый обрезанный сигнал.
    """
    if t_start < 0:
        raise ValueError("t_start не может быть отрицательным")

    _, time_array = signal.time
    value_unit, value_array = signal.values

    # Если t_end не указан — берём конец сигнала
    if t_end is None:
        t_end = float(time_array[-1])

    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) должен быть больше t_start ({t_start})")

    # Находим индексы для обрезки
    mask = (time_array >= t_start) & (time_array <= t_end)
    
    if not fw().any(mask):
        raise ValueError(f"Интервал [{t_start}, {t_end}] не пересекается с сигналом")

    # Обрезаем массивы
    new_time = time_array[mask]
    new_values = value_array[mask]

    # Создаём новый объект TimeFunc
    return TimeFunc(
        values=new_values,
        axis=new_time,
        unit_values=value_unit
    )