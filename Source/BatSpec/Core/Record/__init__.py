from pathlib import Path
from typing import Optional
import numpy as np
from math import gcd

# --- 1. Импорты специфичных для проекта библиотек ---
from BatSpec.Core.Physical.Units import UREG, PintUnit
from BatSpec.Core.Functions import TimeFunc

# --- 2. Внедряем наш новый универсальный фреймворк ---
import MultiArray as ma


def loadRecord(path: Path, unit: PintUnit = UREG.FS) -> 'TimeFunc':
    """
    Загружает аудиофайл как TimeFunc (mono, float64).
    По умолчанию возвращает объект на базе NumPy массивов, так как soundfile работает с NumPy.
    Далее пользователь может перевести объект в любой контекст через .to_context().
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
    Работает с любым фреймворком (NumPy, PyTorch, TF, JAX, CuPy).
    """
    if t_start < 0:
        raise ValueError("t_start не может быть отрицательным")

    time_array, _ = signal.time
    value_array, value_unit = signal.values

    if t_end is None:
        t_end = float(time_array[-1])

    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) должен быть больше t_start ({t_start})")

    # Создаем маску. Операторы >=, <= и & работают одинаково во всех фреймворках
    mask = (time_array >= t_start) & (time_array <= t_end)
    
    # Проверяем, есть ли хоть один элемент в маске (фреймворк-агностично)
    ctx = ma.ArrayContext.from_array(mask)
    if ctx.isTensorflow():
        any_result = ctx.fw.reduce_any(mask)
    else: # .any() работает для numpy, torch, jax, cupy
        any_result = mask.any()
        
    if not any_result:
        raise ValueError(f"Интервал [{t_start}, {t_end}] не пересекается с сигналом")

    # Обрезаем массивы (булево индексирование работает везде одинаково)
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

    if new_sr <= 0:
        raise ValueError("new_sr должен быть положительным целым числом")

    orig_sr = signal.sr
    if orig_sr == new_sr:
        return signal

    # 1. Запоминаем исходный контекст (фреймворк, девайс, dtype)
    original_ctx = signal.context
    
    # 2. Создаем контекст NumPy для работы с SciPy
    numpy_ctx = ma.ArrayContext(ma.Framework.NUMPY, ma.DeviceType.CPU, None)
    
    # 3. Переводим сигнал в NumPy с помощью to_context
    signal_np = signal.to_context(numpy_ctx)
    v_np, v_unit = signal_np.values
    t_np, _ = signal_np.time

    g = gcd(orig_sr, new_sr)
    up = new_sr // g
    down = orig_sr // g

    # 4. Выполняем ресэмплинг (NumPy)
    new_v_np = resample_poly(v_np, up=up, down=down, axis=-1)
    new_length = new_v_np.shape[-1]
    new_dt = 1.0 / new_sr
    new_t_np = np.arange(new_length, dtype=t_np.dtype) * new_dt

    # 5. Собираем временный NumPy-объект
    resampled_signal_np = TimeFunc(
        values=(new_v_np, v_unit),
        axis=new_t_np
    )

    # 6. Возвращаем в исходный контекст!
    return resampled_signal_np.to_context(original_ctx)


def correctDC(f: TimeFunc) -> TimeFunc:
    """
    Удаляет смещение постоянного тока (DC offset), вычитая медиану.
    Работает с любым фреймворком.
    """
    v, v_unit = f.values
    t, _ = f.time
    
    ctx = f.context
    
    if ctx.isTorch():
        # В PyTorch median возвращает namedtuple (values, indices)
        med = ctx.fw.median(v, dim=-1, keepdim=True).values
    elif ctx.isTensorflow():
        # У TF нет простой функции median, поэтому используем NumPy как надёжный fallback
        v_np = ma.to_numpy(v)
        med_np = np.median(v_np, axis=-1, keepdims=True)
        med = ma.convert_to(med_np, ctx)
    else: # NumPy, Jax, Cupy
        # У этих фреймворков median - это функция модуля, а не метод тензора
        med = ctx.fw.median(v, axis=-1, keepdims=True)
        
    corrected_values = v - med
    
    return TimeFunc(
        values=(corrected_values, v_unit),
        axis=t
    )
