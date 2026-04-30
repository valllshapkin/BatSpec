

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

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/MultiArray/__init__.py" encoding="utf-8"
from .Core import ArrayContext, Framework, DeviceType
from .Base import *
from . import Linalg
from .Linalg import matmul, dot, einsum
from . import FFT

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/MultiArray/Base.py" encoding="utf-8"
from typing import Any, Sequence, Tuple
from .Core import ArrayContext, Framework

# ==========================================
# ВНУТРЕННИЕ ХЕЛПЕРЫ
# ==========================================

def _create_tensor(func_name: str, ctx: 'ArrayContext', *args, **kwargs) -> Any:
    fw_module = ctx.fw
    func = getattr(fw_module, func_name)
    
    if ctx.isTensorflow():
        with fw_module.device(ctx.device):
            return func(*args, dtype=ctx.dtype, **kwargs)
            
    elif ctx.isTorch():
        return func(*args, dtype=ctx.dtype, device=ctx.device, **kwargs)
        
    else:
        return func(*args, dtype=ctx.dtype, **kwargs)

# ==========================================
# 1. ФУНКЦИИ СОЗДАНИЯ (Creation)
# ==========================================

def arange(end: int, ctx: 'ArrayContext') -> Any:
    name = 'range' if ctx.isTensorflow() else 'arange'
    return _create_tensor(name, ctx, end)

def zeros(shape: Tuple[int, ...], ctx: 'ArrayContext') -> Any:
    return _create_tensor('zeros', ctx, shape)

def ones(shape: Tuple[int, ...], ctx: 'ArrayContext') -> Any:
    return _create_tensor('ones', ctx, shape)

def empty(shape: Tuple[int, ...], ctx: 'ArrayContext') -> Any:
    if ctx.isTensorflow() or ctx.isJax():
        return _create_tensor('zeros', ctx, shape)
    return _create_tensor('empty', ctx, shape)

def full(shape: Tuple[int, ...], fill_value: Any, ctx: 'ArrayContext') -> Any:
    if ctx.isTensorflow():
        with ctx.fw.device(ctx.device):
            return ctx.fw.fill(shape, fill_value)
    return _create_tensor('full', ctx, shape, fill_value=fill_value)

def eye(n: int, ctx: 'ArrayContext') -> Any:
    return _create_tensor('eye', ctx, n)

def linspace(start: float, stop: float, num: int, ctx: 'ArrayContext') -> Any:
    if ctx.isTorch():
        return ctx.fw.linspace(start, stop, steps=num, dtype=ctx.dtype, device=ctx.device)
    elif ctx.isTensorflow():
        with ctx.fw.device(ctx.device):
            tensor = ctx.fw.linspace(float(start), float(stop), num)
            return ctx.fw.cast(tensor, ctx.dtype) if ctx.dtype else tensor
    else:
        return ctx.fw.linspace(start, stop, num, dtype=ctx.dtype)

# ==========================================
# 2. УНАРНАЯ МАТЕМАТИКА (Элементные операции)
# ==========================================

def abs(x: Any) -> Any:
    return ArrayContext.from_array(x).fw.abs(x)

def exp(x: Any) -> Any:
    return ArrayContext.from_array(x).fw.exp(x)

def log(x: Any) -> Any:
    return ArrayContext.from_array(x).fw.log(x)

def log10(x: Any) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.log10(x)
    elif ctx.isTensorflow():
        return ctx.fw.math.log(x) / ctx.fw.math.log(ctx.fw.constant(10.0, dtype=x.dtype))
    else:
        return ctx.fw.log10(x)

def clamp_min(x: Any, min_val: float) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.clamp(x, min=min_val)
    elif ctx.isTensorflow():
        return ctx.fw.maximum(x, ctx.fw.constant(min_val, dtype=x.dtype))
    elif ctx.isJax():
        # ИСПРАВЛЕНИЕ WARNING: JAX теперь требует kwarg `min`
        return ctx.fw.clip(x, min=min_val)
    else:
        # Для Numpy/CuPy позиционные аргументы (a, a_min, a_max) работают надежнее
        return ctx.fw.clip(x, min_val, None)

def sin(x: Any) -> Any:
    return ArrayContext.from_array(x).fw.sin(x)

def cos(x: Any) -> Any:
    return ArrayContext.from_array(x).fw.cos(x)

def sqrt(x: Any) -> Any:
    return ArrayContext.from_array(x).fw.sqrt(x)

# ==========================================
# 3. АГРЕГАЦИИ И РЕДУКЦИИ
# ==========================================

def sum(x: Any, axis: int = None, keepdims: bool = False) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.sum(x, dim=axis if axis is not None else tuple(), keepdim=keepdims)
    elif ctx.isTensorflow():
        return ctx.fw.reduce_sum(x, axis=axis, keepdims=keepdims)
    else:
        return ctx.fw.sum(x, axis=axis, keepdims=keepdims)

def mean(x: Any, axis: int = None, keepdims: bool = False) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.mean(x, dim=axis if axis is not None else tuple(), keepdim=keepdims)
    elif ctx.isTensorflow():
        return ctx.fw.reduce_mean(x, axis=axis, keepdims=keepdims)
    else:
        return ctx.fw.mean(x, axis=axis, keepdims=keepdims)

def max(x: Any, axis: int = None, keepdims: bool = False) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        if axis is not None:
            return ctx.fw.max(x, dim=axis, keepdim=keepdims)[0]
        return ctx.fw.max(x)
    elif ctx.isTensorflow():
        return ctx.fw.reduce_max(x, axis=axis, keepdims=keepdims)
    else:
        return ctx.fw.max(x, axis=axis, keepdims=keepdims)

def argmax(x: Any, axis: int = None) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.argmax(x, dim=axis)
    return ctx.fw.argmax(x, axis=axis)

# ==========================================
# 4. МАНИПУЛЯЦИИ С ФОРМОЙ (Shape Ops)
# ==========================================

def reshape(x: Any, shape: Tuple[int, ...]) -> Any:
    return ArrayContext.from_array(x).fw.reshape(x, shape)

def transpose(x: Any, axes: Tuple[int, ...] = None) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        if axes is None:
            return x.t() if x.ndim == 2 else x.transpose(0, -1)
        return x.permute(*axes)
    return ctx.fw.transpose(x, axes)

def concatenate(tensors: Sequence[Any], axis: int = 0) -> Any:
    if not tensors:
        raise ValueError("Список тензоров пуст")
    ctx = ArrayContext.from_array(tensors[0])
    if ctx.isTorch():
        return ctx.fw.cat(tensors, dim=axis)
    elif ctx.isTensorflow():
        return ctx.fw.concat(tensors, axis=axis)
    else:
        return ctx.fw.concatenate(tensors, axis=axis)
    
# ==========================================
# 5. ПРЕОБРАЗОВАНИЕ И КОНТЕКСТ (Conversion)
# ==========================================

def to_numpy(x: Any) -> 'numpy.ndarray':
    ctx = ArrayContext.from_array(x)
    if ctx.isNumpy(): return x
    elif ctx.isTorch(): return x.detach().cpu().numpy()
    elif ctx.isTensorflow(): return x.numpy()
    elif ctx.isJax():
        import numpy as np
        return np.array(x)
    elif ctx.isCupy(): return x.get()
    raise TypeError(f"Неизвестный тип массива: {type(x)}")
    
def convert_to(x: Any, target_ctx: 'ArrayContext') -> Any:
    source_ctx = ArrayContext.from_array(x)
    if ArrayContext.fw_equal(source_ctx, target_ctx):
        if target_ctx.isTorch():
            # torch.to крашится если dtype=None, поэтому передаем его только если он есть
            if target_ctx.dtype: return x.to(device=target_ctx.device, dtype=target_ctx.dtype)
            return x.to(device=target_ctx.device)
            
        elif target_ctx.isTensorflow():
            import tensorflow as tf
            with tf.device(target_ctx.device):
                return tf.cast(x, target_ctx.dtype) if target_ctx.dtype else tf.identity(x)
                
        elif target_ctx.isJax():
            import jax
            x_casted = x.astype(target_ctx.dtype) if target_ctx.dtype else x
            return jax.device_put(x_casted, target_ctx.device)
            
        else: 
            return x.astype(target_ctx.dtype) if target_ctx.dtype else x

    np_arr = to_numpy(x)
    
    if target_ctx.dtype and target_ctx.isNumpy():
        np_arr = np_arr.astype(target_ctx.dtype)
        
    if target_ctx.isTorch():
        return target_ctx.fw.tensor(np_arr, device=target_ctx.device, dtype=target_ctx.dtype)
    elif target_ctx.isTensorflow():
        import tensorflow as tf
        with tf.device(target_ctx.device):
            return tf.convert_to_tensor(np_arr, dtype=target_ctx.dtype)
    elif target_ctx.isJax():
        import jax.numpy as jnp
        import jax
        arr = jnp.array(np_arr, dtype=target_ctx.dtype)
        return jax.device_put(arr, target_ctx.device)
    elif target_ctx.isCupy():
        return target_ctx.fw.array(np_arr, dtype=target_ctx.dtype)
        
    return np_arr

def zeros_like(x: Any) -> Any:
    ctx = ArrayContext.from_array(x)
    return ctx.fw.zeros_like(x)

def ones_like(x: Any) -> Any:
    ctx = ArrayContext.from_array(x)
    return ctx.fw.ones_like(x)

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/MultiArray/Core.py" encoding="utf-8"
import enum
import types
from typing import Any, TYPE_CHECKING, Type, Union

# ==========================================
# ENUMS (Перечисления)
# ==========================================

class Framework(enum.Enum):
    NUMPY = "numpy"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    CUPY = "cupy"    

class DeviceType(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"       # CUDA или AMD ROCm

# ==========================================
# ARRAY CONTEXT
# ==========================================

StubFW = Any

if TYPE_CHECKING:   
    import numpy
    import torch
    import tensorflow as tf
    import jax
    import cupy

    # Так делать нельзя, но если хотите чтобы тайпчекер дополнял ваши ctx.fw. Игнорируйте (Проверено с pylance)
    StubFW = Type[numpy] | Type[torch] | Type[tf] | Type[jax] | Type[cupy] | Any # type: ignore


class ArrayContext:
    """
    Контекст массива, хранящий информацию о фреймворке, устройстве и типе данных.
    """
    def __init__(self, framework: Framework, device: DeviceType, dtype: Any):
        self._framework = framework
        self._device = device
        self._dtype = dtype

    # = = = Проверки фреймворка = = = 

    def isNumpy(self) -> bool:
        return self._framework == Framework.NUMPY

    def isTorch(self) -> bool:
        return self._framework == Framework.TORCH
        
    def isTensorflow(self) -> bool:
        return self._framework == Framework.TENSORFLOW
        
    def isJax(self) -> bool:
        return self._framework == Framework.JAX
        
    def isCupy(self) -> bool:
        return self._framework == Framework.CUPY

    # = = = Проверки девайса = = =

    def isGPU(self) -> bool:
        return self._device == DeviceType.GPU

    def isCPU(self) -> bool:
        return self._device == DeviceType.CPU

    # = = = Бесшовная интеграция = = = 
    
    @property
    def device(self) -> Union['torch.device', str, 'jax.Device', None]:
        ''' Возвращает реальный экземпляр девайса фреймворка либо None если это numpy или cupy где нет девайсов '''
        if self._framework in (Framework.NUMPY, Framework.CUPY):
            return None
            
        elif self._framework == Framework.TORCH:
            import torch
            dev_str = "cuda" if self._device == DeviceType.GPU else "cpu"
            return torch.device(dev_str)
            
        elif self._framework == Framework.TENSORFLOW:
            dev_str = "GPU" if self._device == DeviceType.GPU else "CPU"
            return f"/{dev_str}:0"
            
        elif self._framework == Framework.JAX:
            import jax
            dev_str = "gpu" if self._device == DeviceType.GPU else "cpu"
            try:
                return jax.devices(dev_str)[0]
            except RuntimeError:
                return jax.devices()[0] # Безопасный фоллбэк
                
        return None

    @property
    def fw(self) -> Union[types.ModuleType, StubFW]:
        ''' Возвращает реальный модуль фреймворка np, tf, torch и тд '''
        if self._framework == Framework.NUMPY:
            import numpy as np
            return np
            
        elif self._framework == Framework.TORCH:
            import torch
            return torch
            
        elif self._framework == Framework.TENSORFLOW:
            import tensorflow as tf
            return tf
            
        elif self._framework == Framework.JAX:
            import jax.numpy as jnp
            return jnp
            
        elif self._framework == Framework.CUPY:
            import cupy as cp
            return cp
            
        import numpy as np
        return np

    @property
    def dtype(self) -> Union['numpy.dtype[Any]', 'torch.dtype', 'tf.DType', 'cupy.dtype[Any]', Any]:
        ''' Возвращает реальный dtype фреймворка. (У JAX тип данных совпадает с numpy.dtype) '''
        return self._dtype

    # = = = Остальной код = = =

    @classmethod
    def from_array(cls, arr: Any) -> 'ArrayContext':
        """Автоматически определяет контекст на основе переданного массива/тензора."""
        
        # 1. Определяем фреймворк
        mod = type(arr).__module__.split('.')[0]
        
        if mod == 'torch':
            fw = Framework.TORCH
        elif mod in ['tensorflow', 'keras']:
            fw = Framework.TENSORFLOW
        elif mod in ['jax', 'jaxlib']:
            fw = Framework.JAX
        elif mod == 'cupy':
            fw = Framework.CUPY
        else:
            fw = Framework.NUMPY

        # 2. Определяем устройство (Оставили только CPU и GPU)
        device_type = DeviceType.CPU
        
        if fw == Framework.TORCH:
            dev_str = str(getattr(arr, 'device', 'cpu')).lower()
            if 'cuda' in dev_str: 
                device_type = DeviceType.GPU
            
        elif fw == Framework.TENSORFLOW:
            dev_str = str(getattr(arr, 'device', '')).lower()
            if 'gpu' in dev_str: 
                device_type = DeviceType.GPU
            
        elif fw == Framework.JAX:
            if hasattr(arr, 'devices'):
                devs = arr.devices()
                if devs:
                    # ИСПРАВЛЕНИЕ: devs это set (множество), используем итератор
                    platform = next(iter(devs)).platform.lower()
                    if platform == 'gpu': 
                        device_type = DeviceType.GPU

                
        elif fw == Framework.CUPY:
            device_type = DeviceType.GPU

        # 3. Определяем dtype
        dtype = getattr(arr, 'dtype', None)

        return cls(framework=fw, device=device_type, dtype=dtype)

    def __repr__(self) -> str:
        return f"ArrayContext(fw={self._framework.name}, dev={self._device.name}, dtype={self._dtype})"
   
    @staticmethod
    def fw_equal(*ctxs: 'ArrayContext') -> bool:
        """Проверяет, что все переданные контексты принадлежат одному фреймворку."""
        if not ctxs:
            return True
        first_fw = ctxs[0]._framework
        return all(c._framework == first_fw for c in ctxs)
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/MultiArray/FFT.py" encoding="utf-8"
from typing import Any, Tuple
from .Core import ArrayContext
from .Base import convert_to, to_numpy

# ==========================================
# 1. ЧАСТОТНЫЕ ОСИ
# ==========================================

def fftfreq(n: int, d: float, ctx: 'ArrayContext') -> Any:
    if ctx.isTorch():
        return ctx.fw.fft.fftfreq(n, d=d, device=ctx.device)
    else:
        import numpy as np
        arr = np.fft.fftfreq(n, d=d)
        return convert_to(arr, ctx)

def rfftfreq(n: int, d: float, ctx: 'ArrayContext') -> Any:
    if ctx.isTorch():
        return ctx.fw.fft.rfftfreq(n, d=d, device=ctx.device)
    else:
        import numpy as np
        arr = np.fft.rfftfreq(n, d=d)
        return convert_to(arr, ctx)

# ==========================================
# 2. 2D ПРЕОБРАЗОВАНИЯ ФУРЬЕ (FFT)
# ==========================================

def fft2(x: Any, axes: Tuple[int, int] = (-2, -1)) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.fft.fft2(x, dim=axes)
    elif ctx.isTensorflow():
        return ctx.fw.signal.fft2d(x)
    else:
        return ctx.fw.fft.fft2(x, axes=axes)

def ifft2(x: Any, axes: Tuple[int, int] = (-2, -1)) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.fft.ifft2(x, dim=axes)
    elif ctx.isTensorflow():
        return ctx.fw.signal.ifft2d(x)
    else:
        return ctx.fw.fft.ifft2(x, axes=axes)

def rfft2(x: Any, axes: Tuple[int, int] = (-2, -1)) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.fft.rfft2(x, dim=axes)
    elif ctx.isTensorflow():
        return ctx.fw.signal.rfft2d(x)
    else:
        return ctx.fw.fft.rfft2(x, axes=axes)

def irfft2(x: Any, s: Tuple[int, int] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
    ctx = ArrayContext.from_array(x)
    if ctx.isTorch():
        return ctx.fw.fft.irfft2(x, s=s, dim=axes)
    elif ctx.isTensorflow():
        return ctx.fw.signal.irfft2d(x)
    else:
        return ctx.fw.fft.irfft2(x, s=s, axes=axes)

# ==========================================
# 3. СЛОЖНАЯ ОБРАБОТКА СИГНАЛОВ (DSP)
# ==========================================

def stft(signal: Any, frame_length: int, frame_step: int, fft_length: int = None, 
         window: Any = None, pad_end: bool = False) -> Any:
    ctx = ArrayContext.from_array(signal)
    if fft_length is None:
        fft_length = frame_length

    pad_len = 0
    sig_len = signal.shape[-1]

    if pad_end:
        if sig_len < frame_length:
            pad_len = frame_length - sig_len
        else:
            rem = (sig_len - frame_length) % frame_step
            pad_len = (frame_step - rem) % frame_step if rem != 0 else 0

    if ctx.isTorch() and fft_length > frame_length:
        pad_len += (fft_length - frame_length)

    if pad_len > 0:
        if ctx.isTensorflow():
            paddings = [[0, 0]] * (len(signal.shape) - 1) + [[0, pad_len]]
            signal = ctx.fw.pad(signal, paddings)
        elif ctx.isTorch():
            import torch.nn.functional as F
            signal = F.pad(signal, (0, pad_len))
        else:
            import numpy as np
            signal_np = to_numpy(signal)
            pad_tuple = [(0, 0)] * (signal_np.ndim - 1) + [(0, pad_len)]
            signal = np.pad(signal_np, pad_tuple)
            if not ctx.isNumpy():
                signal = convert_to(signal, ctx)

    if ctx.isTensorflow():
        # ИСПРАВЛЕНИЕ: TF требует чтобы dtype окна строго совпадал с dtype сигнала (даже если он complex)
        def win_fn(length, dtype):
            if window is None:
                return None
            w = ctx.fw.convert_to_tensor(window)
            if w.dtype != dtype:
                w = ctx.fw.cast(w, dtype)
            return w
            
        res = ctx.fw.signal.stft(signal, frame_length=frame_length, frame_step=frame_step, 
                                 fft_length=fft_length, window_fn=win_fn if window is not None else None, 
                                 pad_end=False)
        return ctx.fw.linalg.matrix_transpose(res)
                                  
    elif ctx.isTorch():
        import torch.nn.functional as F
        if window is not None and fft_length > frame_length:
            pt_window = F.pad(window, (0, fft_length - frame_length))
        else:
            pt_window = window
            
        stft_matrix = ctx.fw.stft(signal, n_fft=fft_length, hop_length=frame_step, 
                                  win_length=fft_length, window=pt_window, 
                                  center=False, return_complex=True)
        return stft_matrix
        
    else:
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view
        
        np_sig = to_numpy(signal)
        np_win = to_numpy(window) if window is not None else np.ones(frame_length, dtype=np_sig.dtype)
        
        frames = sliding_window_view(np_sig, frame_length, axis=-1)
        frames = frames[..., ::frame_step, :]
        frames = frames * np_win
        
        res = np.fft.rfft(frames, n=fft_length, axis=-1)
        res = np.swapaxes(res, -2, -1)
        
        target_ctx = ArrayContext(ctx._framework, ctx._device, None)
        return convert_to(res, target_ctx)


def istft(stft_matrix: Any, frame_length: int, frame_step: int, fft_length: int = None, 
          window: Any = None, length: int = None) -> Any:
    ctx = ArrayContext.from_array(stft_matrix)
    if fft_length is None:
        fft_length = frame_length
        
    if ctx.isTensorflow():
        def win_fn(length, dtype):
            if window is None:
                return None
            w = ctx.fw.convert_to_tensor(window)
            if w.dtype != dtype:
                w = ctx.fw.cast(w, dtype)
            return w

        stfts = ctx.fw.linalg.matrix_transpose(stft_matrix)
        res = ctx.fw.signal.inverse_stft(stfts, frame_length=frame_length, 
                                         frame_step=frame_step, fft_length=fft_length, 
                                         window_fn=win_fn if window is not None else None)
        if length is not None:
            res = res[..., :length]
        return res
        
    elif ctx.isTorch():
        stfts = stft_matrix.transpose(-2, -1)
        frames = ctx.fw.fft.irfft(stfts, n=fft_length, dim=-1)
        
        frames = frames[..., :frame_length]
        
        if window is not None:
            frames = frames * window[:frames.shape[-1]]
            
        num_frames = frames.shape[-2]
        expected_len = (num_frames - 1) * frame_step + frame_length
        
        out_shape = list(frames.shape[:-2]) + [expected_len]
        out_sig = ctx.fw.zeros(out_shape, dtype=frames.dtype, device=frames.device)
        
        for i in range(num_frames):
            start = i * frame_step
            out_sig[..., start:start+frames.shape[-1]] += frames[..., i, :]
            
        if length is not None:
            out_sig = out_sig[..., :length]
            
        return out_sig
                            
    else:
        import numpy as np
        stfts = np.swapaxes(to_numpy(stft_matrix), -2, -1)
        win = to_numpy(window) if window is not None else 1.0
        
        frames = np.fft.irfft(stfts, n=fft_length, axis=-1)
        
        frames = frames[..., :frame_length]
        
        if window is not None:
            frames = frames * win[:frames.shape[-1]]
        
        num_frames = frames.shape[-2]
        expected_len = (num_frames - 1) * frame_step + frame_length
        
        out_shape = list(frames.shape[:-2]) + [expected_len]
        out_sig = np.zeros(out_shape, dtype=frames.dtype)
        
        for i in range(num_frames):
            start = i * frame_step
            out_sig[..., start:start+frames.shape[-1]] += frames[..., i, :]
            
        if length is not None:
            out_sig = out_sig[..., :length]
            
        target_ctx = ArrayContext(ctx._framework, ctx._device, None)
        return convert_to(out_sig, target_ctx)

# ==========================================
# 4. ИНТЕРПОЛЯЦИЯ 2D СЕТОК
# ==========================================

def interpolate_2d(matrix: Any, old_x: Any, old_y: Any, new_x: Any, new_y: Any) -> Any:
    ctx = ArrayContext.from_array(matrix)
    
    if ctx.isTorch():
        import torch
        import torch.nn.functional as F
        
        is_complex = matrix.is_complex()
        
        def normalize(coords, old_coords):
            c_min, c_max = old_coords[0], old_coords[-1]
            return 2.0 * (coords - c_min) / (c_max - c_min) - 1.0
            
        norm_x = normalize(new_x, old_x)
        norm_y = normalize(new_y, old_y)
        
        grid_y, grid_x = torch.meshgrid(norm_y, norm_x, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        
        orig_shape = matrix.shape
        m = matrix.unsqueeze(0).unsqueeze(0) if len(orig_shape) == 2 else matrix.unsqueeze(1)
            
        if is_complex:
            out_r = F.grid_sample(m.real, grid.to(m.device), align_corners=True)
            out_i = F.grid_sample(m.imag, grid.to(m.device), align_corners=True)
            out = torch.complex(out_r, out_i)
        else:
            out = F.grid_sample(m, grid.to(m.device), align_corners=True)
            
        return out.squeeze(1).squeeze(0) if len(orig_shape) == 2 else out.squeeze(1)
        
    else:
        import numpy as np
        from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
        
        np_mat = to_numpy(matrix)
        nx, ny = to_numpy(new_x), to_numpy(new_y)
        ox, oy = to_numpy(old_x), to_numpy(old_y)
        
        is_complex = np.iscomplexobj(np_mat)
        
        if np_mat.ndim == 2:
            if is_complex:
                spline_r = RectBivariateSpline(oy, ox, np_mat.real)
                spline_i = RectBivariateSpline(oy, ox, np_mat.imag)
                res = spline_r(ny, nx) + 1j * spline_i(ny, nx)
            else:
                spline = RectBivariateSpline(oy, ox, np_mat)
                res = spline(ny, nx)
        else:
            interpolator = RegularGridInterpolator((oy, ox), np_mat, method='linear', bounds_error=False, fill_value=0)
            X, Y = np.meshgrid(nx, ny)
            pts = np.stack([Y.ravel(), X.ravel()], axis=-1)
            res = interpolator(pts).reshape(*np_mat.shape[:-2], len(ny), len(nx))
            
        return convert_to(res, ctx)
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/MultiArray/Linalg.py" encoding="utf-8"
from typing import Any, Tuple
from .Core import ArrayContext

# ==========================================
# УРОВЕНЬ 2: ЛИНЕЙНАЯ АЛГЕБРА (Linalg)
# ==========================================

def matmul(a: Any, b: Any) -> Any:
    """Умножение матриц (матричное произведение)."""
    ctx = ArrayContext.from_array(a)
    return ctx.fw.matmul(a, b)

def dot(a: Any, b: Any) -> Any:
    """
    Скалярное произведение.
    Разрешает проблемы PyTorch (где dot работает ТОЛЬКО с 1D-векторами).
    """
    ctx = ArrayContext.from_array(a)
    if ctx.isTensorflow():
        return ctx.fw.tensordot(a, b, axes=1)
    elif ctx.isTorch():
        if a.ndim == 1 and b.ndim == 1:
            return ctx.fw.dot(a, b)
        return ctx.fw.tensordot(a, b, dims=1)
    else:
        return ctx.fw.dot(a, b)

def einsum(equation: str, *operands: Any) -> Any:
    """Обобщенное сокращение тензоров (Einstein summation convention)."""
    ctx = ArrayContext.from_array(operands[0])
    return ctx.fw.einsum(equation, *operands)

def inv(a: Any) -> Any:
    """Обратная матрица квадратной матрицы."""
    ctx = ArrayContext.from_array(a)
    # К счастью, почти все фреймворки договорились класть inv в linalg
    return ctx.fw.linalg.inv(a)

def svd(a: Any, full_matrices: bool = True) -> Tuple[Any, Any, Any]:
    """
    Сингулярное разложение матрицы SVD (Singular Value Decomposition).
    Унифицированно возвращает кортеж: (U, S, Vh).
    
    ВНИМАНИЕ: Разрешает конфликт TensorFlow, который по умолчанию возвращает (S, U, V).
    """
    ctx = ArrayContext.from_array(a)
    
    if ctx.isTensorflow():
        # TensorFlow возвращает (S, U, V), где a = U * diag(S) * V^H
        # NumPy и PyTorch возвращают (U, S, Vh). Vh — это эрмитово-сопряженная матрица V.
        s, u, v = ctx.fw.linalg.svd(a, full_matrices=full_matrices)
        vh = ctx.fw.linalg.adjoint(v)
        return u, s, vh
        
    else:
        # NumPy, PyTorch (torch.linalg.svd), JAX, CuPy
        return ctx.fw.linalg.svd(a, full_matrices=full_matrices)

def eig(a: Any) -> Tuple[Any, Any]:
    """
    Собственные значения и собственные векторы.
    Возвращает: (eigenvalues, eigenvectors).
    """
    ctx = ArrayContext.from_array(a)
    return ctx.fw.linalg.eig(a)

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/MultiArray/test_base.py" encoding="utf-8"
import pytest
import numpy as np

import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType

# ==========================================
# ОБНАРУЖЕНИЕ УСТАНОВЛЕННЫХ ФРЕЙМВОРКОВ
# ==========================================

AVAILABLE_FRAMEWORKS = [Framework.NUMPY]

try: import torch; AVAILABLE_FRAMEWORKS.append(Framework.TORCH)
except ImportError: pass

try: import tensorflow as tf; AVAILABLE_FRAMEWORKS.append(Framework.TENSORFLOW)
except ImportError: pass

try: import jax; AVAILABLE_FRAMEWORKS.append(Framework.JAX)
except ImportError: pass

try: import cupy; AVAILABLE_FRAMEWORKS.append(Framework.CUPY)
except ImportError: pass

CPU_CONTEXTS = [
    ArrayContext(fw, DeviceType.CPU, None) for fw in AVAILABLE_FRAMEWORKS 
    if fw != Framework.CUPY
]

# Хелпер для Pytest, чтобы красиво называть тесты без доступа к _framework
def get_fw_name(ctx: ArrayContext) -> str:
    if ctx.isNumpy(): return "NUMPY"
    if ctx.isTorch(): return "TORCH"
    if ctx.isTensorflow(): return "TENSORFLOW"
    if ctx.isJax(): return "JAX"
    if ctx.isCupy(): return "CUPY"
    return "UNKNOWN"

# ==========================================
# ТЕСТЫ
# ==========================================

@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_creation_and_conversion(ctx):
    # 1. Создаем массив через наш API
    arr = ma.arange(5, ctx)
    
    # ПРОБЛЕМА РЕШЕНА: Используем fw_equal вместо .framework
    assert ArrayContext.fw_equal(ArrayContext.from_array(arr), ctx)
    
    # 2. Создаем zeros_like
    z = ma.zeros_like(arr)
    assert ArrayContext.fw_equal(ArrayContext.from_array(z), ctx)
    
    # 3. Перегоняем в NumPy и сравниваем с эталоном
    np_arr = ma.to_numpy(arr)
    np_z = ma.to_numpy(z)
    
    assert isinstance(np_arr, np.ndarray)
    np.testing.assert_array_equal(np_arr, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(np_z, np.array([0, 0, 0, 0, 0]))


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_math_operations(ctx):
    np_ref = np.array([[-1.0, 2.0], [3.0, -4.0]])
    arr = ma.convert_to(np_ref, ctx)
    
    arr_abs = ma.abs(arr)
    arr_sum = ma.sum(arr, axis=0)
    arr_max = ma.max(arr)
    
    np.testing.assert_allclose(ma.to_numpy(arr_abs), np.abs(np_ref))
    np.testing.assert_allclose(ma.to_numpy(arr_sum), np.sum(np_ref, axis=0))
    np.testing.assert_allclose(ma.to_numpy(arr_max), np.max(np_ref))


def test_cross_framework_conversion():
    if Framework.TORCH not in AVAILABLE_FRAMEWORKS or Framework.TENSORFLOW not in AVAILABLE_FRAMEWORKS:
        pytest.skip("Нужны Torch и TensorFlow")
        
    ctx_torch = ArrayContext(Framework.TORCH, DeviceType.CPU, None)
    ctx_tf = ArrayContext(Framework.TENSORFLOW, DeviceType.CPU, None)
    
    t_arr = ma.arange(10, ctx_torch)
    # Используем isTorch()
    assert ArrayContext.from_array(t_arr).isTorch()
    
    tf_arr = ma.convert_to(t_arr, ctx_tf)
    # Используем isTensorflow()
    assert ArrayContext.from_array(tf_arr).isTensorflow()
    
    np.testing.assert_array_equal(ma.to_numpy(tf_arr), np.arange(10))


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_shape_manipulations(ctx):
    arr1 = ma.convert_to(np.zeros((2, 3)), ctx)
    arr2 = ma.convert_to(np.ones((2, 3)), ctx)
    
    concat_arr = ma.concatenate([arr1, arr2], axis=0)
    assert ma.to_numpy(concat_arr).shape == (4, 3)
    
    reshaped_arr = ma.reshape(concat_arr, (3, 4))
    assert ma.to_numpy(reshaped_arr).shape == (3, 4)
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/MultiArray/test_fft.py" encoding="utf-8"
import pytest
import numpy as np

import MultiArray as ma
import MultiArray.FFT as mfft
from MultiArray.Core import ArrayContext, Framework, DeviceType

# ==========================================
# ОБНАРУЖЕНИЕ УСТАНОВЛЕННЫХ ФРЕЙМВОРКОВ
# ==========================================

AVAILABLE_FRAMEWORKS = [Framework.NUMPY]

try: import torch; AVAILABLE_FRAMEWORKS.append(Framework.TORCH)
except ImportError: pass

try: import tensorflow as tf; AVAILABLE_FRAMEWORKS.append(Framework.TENSORFLOW)
except ImportError: pass

try: import jax; AVAILABLE_FRAMEWORKS.append(Framework.JAX)
except ImportError: pass

CPU_CONTEXTS = [
    ArrayContext(fw, DeviceType.CPU, None) for fw in AVAILABLE_FRAMEWORKS 
    if fw != Framework.CUPY
]

def get_fw_name(ctx: ArrayContext) -> str:
    if ctx.isNumpy(): return "NUMPY"
    if ctx.isTorch(): return "TORCH"
    if ctx.isTensorflow(): return "TENSORFLOW"
    if ctx.isJax(): return "JAX"
    return "UNKNOWN"

# ==========================================
# ТЕСТЫ ДЛЯ FFT И DSP
# ==========================================

@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_math_extensions(ctx):
    np_arr = np.array([1.0, 10.0, 100.0], dtype=np.float32)
    arr = ma.convert_to(np_arr, ctx)
    
    # Test log10
    arr_log10 = ma.log10(arr)
    np.testing.assert_allclose(ma.to_numpy(arr_log10), [0.0, 1.0, 2.0], rtol=1e-5)
    
    # Test clamp
    arr_clamp = ma.clamp_min(arr, 5.0)
    np.testing.assert_allclose(ma.to_numpy(arr_clamp), [5.0, 10.0, 100.0])


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_fftfreq(ctx):
    n = 10
    d = 0.1
    
    freqs = mfft.fftfreq(n, d, ctx)
    expected = np.fft.fftfreq(n, d)
    
    np.testing.assert_allclose(ma.to_numpy(freqs), expected, rtol=1e-5)


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_fft2_ifft2(ctx):
    np_data = np.random.rand(4, 4).astype(np.float32)
    data = ma.convert_to(np_data, ctx)
    
    if ctx.isTensorflow():
        import tensorflow as tf
        data = tf.cast(data, tf.complex64)
    
    fft_res = mfft.fft2(data)
    ifft_res = mfft.ifft2(fft_res)
    
    np.testing.assert_allclose(np.real(ma.to_numpy(ifft_res)), np_data, atol=1e-5)


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_stft_istft(ctx):
    signal_len = 1024
    np_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, signal_len)).astype(np.float32)
    
    frame_length = 256
    frame_step = 64
    fft_length = 256
    np_window = np.hanning(frame_length).astype(np.float32)
    
    signal = ma.convert_to(np_signal, ctx)
    window = ma.convert_to(np_window, ctx)
    
    # STFT (Unified Style)
    stft_matrix = mfft.stft(signal, frame_length=frame_length, frame_step=frame_step, 
                            fft_length=fft_length, window=window, pad_end=True)
    
    np_stft = ma.to_numpy(stft_matrix)
    
    # Проверка нашей новой размерности [..., Bins, Frames]
    assert np_stft.shape[-2] == fft_length // 2 + 1
    
    # iSTFT
    recon_signal = mfft.istft(stft_matrix, frame_length=frame_length, frame_step=frame_step, 
                              fft_length=fft_length, window=window, length=signal_len)
    
    np_recon = ma.to_numpy(recon_signal)
    
    np_recon_norm = np_recon / np.max(np.abs(np_recon))
    np_sig_norm = np_signal / np.max(np.abs(np_signal))
    
    # Сравниваем обрезанные края из-за краевых эффектов окна
    np.testing.assert_allclose(np_sig_norm[256:-256], np_recon_norm[256:-256], atol=1e-3)

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/MultiArray/test_linalg.py" encoding="utf-8"
import pytest
import numpy as np

import MultiArray as ma
import MultiArray.Linalg as ml
from MultiArray.Core import ArrayContext, Framework, DeviceType

# ==========================================
# ОБНАРУЖЕНИЕ УСТАНОВЛЕННЫХ ФРЕЙМВОРКОВ
# ==========================================

AVAILABLE_FRAMEWORKS = [Framework.NUMPY]

try: import torch; AVAILABLE_FRAMEWORKS.append(Framework.TORCH)
except ImportError: pass

try: import tensorflow as tf; AVAILABLE_FRAMEWORKS.append(Framework.TENSORFLOW)
except ImportError: pass

try: import jax; AVAILABLE_FRAMEWORKS.append(Framework.JAX)
except ImportError: pass

# Не тестируем Linalg на GPU в рамках базовых тестов, чтобы избежать падений из-за памяти
CPU_CONTEXTS = [
    ArrayContext(fw, DeviceType.CPU, None) for fw in AVAILABLE_FRAMEWORKS 
    if fw != Framework.CUPY
]

def get_fw_name(ctx: ArrayContext) -> str:
    if ctx.isNumpy(): return "NUMPY"
    if ctx.isTorch(): return "TORCH"
    if ctx.isTensorflow(): return "TENSORFLOW"
    if ctx.isJax(): return "JAX"
    return "UNKNOWN"

# ==========================================
# ТЕСТЫ LINALG
# ==========================================

@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_matmul_and_dot(ctx):
    # Двумерные матрицы
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np_b = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    
    a = ma.convert_to(np_a, ctx)
    b = ma.convert_to(np_b, ctx)
    
    res_matmul = ml.matmul(a, b)
    np.testing.assert_allclose(ma.to_numpy(res_matmul), np.matmul(np_a, np_b), rtol=1e-5, atol=1e-5)
    
    # Одномерные векторы
    np_v1 = np.array([1.0, 2.0], dtype=np.float32)
    np_v2 = np.array([3.0, 4.0], dtype=np.float32)
    
    v1 = ma.convert_to(np_v1, ctx)
    v2 = ma.convert_to(np_v2, ctx)
    
    res_dot = ml.dot(v1, v2)
    np.testing.assert_allclose(ma.to_numpy(res_dot), np.dot(np_v1, np_v2), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_inv(ctx):
    # Матрица, которая легко обращается
    np_a = np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float32)
    a = ma.convert_to(np_a, ctx)
    
    a_inv = ml.inv(a)
    np.testing.assert_allclose(ma.to_numpy(a_inv), np.linalg.inv(np_a), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_svd(ctx):
    # Используем вещественную матрицу (float32 критично для SVD в TF)
    np_a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    a = ma.convert_to(np_a, ctx)
    
    u, s, vh = ml.svd(a, full_matrices=False)
    
    np_u = ma.to_numpy(u)
    np_s = ma.to_numpy(s)
    np_vh = ma.to_numpy(vh)
    
    # Проверяем размерности (full_matrices=False)
    assert np_u.shape == (3, 2)
    assert np_s.shape == (2,)
    assert np_vh.shape == (2, 2)
    
    # Проверяем реконструкцию исходной матрицы: A = U * S * Vh
    # У SVD есть проблема неоднозначности знаков у разных бэкендов, поэтому 
    # лучше проверять именно успешность восстановления исходной матрицы.
    reconstructed = np_u @ np.diag(np_s) @ np_vh
    np.testing.assert_allclose(reconstructed, np_a, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("ctx", CPU_CONTEXTS, ids=[get_fw_name(c) for c in CPU_CONTEXTS])
def test_einsum(ctx):
    # Тестируем транспонирование и умножение через einsum: C_ij = A_ki * B_kj
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np_b = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    
    a = ma.convert_to(np_a, ctx)
    b = ma.convert_to(np_b, ctx)
    
    # "ki,kj->ij"
    res = ml.einsum("ki,kj->ij", a, b)
    expected = np.einsum("ki,kj->ij", np_a, np_b)
    
    np.testing.assert_allclose(ma.to_numpy(res), expected, rtol=1e-5, atol=1e-5)

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Functions/__init__.py" encoding="utf-8"
from pathlib import Path
from typing import Any, Iterator, Self, Callable, Union
from jaxtyping import Float, Shaped

# --- 1. Импорты специфичных для проекта библиотек ---
from BatSpec.Core.Physical.Units import PintUnit, Phisical, UREG, unit_mul
from BatSpec.Core.SaveIntegral import SaveIntegral

# Внедряем наш новый универсальный фреймворк
import MultiArray as ma
from MultiArray import ArrayContext

ArrayLike = Any 

# ==========================================
# БАЗОВЫЕ КЛАССЫ
# ==========================================

class Function:
    @property
    def _primary_tensor(self) -> ArrayLike:
        """Абстрактное свойство: возвращает главный тензор объекта для определения контекста."""
        raise NotImplementedError()

    @property
    def context(self) -> ArrayContext:
        """Единый контекст массива (фреймворк, устройство, тип данных)."""
        return ArrayContext.from_array(self._primary_tensor)


class Function1D(Function, SaveIntegral[Phisical[Shaped[ArrayLike, '...']]]):
    _value_a: Shaped[ArrayLike, '... x']
    _value_u: PintUnit
    _axis__a: Shaped[ArrayLike, 'x']
    _axis__u: PintUnit

    @property
    def _primary_tensor(self) -> ArrayLike:
        return self._value_a

    def __init__(self, values: Phisical[Shaped[ArrayLike, '... x']], axis: Phisical[Float[ArrayLike, "axis"]]):
        self._value_a, self._value_u = values
        self._axis__a, self._axis__u = axis
       
    @property
    def _d_axis__a(self) -> Phisical[float]:
        if len(self._axis__a) < 2:
            raise ValueError("Cannot calculate step size for an AnyArray with less than 2 elements.")
        step = float((self._axis__a[-1] - self._axis__a[0]) / (len(self._axis__a) - 1))
        return step, self._axis__u

    def _SaveIntegral_Energy(self) -> Phisical[Shaped[ArrayLike, '...']]:
        dA, dU = self._d_axis__a
        energy_tensor = ma.abs(self._value_a) ** 2
        integral = ma.sum(energy_tensor, axis=-1) * dA
        return integral, unit_mul(self._value_u, self._value_u, dU)

    def _SaveIntegral_Area(self) -> Phisical[Shaped[ArrayLike, '...']]:
        dA, dU = self._d_axis__a
        integral = ma.sum(self._value_a, axis=-1) * dA
        return integral, unit_mul(self._value_u, dU)

    def _FileSystem_Save(self, path: Path) -> None:
        raise NotImplementedError()
    
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        raise NotImplementedError()

    def __len__(self) -> int:
        if self._value_a.ndim <= 1:
            raise TypeError("Нет батчевого измерения.")
        return self._value_a.shape[0]

    def __getitem__(self, idx: Union[int, slice]) -> 'Function1D':
        if self._value_a.ndim <= 1:
            raise TypeError("Нет батчевого измерения.")
        return Function1D(
            values=(self._value_a[idx], self._value_u),
            axis=(self._axis__a, self._axis__u)
        )

    def __iter__(self) -> Iterator['Function1D']:
        for i in range(len(self)):
            yield self[i]

    def to_context(self, ctx: ArrayContext) -> 'Function1D':
        """Новый стандартный метод миграции данных."""
        new_value_a = ma.convert_to(self._value_a, ctx)
        
        # ИСПРАВЛЕНИЕ АРХИТЕКТУРЫ: Оси не должны перенимать dtype матрицы (особенно если матрица комплексная).
        # Передавая None в dtype, мы заставляем фреймворк сохранить естественный (вещественный) тип оси.
        axis_ctx = ArrayContext(ctx._framework, ctx._device, None)
        new_axis__a = ma.convert_to(self._axis__a, axis_ctx)
        
        return Function1D(
            values=(new_value_a, self._value_u), 
            axis=(new_axis__a, self._axis__u)
        )

class TimeFunc(Function1D):
    _axis__u: PintUnit = UREG.second

    def __init__(self, values: Phisical[Shaped[ArrayLike, '... time']], axis: Float[ArrayLike, 'time']):
        if values[0].shape[-1] != axis.shape[-1]: 
            raise ValueError(f"Последнее измерение values ({values[0].shape[-1]}) и длина axis ({axis.shape[-1]}) должны совпадать")
        self._value_a, self._value_u = values
        self._axis__a = axis

    @classmethod
    def from_Function1D(cls, func: Function1D) -> Self:
        return cls((func._value_a, func._value_u), func._axis__a)
    
    @property
    def time(self) -> Phisical[Float[ArrayLike, 'time']]:
        return self._axis__a, self._axis__u
    
    @time.setter
    def time(self, value: Float[ArrayLike, 'time']):
        if len(value) != len(self._axis__a):
            raise ValueError(f"New time AnyArray must have length {len(self._axis__a)}, got {len(value)}")
        self._axis__a = value

    @property
    def start(self) -> float:
        return float(self._axis__a[0])
    
    @property
    def end(self) -> float:
        return float(self._axis__a[-1])
    
    @property
    def values(self) -> Phisical[Shaped[ArrayLike, '... time']]:
        return self._value_a, self._value_u

    @values.setter
    def values(self, value: Phisical[Shaped[ArrayLike, '... time']]):
        if value[0].shape[-1] != self._axis__a.shape[-1]:
            raise ValueError(f"Новый массив значений должен иметь размер последнего измерения {self._axis__a.shape[-1]}, получено {value[0].shape[-1]}")
        self._value_a, self._value_u = value
        
    @property
    def dt(self) -> Phisical[float]:
        return self._d_axis__a

    @property
    def sr(self) -> int:
        dT, _ = self.dt
        return round(1.0 / dT)

    def _FileSystem_Save(self, path: Path):
        Function1D._FileSystem_Save(self, path)
        
    @classmethod
    def _FileSystem_Load(cls, path: Path):
        return cls.from_Function1D(Function1D._FileSystem_Load(path))

    def __getitem__(self, idx: Union[int, slice]) -> 'TimeFunc':
        return TimeFunc.from_Function1D(Function1D.__getitem__(self, idx))
    
    def __iter__(self) -> Iterator['TimeFunc']:
        for i in range(len(self)):
            yield self[i]

    def to_context(self, ctx: ArrayContext) -> 'TimeFunc':
        return TimeFunc.from_Function1D(Function1D.to_context(self, ctx))


class FreqFunc(Function1D):
    _axis__u: PintUnit = UREG.hertz

    def __init__(self, values: Phisical[Shaped[ArrayLike, '... freq']], axis: Float[ArrayLike, 'freq']):
        if values[0].shape[-1] != axis.shape[-1]: 
            raise ValueError(f"Последнее измерение values ({values[0].shape[-1]}) и длина axis ({axis.shape[-1]}) должны совпадать")
        self._value_a, self._value_u = values
        self._axis__a = axis

    @classmethod
    def from_Function1D(cls, func: Function1D) -> Self:
        return cls((func._value_a, func._value_u), func._axis__a)
    
    @property
    def freq(self) -> Phisical[Float[ArrayLike, 'freq']]:
        return self._axis__a, self._axis__u

    @freq.setter
    def freq(self, value: Float[ArrayLike, 'freq']):
        if len(value) != len(self._axis__a):
            raise ValueError(f"Новый массив частот должен иметь длину {len(self._axis__a)}, получено {len(value)}")
        self._axis__a = value

    @property
    def min_freq(self) -> float:
        return float(self._axis__a[0])
    
    @property
    def max_freq(self) -> float:
        return float(self._axis__a[-1])
    
    @property
    def values(self) -> Phisical[Shaped[ArrayLike, '... freq']]:
        return self._value_a, self._value_u

    @values.setter
    def values(self, value: Phisical[Shaped[ArrayLike, '... freq']]):
        if value[0].shape[-1] != self._axis__a.shape[-1]:
            raise ValueError(f"Новый массив значений должен иметь размер последнего измерения {self._axis__a.shape[-1]}, получено {value[0].shape[-1]}")
        self._value_a, self._value_u = value
        
    @property
    def df(self) -> Phisical[float]:
        return self._d_axis__a

    def _FileSystem_Save(self, path: Path):
        Function1D._FileSystem_Save(self, path)
        
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        return cls.from_Function1D(Function1D._FileSystem_Load(path))
    
    def __getitem__(self, idx: Union[int, slice]) -> 'FreqFunc':
        return FreqFunc.from_Function1D(Function1D.__getitem__(self, idx))
    
    def __iter__(self) -> Iterator['FreqFunc']:
        for i in range(len(self)):
            yield self[i]
    
    def to_context(self, ctx: ArrayContext) -> 'FreqFunc':
        return FreqFunc.from_Function1D(Function1D.to_context(self, ctx))


class Function2D(Function, SaveIntegral[Phisical[Shaped[ArrayLike, '...']]]):
    _matrx_a: Shaped[ArrayLike, '... x y']
    _matrx_u: PintUnit
    _first_a: Shaped[ArrayLike, 'x']
    _first_u: PintUnit
    _sec___a: Shaped[ArrayLike, 'y']
    _sec___u: PintUnit

    @property
    def _primary_tensor(self) -> ArrayLike:
        return self._matrx_a

    def __init__(self, 
                 matrix: Phisical[Shaped[ArrayLike, '... x y']], 
                 first: Phisical[Float[ArrayLike, "x"]], 
                 sec: Phisical[Float[ArrayLike, "y"]]):
        self._matrx_a, self._matrx_u = matrix
        self._first_a, self._first_u = first
        self._sec___a, self._sec___u = sec

    @property
    def _d_first_a(self) -> Phisical[float]:
        if len(self._first_a) < 2:
            raise ValueError("Cannot calculate step size for an array with less than 2 elements.")
        step = float((self._first_a[-1] - self._first_a[0]) / (len(self._first_a) - 1))
        return step, self._first_u

    @property
    def _d_sec___a(self) -> Phisical[float]:
        if len(self._sec___a) < 2:
            raise ValueError("Cannot calculate step size for an array with less than 2 elements.")
        step = float((self._sec___a[-1] - self._sec___a[0]) / (len(self._sec___a) - 1))
        return step, self._sec___u

    def _SaveIntegral_Area(self) -> Phisical[Shaped[ArrayLike, '...']]:
        f_da, f_du = self._d_first_a
        s_da, s_du = self._d_sec___a
        integral = ma.sum(self._matrx_a, axis=(-2, -1)) * f_da * s_da
        return integral, unit_mul(self._matrx_u, f_du, s_du)
    
    def _SaveIntegral_Energy(self) -> Phisical[Shaped[ArrayLike, '...']]:
        f_da, f_du = self._d_first_a
        s_da, s_du = self._d_sec___a
        energy_tensor = ma.abs(self._matrx_a) ** 2
        integral = ma.sum(energy_tensor, axis=(-2, -1)) * f_da * s_da
        return integral, unit_mul(self._matrx_u, self._matrx_u, f_du, s_du)

    def _FileSystem_Save(self, path: Path) -> None:
        raise NotImplementedError()
    
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        raise NotImplementedError()

    def __len__(self) -> int:
        if self._matrx_a.ndim <= 2:
            raise TypeError("Нет батчевого измерения.")
        return self._matrx_a.shape[0]

    def __getitem__(self, idx: Union[int, slice]) -> 'Function2D':
        if self._matrx_a.ndim <= 2:
            raise TypeError("Нет батчевого измерения.")
        return Function2D(
            matrix=(self._matrx_a[idx], self._matrx_u),
            first=(self._first_a, self._first_u),
            sec=(self._sec___a, self._sec___u)
        )

    def __iter__(self) -> Iterator['Function2D']:
        for i in range(len(self)):
            yield self[i]

    def to_context(self, ctx: ArrayContext) -> 'Function2D':
        new_matrx = ma.convert_to(self._matrx_a, ctx)
        
        # ИСПРАВЛЕНИЕ АРХИТЕКТУРЫ: Изолируем оси от комплексного типа матрицы
        axis_ctx = ArrayContext(ctx._framework, ctx._device, None)
        new_first = ma.convert_to(self._first_a, axis_ctx)
        new_sec   = ma.convert_to(self._sec___a, axis_ctx)

        return Function2D(
            matrix=(new_matrx, self._matrx_u),
            first=(new_first, self._first_u),
            sec=(new_sec, self._sec___u)
        )


class SpecFunc(Function2D):
    _first_u: PintUnit = UREG.hertz
    _sec___u: PintUnit = UREG.second

    def __init__(self, 
                 matrix: Phisical[Shaped[ArrayLike, '... freq time']], 
                 freq: Float[ArrayLike, 'freq'],
                 time: Float[ArrayLike, 'time']):
        if matrix[0].shape[-2] != freq.shape[-1]:
            raise ValueError(f"Размер матрицы по оси частот ({matrix[0].shape[-2]}) не совпадает с длиной оси freq ({freq.shape[-1]})")
        if matrix[0].shape[-1] != time.shape[-1]:
            raise ValueError(f"Размер матрицы по оси времени ({matrix[0].shape[-1]}) не совпадает с длиной оси time ({time.shape[-1]})")
        
        self._matrx_a, self._matrx_u = matrix
        self._first_a = freq  
        self._sec___a = time  

    @classmethod
    def from_Function2D(cls, func: Function2D) -> Self:
        return cls(
            matrix=(func._matrx_a, func._matrx_u), 
            freq=func._first_a, 
            time=func._sec___a
        )

    @property
    def freq(self) -> Phisical[Float[ArrayLike, 'freq']]:
        return self._first_a, self._first_u
    
    @freq.setter
    def freq(self, value: Float[ArrayLike, 'freq']):
        if value.shape[-1] != self._first_a.shape[-1]:
            raise ValueError(f"Новая ось частот должна иметь длину {self._first_a.shape[-1]}, получено {value.shape[-1]}")
        self._first_a = value

    @property
    def time(self) -> Phisical[Float[ArrayLike, 'time']]:
        return self._sec___a, self._sec___u
    
    @time.setter
    def time(self, value: Float[ArrayLike, 'time']): 
        if value.shape[-1] != self._sec___a.shape[-1]:
            raise ValueError(f"Новая ось времени должна иметь длину {self._sec___a.shape[-1]}, получено {value.shape[-1]}")
        self._sec___a = value

    @property
    def values(self) -> Phisical[Shaped[ArrayLike, '... freq time']]:
        return self._matrx_a, self._matrx_u
    
    @values.setter
    def values(self, value: Phisical[Shaped[ArrayLike, '... freq time']]):
        if value[0].shape[-2] != self._first_a.shape[-1]:
            raise ValueError(f"Размер матрицы по частоте (-2) должен быть {self._first_a.shape[-1]}")
        if value[0].shape[-1] != self._sec___a.shape[-1]:
            raise ValueError(f"Размер матрицы по времени (-1) должен быть {self._sec___a.shape[-1]}")
        self._matrx_a, self._matrx_u = value

    @property
    def df(self) -> Phisical[float]:
        return self._d_first_a

    @property
    def dt(self) -> Phisical[float]:
        return self._d_sec___a

    def _FileSystem_Save(self, path: Path):
        Function2D._FileSystem_Save(self, path)
        
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        return cls.from_Function2D(Function2D._FileSystem_Load(path))

    def integrateOverTime(self) -> FreqFunc:
        dt_value, dt_unit = self.dt
        integrated_values = ma.sum(self._matrx_a, axis=-1) * dt_value
        new_unit = unit_mul(self._matrx_u, dt_unit)
        return FreqFunc(
            values=(integrated_values, new_unit),
            axis=self._first_a
        )

    def integrateOverFreq(self) -> TimeFunc:
        df_value, df_unit = self.df
        integrated_values = ma.sum(self._matrx_a, axis=-2) * df_value
        new_unit = unit_mul(self._matrx_u, df_unit)
        return TimeFunc(
            values=(integrated_values, new_unit),
            axis=self._sec___a
        )
    
    def cloneApply(self, func: Callable[[ArrayLike], ArrayLike], new_unit: PintUnit) -> 'SpecFunc':
        return SpecFunc(
            matrix=(func(self._matrx_a), new_unit), 
            freq=self._first_a,
            time=self._sec___a
        )

    def __getitem__(self, idx: Union[int, slice]) -> 'SpecFunc':
        return SpecFunc.from_Function2D(Function2D.__getitem__(self, idx))
    
    def __iter__(self) -> Iterator['SpecFunc']:
        for i in range(len(self)):
            yield self[i]

    def to_context(self, ctx: ArrayContext) -> 'SpecFunc':
        return SpecFunc.from_Function2D(Function2D.to_context(self, ctx))


class CorrelFunc(Function2D):
    _first_u: PintUnit = UREG.second  # Задержка (Delay)
    _sec___u: PintUnit = UREG.second  # Реальное время (Time)

    def __init__(self, 
                 matrix: Phisical[Shaped[ArrayLike, '... delay time']], 
                 delay: Float[ArrayLike, 'delay'],
                 time: Float[ArrayLike, 'time']):
        if matrix[0].shape[-2] != delay.shape[-1]:
            raise ValueError(f"Размер матрицы по оси задержки ({matrix[0].shape[-2]}) не совпадает с длиной оси delay ({delay.shape[-1]})")
        if matrix[0].shape[-1] != time.shape[-1]:
            raise ValueError(f"Размер матрицы по оси времени ({matrix[0].shape[-1]}) не совпадает с длиной оси time ({time.shape[-1]})")
        
        self._matrx_a, self._matrx_u = matrix
        self._first_a = delay  
        self._sec___a = time  

    @classmethod
    def from_Function2D(cls, func: Function2D) -> Self:
        return cls(
            matrix=(func._matrx_a, func._matrx_u), 
            delay=func._first_a, 
            time=func._sec___a
        )

    @property
    def delay(self) -> Phisical[Float[ArrayLike, 'delay']]:
        return self._first_a, self._first_u
    
    @delay.setter
    def delay(self, value: Float[ArrayLike, 'delay']):
        if value.shape[-1] != self._first_a.shape[-1]:
            raise ValueError(f"Новая ось задержки должна иметь длину {self._first_a.shape[-1]}, получено {value.shape[-1]}")
        self._first_a = value

    @property
    def time(self) -> Phisical[Float[ArrayLike, 'time']]:
        return self._sec___a, self._sec___u
    
    @time.setter
    def time(self, value: Float[ArrayLike, 'time']): 
        if value.shape[-1] != self._sec___a.shape[-1]:
            raise ValueError(f"Новая ось времени должна иметь длину {self._sec___a.shape[-1]}, получено {value.shape[-1]}")
        self._sec___a = value

    @property
    def values(self) -> Phisical[Shaped[ArrayLike, '... delay time']]:
        return self._matrx_a, self._matrx_u
    
    @values.setter
    def values(self, value: Phisical[Shaped[ArrayLike, '... delay time']]):
        if value[0].shape[-2] != self._first_a.shape[-1]:
            raise ValueError(f"Размер матрицы по задержке (-2) должен быть {self._first_a.shape[-1]}")
        if value[0].shape[-1] != self._sec___a.shape[-1]:
            raise ValueError(f"Размер матрицы по времени (-1) должен быть {self._sec___a.shape[-1]}")
        self._matrx_a, self._matrx_u = value

    def __getitem__(self, idx: Union[int, slice]) -> 'CorrelFunc':
        return CorrelFunc.from_Function2D(Function2D.__getitem__(self, idx))
    
    def __iter__(self) -> Iterator['CorrelFunc']:
        for i in range(len(self)):
            yield self[i]

    def to_context(self, ctx: ArrayContext) -> 'CorrelFunc':
        return CorrelFunc.from_Function2D(Function2D.to_context(self, ctx))
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/__init__.py" encoding="utf-8"
import math
import threading
from contextlib import contextmanager
from typing import Tuple, Union, Any
import warnings

# --- Внедряем наш новый универсальный фреймворк ---
import MultiArray as ma
import MultiArray.FFT as mfft

from BatSpec.Core.Functions import TimeFunc, SpecFunc, CorrelFunc
from BatSpec.Core.ConvWindow import Window, WindowNorm, WindowNormMismatchError
from BatSpec.Core.Physical.Units import UREG, unit_devide, unit_sqrt, unit_mul


# =====================================================================
# КОНТЕКСТНЫЙ МЕНЕДЖЕР ВЫЧИСЛЕНИЙ
# =====================================================================

_dsp_tls = threading.local()

@contextmanager
def DSPContext(ctx: ma.ArrayContext):
    """
    Контекстный менеджер для переопределения вычислительного бэкенда (фреймворка и устройства).
    """
    old_ctx = getattr(_dsp_tls, 'math_ctx', None)
    _dsp_tls.math_ctx = ctx
    try:
        yield
    finally:
        _dsp_tls.math_ctx = old_ctx

def _get_math_ctx(default_ctx: ma.ArrayContext) -> ma.ArrayContext:
    override = getattr(_dsp_tls, 'math_ctx', None)
    return override if override is not None else default_ctx


def _get_real_dtype(tensor: Any, ctx: ma.ArrayContext) -> Any:
    """Безопасно извлекает вещественный тип данных."""
    if ctx.isTensorflow():
        return tensor.dtype.real_dtype if hasattr(tensor.dtype, 'real_dtype') else tensor.dtype
    elif ctx.isTorch():
        return tensor.real.dtype if tensor.is_complex() else tensor.dtype
    else:
        return tensor.real.dtype if hasattr(tensor, 'real') else getattr(tensor, 'dtype', None)

# =====================================================================
# БЛОК 1: Сложные DSP функции (STFT, IFFT, Correlogram)
# =====================================================================

def makeComplexSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)

    orig_ctx = signal.context
    math_ctx = _get_math_ctx(orig_ctx)

    signal_math = signal.to_context(math_ctx)
    signal_a, signal_unit = signal_math.values
    sr = signal_math.sr

    real_dtype = _get_real_dtype(signal_a, math_ctx)
    real_ctx = ma.ArrayContext(math_ctx._framework, math_ctx._device, real_dtype)
    
    orig_win_length = int(window.time * sr)
    fft_length = (bins - 1) * 2
    hop = max(1, int(orig_win_length * (1 - overlap)))

    win = window.get_array(sr, real_ctx)

    # ====================================================================
    # ЭКСТРЕМАЛЬНЫЙ DSP: Direct DTFT
    # ====================================================================
    if fft_length < orig_win_length:
        warnings.warn(f"fft_length ({fft_length}) < win_length ({orig_win_length}). "
                      "Используется точный матричный расчет DTFT (Без обрезки окна!).")
        
        pad_len = 0
        sig_len = signal_a.shape[-1]
        if sig_len < orig_win_length:
            pad_len = orig_win_length - sig_len
        else:
            rem = (sig_len - orig_win_length) % hop
            pad_len = (hop - rem) % hop if rem != 0 else 0
            
        if pad_len > 0:
            if math_ctx.isTensorflow():
                paddings = [[0, 0]] * (len(signal_a.shape) - 1) + [[0, pad_len]]
                signal_a = math_ctx.fw.pad(signal_a, paddings)
            elif math_ctx.isTorch():
                import torch.nn.functional as F
                signal_a = F.pad(signal_a, (0, pad_len))
            else:
                import numpy as np
                sig_np = ma.to_numpy(signal_a)
                pad_tuple = [(0, 0)] * (sig_np.ndim - 1) + [(0, pad_len)]
                signal_a = ma.convert_to(np.pad(sig_np, pad_tuple), math_ctx)

        if math_ctx.isTorch():
            frames = signal_a.unfold(-1, orig_win_length, hop)
        elif math_ctx.isTensorflow():
            frames = math_ctx.fw.signal.frame(signal_a, orig_win_length, hop)
        else:
            import numpy as np
            from numpy.lib.stride_tricks import sliding_window_view
            frames_np = sliding_window_view(ma.to_numpy(signal_a), orig_win_length, axis=-1)
            frames_np = frames_np[..., ::hop, :]
            frames = ma.convert_to(frames_np, math_ctx)
            
        n_vec = ma.reshape(ma.arange(orig_win_length, real_ctx), (orig_win_length, 1))
        k_vec = ma.reshape(ma.arange(bins, real_ctx), (1, bins))
        
        phase = (-2.0 * math.pi) * (n_vec * k_vec) / float(fft_length)
        
        win_col = ma.reshape(win, (orig_win_length, 1))
        E_real = win_col * ma.cos(phase)
        E_imag = win_col * ma.sin(phase)
        
        real_part = ma.matmul(frames, E_real)
        imag_part = ma.matmul(frames, E_imag)
        
        if math_ctx.isTorch():
            import torch
            stft = torch.complex(real_part, imag_part).transpose(-2, -1)
        elif math_ctx.isTensorflow():
            import tensorflow as tf
            stft = tf.linalg.matrix_transpose(tf.complex(real_part, imag_part))
        else:
            import numpy as np
            stft_np = ma.to_numpy(real_part) + 1j * ma.to_numpy(imag_part)
            stft = ma.convert_to(np.swapaxes(stft_np, -2, -1), math_ctx)
            
    else:
        # Стандартный путь: Обычный FFT
        # ИСПРАВЛЕНИЕ 1: pad_end=False для точного совпадения формы матриц с Legacy TF
        stft = mfft.stft(signal_a, frame_length=orig_win_length, frame_step=hop, fft_length=fft_length, window=win, pad_end=False)

    scale_factor = math.sqrt(2.0 / sr)
    num_frames = stft.shape[-1]
    
    freq_axis = mfft.rfftfreq(fft_length, d=1.0 / sr, ctx=real_ctx)
    time_axis = ma.arange(num_frames, real_ctx) * (hop / sr)
    spec_unit = unit_devide(signal_unit, unit_sqrt(UREG.Hz))

    spec_math = SpecFunc(matrix=(stft * scale_factor, spec_unit), freq=freq_axis, time=time_axis)
    
    # ИСПРАВЛЕНИЕ 2: Возвращаем данные в исходный фреймворк, но СОХРАНЯЕМ КОМПЛЕКСНЫЙ ТИП!
    # Если мы отдадим orig_ctx (у которого тип float32), мнимая часть обрежется.
    out_dtype = getattr(stft, 'dtype', None)
    return_ctx = ma.ArrayContext(orig_ctx._framework, orig_ctx._device, out_dtype)
    
    return spec_math.to_context(return_ctx)


def inverseComplexSpec(spec: SpecFunc, window: Window, overlap: float = 0.5) -> TimeFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)
    
    orig_ctx = spec.context
    math_ctx = _get_math_ctx(orig_ctx)

    spec_math = spec.to_context(math_ctx)
    complex_matrix, spec_unit = spec_math.values
    
    real_dtype = _get_real_dtype(complex_matrix, math_ctx)
    real_ctx = ma.ArrayContext(math_ctx._framework, math_ctx._device, real_dtype) 
    
    freq_axis, _ = spec_math.freq

    fft_length = (len(freq_axis) - 1) * 2
    sr_f_val, _ = spec_math.df 
    sr = int(round(sr_f_val * fft_length))
    
    orig_win_length = int(window.time * sr)
    hop = max(1, int(orig_win_length * (1 - overlap)))
    
    win = window.get_array(sr, real_ctx)

    if fft_length < orig_win_length:
        start = (orig_win_length - fft_length) // 2
        win = win[start : start + fft_length]
        win_length = fft_length
    else:
        win_length = orig_win_length

    scale_factor = math.sqrt(2.0 / sr)
    unscaled = complex_matrix / scale_factor

    signal_a = mfft.istft(unscaled, frame_length=win_length, frame_step=hop, fft_length=fft_length, window=win)
    
    if hasattr(signal_a, 'real') and not math_ctx.isTorch():
        signal_a = signal_a.real
    elif math_ctx.isTorch() and signal_a.is_complex():
        signal_a = signal_a.real
    
    new_time_axis = ma.arange(signal_a.shape[-1], real_ctx) / float(sr)
    signal_unit = unit_mul(spec_unit, unit_sqrt(UREG.Hz))

    time_math = TimeFunc(values=(signal_a, signal_unit), axis=new_time_axis)
    
    return time_math.to_context(orig_ctx)


def makeSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    orig_ctx = signal.context
    math_ctx = _get_math_ctx(orig_ctx)

    signal_math = signal.to_context(math_ctx)
    
    # 1. Получаем КОМПЛЕКСНУЮ спектрограмму
    complex_spec_math = makeComplexSpec(signal_math, window, overlap, bins)
    
    # 2. Берем модуль (абсолютное значение: sqrt(Re^2 + Im^2))
    amp_matrix = ma.abs(complex_spec_math.values[0])
    
    spec_math = SpecFunc(
        matrix=(amp_matrix, complex_spec_math.values[1]), 
        freq=complex_spec_math.freq[0], 
        time=complex_spec_math.time[0]
    )
    
    # 3. Возвращаем в исходный фреймворк, сохраняя вещественный тип амплитуды
    out_dtype = getattr(amp_matrix, 'dtype', None)
    return_ctx = ma.ArrayContext(orig_ctx._framework, orig_ctx._device, out_dtype)
    
    return spec_math.to_context(return_ctx)


def makeRobustSpec(
    signal: TimeFunc,
    window,  # одно окно или тюпл окон
    overlap: float = 0.5,
    bins: int = 300,
    shifts: int | tuple[int, ...] = 4
) -> SpecFunc:
    """
    Вычисляет спектрограмму, устойчивую к сингулярностям.
    """
    windows = window if isinstance(window, tuple) else (window,)

    if isinstance(shifts, int):
        shift_list = [s for s in range(-shifts, shifts + 1) if s != 0]
    else:
        shift_list = [s for s in shifts if s != 0]

    sig_a, sig_u = signal.values

    def apply_shift(shift: int) -> TimeFunc:
        ctx = signal.context
        shape_pad = list(sig_a.shape)
        shape_pad[-1] = abs(shift)
        pad = ma.zeros(tuple(shape_pad), ctx)
        if shift > 0:
            shifted_a = ma.concatenate([pad, sig_a[..., :-shift]], axis=-1)
        else:
            shifted_a = ma.concatenate([sig_a[..., -shift:], pad], axis=-1)
        return TimeFunc(values=(shifted_a, sig_u), axis=signal.time[0])

    def elem_max(a, b, context):
        if context.isTorch():
            return context.fw.maximum(a, b)
        elif context.isTensorflow():
            import tensorflow as tf
            return tf.maximum(a, b)
        else:
            import numpy as np
            return np.maximum(a, b)

    base_spec = makeSpec(signal, windows[0], overlap, bins)
    mat_max, unit = base_spec.values
    ctx = base_spec.context

    for win in windows:
        if win is not windows[0]:
            s, _ = makeSpec(signal, win, overlap, bins).values
            mat_max = elem_max(mat_max, s, ctx)

        for shift in shift_list:
            s, _ = makeSpec(apply_shift(shift), win, overlap, bins).values
            mat_max = elem_max(mat_max, s, ctx)

    return SpecFunc(matrix=(mat_max, unit), freq=base_spec.freq[0], time=base_spec.time[0])


def makeCorrelogram(spec: SpecFunc, max_delay_ms: float = 10.0, blur_sigma: Union[float, Tuple[float, float]] = 0.0) -> CorrelFunc:
    """
    Создает коррелограмму на основе спектрограммы. 
    Рассчитывает MSE по оси частот между исходной спектрограммой и средним 
    двух смещенных спектрограмм (+dt и -dt).
    """
    do_blur = False
    if isinstance(blur_sigma, tuple):
        if blur_sigma[0] > 0 or blur_sigma[1] > 0:
            do_blur = True
    elif blur_sigma > 0:
        do_blur = True
        
    if do_blur:
        spec = blurSpec(spec, blur_sigma)
        
    mat_a, mat_u = spec.values
    ctx = spec.context
    
    mat_a = ma.abs(mat_a)

    dt_val, _ = spec.dt  
    max_delay_sec = float(max_delay_ms) / 1000.0
    
    max_delay_bins = int(max_delay_sec / dt_val) + 1
    max_delay_bins = min(max_delay_bins, mat_a.shape[-1])
    if max_delay_bins < 1:
        max_delay_bins = 1

    def apply_shift(mat: Any, shift: int) -> Any:
        if shift == 0:
            return mat
        shape_pad = list(mat.shape)
        shape_pad[-1] = abs(shift)
        pad = ma.zeros(tuple(shape_pad), ctx)
        if shift > 0:
            return ma.concatenate([pad, mat[..., :-shift]], axis=-1)
        else:
            return ma.concatenate([mat[..., -shift:], pad], axis=-1)

    results = []
    
    for k in range(max_delay_bins):
        if k == 0:
            diff = ma.zeros_like(mat_a)
        else:
            s_plus = apply_shift(mat_a, k)
            s_minus = apply_shift(mat_a, -k)
            mean_s = (s_plus + s_minus) / 2.0
            diff = mean_s - mat_a
            
        sq_diff = diff ** 2
        mse = ma.mean(sq_diff, axis=-2)
        
        new_shape = mse.shape[:-1] + (1, mse.shape[-1])
        mse_reshaped = ma.reshape(mse, new_shape)
        results.append(mse_reshaped)

    out_mat = ma.concatenate(results, axis=-2)
    
    real_dtype = _get_real_dtype(mat_a, ctx)
    real_ctx = ma.ArrayContext(ctx._framework, ctx._device, real_dtype)
    delay_axis = ma.arange(max_delay_bins, real_ctx) * float(dt_val)
    
    out_unit = unit_mul(mat_u, mat_u)

    return CorrelFunc(matrix=(out_mat, out_unit), delay=delay_axis, time=spec.time[0])


# =====================================================================
# БЛОК 2: Чистая математика
# =====================================================================

REF_PA_ASD = 2e-5  
REF_FS_ASD = 1.0   
REF_V_ASD  = 1.0   

def makeLogDB(spec: SpecFunc, add_one: bool = False) -> SpecFunc:
    _, old_unit = spec.values

    if old_unit.is_compatible_with(UREG.Pa / (UREG.Hz ** 0.5)): ref_val = REF_PA_ASD
    elif old_unit.is_compatible_with(UREG.FS / (UREG.Hz ** 0.5)): ref_val = REF_FS_ASD
    elif old_unit.is_compatible_with(UREG.V / (UREG.Hz ** 0.5)): ref_val = REF_V_ASD
    else: ref_val = 1.0

    def to_db(arr: Any) -> Any:
        amplitude = ma.abs(arr)
        ratio = amplitude / ref_val
        
        if add_one:
            return 20 * ma.log10(ratio + 1.0)
        else:
            scaled_ratio = ma.clamp_min(ratio, min_val=1e-9)
            return 20 * ma.log10(scaled_ratio)

    return spec.cloneApply(func=to_db, new_unit=UREG.dB)


# =====================================================================
# БЛОК 3: Интерполяция и Свертки
# =====================================================================

def interpolate(spec: SpecFunc, new_freq: Any, new_time: Any) -> 'SpecFunc':
    ctx = spec.context
    mat_a, mat_u = spec.values
    old_freq, _ = spec.freq
    old_time, _ = spec.time
    
    real_dtype = _get_real_dtype(mat_a, ctx)
    real_ctx = ma.ArrayContext(ctx._framework, ctx._device, real_dtype)
    
    new_freq_t = ma.convert_to(new_freq, real_ctx)
    new_time_t = ma.convert_to(new_time, real_ctx)

    out_mat = mfft.interpolate_2d(mat_a, old_time, old_freq, new_time_t, new_freq_t)

    return SpecFunc(matrix=(out_mat, mat_u), freq=new_freq_t, time=new_time_t)


def interpolateToShape(spec: 'SpecFunc', target_shape: Tuple[int, int]) -> 'SpecFunc':
    old_freq, _ = spec.freq
    old_time, _ = spec.time
    ctx = spec.context
    
    real_dtype = _get_real_dtype(spec.values[0], ctx)
    real_ctx = ma.ArrayContext(ctx._framework, ctx._device, real_dtype)
    
    new_freq = ma.linspace(float(old_freq[0]), float(old_freq[-1]), target_shape[0], real_ctx)
    new_time = ma.linspace(float(old_time[0]), float(old_time[-1]), target_shape[1], real_ctx)

    return interpolate(spec, new_freq, new_time)


def interpolateByFactors(spec: 'SpecFunc', factors: Tuple[float, float]) -> 'SpecFunc':
    old_freq, _ = spec.freq
    old_time, _ = spec.time

    new_nfreq = max(1, int(round(old_freq.shape[0] * factors[0])))
    new_ntime = max(1, int(round(old_time.shape[0] * factors[1])))

    return interpolateToShape(spec, target_shape=(new_nfreq, new_ntime))


def blurSpec(spec: 'SpecFunc', sigma: Union[float, Tuple[float, float]] = 1.0) -> 'SpecFunc':
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))

    kernel_size_f = int(round(sigma[0] * 3)) * 2 + 1
    kernel_size_t = int(round(sigma[1] * 3)) * 2 + 1
    kernel_size = (kernel_size_f, kernel_size_t)

    _, current_unit = spec.values

    def apply_gaussian_blur_fft(mat: Any) -> Any:
        ctx_orig = ma.ArrayContext.from_array(mat)
        
        import torch
        dev_enum = ma.DeviceType.GPU if ctx_orig.isGPU() else ma.DeviceType.CPU
        ctx_torch = ma.ArrayContext(ma.Framework.TORCH, dev_enum, None)
        
        mat_torch = ma.convert_to(mat, ctx_torch)
        device = mat_torch.device
        dtype = mat_torch.real.dtype if mat_torch.is_complex() else mat_torch.dtype
        img_h, img_w = mat_torch.shape[-2], mat_torch.shape[-1]
        
        def get_1d_kernel(k: int, s: float) -> torch.Tensor:
            limit = (k - 1) / 2.0
            x = torch.linspace(-limit, limit, steps=k, device=device, dtype=dtype)
            gauss = torch.exp(-0.5 * (x / s).pow(2))
            return gauss / gauss.sum()

        kernel_y = get_1d_kernel(kernel_size[0], sigma[0])
        kernel_x = get_1d_kernel(kernel_size[1], sigma[1])
        kernel_2d_small = (kernel_y.unsqueeze(-1) * kernel_x.unsqueeze(0))
        
        padded_kernel = torch.zeros(img_h, img_w, device=device, dtype=dtype)
        k_h, k_w = kernel_2d_small.shape
        padded_kernel[:k_h, :k_w] = kernel_2d_small
        
        padded_kernel = torch.roll(padded_kernel, shifts=(-k_h // 2, -k_w // 2), dims=(-2, -1))
        
        fft_mat = torch.fft.fft2(mat_torch, dim=(-2, -1))
        fft_kernel = torch.fft.fft2(padded_kernel, dim=(-2, -1))
        fft_result = fft_mat * fft_kernel
        ifft_result = torch.fft.ifft2(fft_result, dim=(-2, -1))

        res = ifft_result if mat_torch.is_complex() else ifft_result.real
        
        return ma.convert_to(res, ctx_orig)

    return spec.cloneApply(func=apply_gaussian_blur_fft, new_unit=current_unit)
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/__test__/old_method.py" encoding="utf-8"
from pathlib import Path
import numpy as np
from typing import cast


ScriptDir = Path(__file__).parent

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW

# --- Импорты для Legacy Функции ---
import tensorflow as tf
from BatSpec.Core.ConvWindow import Window, WindowNorm, WindowNormMismatchError
from BatSpec.Core.Physical.Units import UREG, unit_devide, unit_sqrt

# --- Основные импорты ---
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.Functions import SpecFunc, TimeFunc
from BatSpec.Core.Spectral import makeSpec, makeLogDB, DSPContext
from BatSpec.Visualize import update_spec2d, run_visualizer

# --- Импорты MultiArray ---
from MultiArray.Core import ArrayContext, Framework, DeviceType
import MultiArray as ma

# =====================================================================
# LEGACY ФУНКЦИЯ (Встроена для сравнения)
# =====================================================================

def makeSpec_legacy(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    """Амплитудная спектрограмма (|STFT| * scale) - СТАРАЯ ВЕРСИЯ НА TF."""
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)

    # Убеждаемся, что на входе NumPy массив
    signal_a, signal_unit = ma.to_numpy(signal.values[0]), signal.values[1]
    sr = signal.sr

    win = ma.to_numpy(window.get_array(sr, ArrayContext(Framework.NUMPY, DeviceType.CPU, None)))
    fft_length = (bins - 1) * 2
    win_length = len(win)
    hop = max(1, int(win_length * (1 - overlap)))

    def win_func(length: int, dtype: tf.DType = tf.float32):
        assert length == win_length
        return tf.convert_to_tensor(win, dtype=dtype)

    stft = cast(tf.Tensor, tf.signal.stft(
        signal_a.astype(np.float32),
        frame_length=win_length,
        frame_step=hop,
        fft_length=fft_length,
        window_fn=win_func,
    ))

    # tf.signal.stft возвращает (Time, Freq)
    amplitude_matrix_tf_shape = np.abs(stft.numpy())
    
    # SpecFunc ожидает (Freq, Time), поэтому ТРАНСПОНИРУЕМ
    amplitude_matrix = amplitude_matrix_tf_shape.T

    scale = np.sqrt(2 / sr)
    num_frames: int = amplitude_matrix.shape[1]
    freq_axis = np.fft.rfftfreq(fft_length, d=1.0 / sr)
    time_axis = (np.arange(num_frames) * hop) / sr

    spec_unit = unit_devide(signal_unit, unit_sqrt(UREG.Hz))

    return SpecFunc(
        matrix=(amplitude_matrix * scale, spec_unit),
        time=time_axis,
        freq=freq_axis,
    )

# =====================================================================
# ОСНОВНОЙ СКРИПТ
# =====================================================================

@run_visualizer
def main():
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    print(record.context)

    ctx_numpy_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    ctx_torch_gpu = ArrayContext(Framework.TENSORFLOW, DeviceType.GPU, None)
    
    window = TEST_HANN_WINODW
    overlap = 0.8
    bins = 300

    print("--- 1. Расчет спектрограммы новым методом (MultiArray) ---")
    with DSPContext(ctx_torch_gpu):
        spec_new = makeSpec(
            record,
            window=window,
            overlap=overlap,
            bins=bins,
        ).to_context(ctx_numpy_cpu)
    update_spec2d("1. MultiArray Spec", makeLogDB(spec_new, add_one=False))

    print("--- 2. Расчет спектрограммы старым методом (Legacy TF) ---")
    spec_legacy = makeSpec_legacy(
        record,
        window=window,
        overlap=overlap,
        bins=bins,
    )
    update_spec2d("2. Legacy TF Spec", makeLogDB(spec_legacy, add_one=False))

    print("--- 3. Сравнение результатов ---")
    matrix_new, _ = spec_new.values
    matrix_legacy, _ = spec_legacy.values
    
    assert matrix_new.shape == matrix_legacy.shape, \
        f"Формы матриц не совпадают! New: {matrix_new.shape}, Legacy: {matrix_legacy.shape}"

    diff_matrix = np.abs(matrix_new - matrix_legacy)
    mae = np.mean(diff_matrix)
    print(f"Средняя абсолютная ошибка (MAE): {mae:.12f}")
    
    # Визуализируем разницу. Если всё черное, значит, результаты идентичны.
    diff_spec = SpecFunc(
        matrix=(diff_matrix, spec_new.values[1]),
        time=spec_new.time[0],
        freq=spec_new.freq[0]
    )
    update_spec2d("3. Difference (Abs)", diff_spec)
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/__test__/2.py" encoding="utf-8"
from pathlib import Path
import time
import numpy as np

ScriptDir = Path(__file__).parent

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW, TEST_BLHA_WINODW
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.SaveIntegral import SaveIntegral
from BatSpec.Core.Spectral import makeSpec, makeLogDB, DSPContext, makeRobustSpec
from BatSpec.Core.Functions import TimeFunc, SpecFunc
from BatSpec.Core.Physical.Units import UREG
from BatSpec.Visualize import update_spec2d, run_visualizer
from MultiArray.Core import ArrayContext, Framework, DeviceType
# --- Импорты MultiArray ---

@run_visualizer
def main():
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    
    
    ctx_torch_gpu = ArrayContext(Framework.TORCH, DeviceType.GPU, None)
    
    print("--- Основные вычисления ---")
    
    # 1. Оригинальная (классическая) спектрограмма
    t0 = time.time()
    with DSPContext(ctx_torch_gpu):
        spec_standard = makeSpec(record, TEST_HANN_WINODW, overlap=0.8, bins=300)
    print(f"Standard STFT: {time.time() - t0:.3f} сек")
    update_spec2d("1. Standard (С дырками)", makeLogDB(spec_standard, add_one=False))

    # 2. Робастная спектрограмма (Jitter Ensemble)
    t0 = time.time()
    with DSPContext(ctx_torch_gpu):
        spec_robust = makeRobustSpec(
            record,
            window=(TEST_HANN_WINODW, TEST_BLHA_WINODW),
            overlap=0.8,
            bins=300,
            shifts=(-1, 1)  
        )
    print(f"Robust STFT (x6 computations): {time.time() - t0:.3f} сек")
    update_spec2d("2. Robust (Без дырок!)", makeLogDB(spec_robust, add_one=False))
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/__test__/1.py" encoding="utf-8"
from pathlib import Path
ScriptDir = Path(__file__).parent
import time
import numpy as np

from BatSpec.Core.ConvWindow import TEST_HANN_WINODW
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.SaveIntegral import SaveIntegral
from BatSpec.Core.Spectral import makeSpec, makeLogDB, DSPContext
from BatSpec.Core.Functions import TimeFunc, SpecFunc
from BatSpec.Core.Physical.Units import UREG
from BatSpec.Visualize import update_function, update_spec2d, run_visualizer

from MultiArray.Core import ArrayContext, Framework, DeviceType


def warmup_gpu():
    """Функция для инициализации драйверов CUDA/XLA перед замерами времени."""
    print("--- Прогрев GPU (Инициализация CUDA) ---")
    
    # Генерируем 1 секунду шума
    dummy_sr = 48000
    dummy_time = np.linspace(0, 1.0, dummy_sr)
    dummy_vals = np.random.randn(dummy_sr).astype(np.float32)
    dummy_sig = TimeFunc(values=(dummy_vals, UREG.FS), axis=dummy_time)
    
    ctx_tf = ArrayContext(Framework.TENSORFLOW, DeviceType.GPU, None)
    ctx_torch = ArrayContext(Framework.TORCH, DeviceType.GPU, None)
    
    try:
        t0 = time.time()
        with DSPContext(ctx_torch):
            makeSpec(dummy_sig, TEST_HANN_WINODW, overlap=0.5, bins=100)
        print(f"Torch GPU прогрет за {time.time() - t0:.3f} сек")
    except Exception as e:
        print("Torch недоступен или ошибка прогрева:", e)
        
    try:
        t0 = time.time()
        with DSPContext(ctx_tf):
            makeSpec(dummy_sig, TEST_HANN_WINODW, overlap=0.5, bins=100)
        print(f"TF GPU прогрет за {time.time() - t0:.3f} сек")
    except Exception as e:
        print("TF недоступен или ошибка прогрева:", e)
        
    print("----------------------------------------\n")


@run_visualizer
def main():
    # 1. Прогрев бэкендов
    warmup_gpu()
    
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    
    # Контексты
    ctx_np_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    ctx_tf_gpu = ArrayContext(Framework.TENSORFLOW, DeviceType.GPU, None)
    ctx_torch_gpu = ArrayContext(Framework.TORCH, DeviceType.GPU, None)
    
    print("--- Основные вычисления ---")
    
    # 1. NumPy
    t0 = time.time()
    with DSPContext(ctx_np_cpu):
        spec_np = makeSpec(record, TEST_HANN_WINODW, overlap=0.8, bins=300)
    print(f"NumPy STFT: {time.time() - t0:.3f} сек")
    update_spec2d("1. NumPy", makeLogDB(spec_np, add_one=False))
    
    # 2. PyTorch GPU
    t0 = time.time()
    with DSPContext(ctx_torch_gpu):
        spec_torch = makeSpec(record, TEST_HANN_WINODW, overlap=0.8, bins=300)
    print(f"Torch GPU STFT: {time.time() - t0:.3f} сек")
    update_spec2d("2. PyTorch", makeLogDB(spec_torch, add_one=False))

    # 3. TensorFlow GPU
    t0 = time.time()
    with DSPContext(ctx_tf_gpu):
        spec_tf = makeSpec(record, TEST_HANN_WINODW, overlap=0.8, bins=300)
    print(f"TF GPU STFT: {time.time() - t0:.3f} сек")
    update_spec2d("3. TensorFlow", makeLogDB(spec_tf, add_one=False))

    # 4. Сборка гибридной спектрограммы
    # Так как makeSpec возвращает объекты в исходном контексте (тут это NumPy),
    # мы можем спокойно использовать классический np.maximum!
    mat_tf, u = spec_tf.values
    mat_torch, _ = spec_torch.values

    print(f'''
DEBUG SHAPE: spec_tf.values[0].shape = {spec_tf.values[0].shape}
DEBUG SHAPE: spec_tf.time[0].shape = {spec_tf.time[0].shape}
DEBUG SHAPE: spec_tf.freq[0].shape = {spec_tf.freq[0].shape}

DEBUG SHAPE: spec_torch.values[0].shape = {spec_torch.values[0].shape}
DEBUG SHAPE: spec_torch.time[0].shape = {spec_torch.time[0].shape}
DEBUG SHAPE: spec_torch.freq[0].shape = {spec_torch.freq[0].shape}

DEBUG SHAPE: spec_np.values[0].shape = {spec_np.values[0].shape}
DEBUG SHAPE: spec_np.time[0].shape = {spec_np.time[0].shape}
DEBUG SHAPE: spec_np.freq[0].shape = {spec_np.freq[0].shape}
''')
    
    mat_max = np.maximum(mat_tf, mat_torch)
    
    spec_hybrid = SpecFunc(
        matrix=(mat_max, u), 
        freq=spec_tf.freq[0], 
        time=spec_tf.time[0]
    )
    
    update_spec2d("4. Hybrid (TF max Torch)", makeLogDB(spec_hybrid, add_one=False))
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Record/__init__.py" encoding="utf-8"
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

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Record/Calibration.py" encoding="utf-8"
from jaxtyping import Float, ArrayLike

# --- 1. Импорты специфичных для проекта библиотек ---
from BatSpec.Core.Physical.Units import UREG, unit_mul, unit_devide
from BatSpec.Core.Functions import TimeFunc, FreqFunc, SpecFunc
from BatSpec.Core.Spectral import makeComplexSpec, inverseComplexSpec
from BatSpec.Core.ConvWindow import Window, TEST_HANN_WINODW

# --- 2. Внедряем наш новый универсальный фреймворк ---
import MultiArray as ma

# =====================================================================
# 1. Аналитические модели приборов (теперь с MultiArray)
# =====================================================================

class CalibrationModel:
    """Базовый класс для аналитической модели АЧХ прибора."""
    
    def __call__(self, freq_hz: Float[ArrayLike, 'freq']) -> Float[ArrayLike, 'freq']:
        raise NotImplementedError

    def generate_calibration_curve(self, freq_axis: Float[ArrayLike, 'freq']) -> FreqFunc:
        """Создает калибровочную кривую в том же фреймворке, что и входная ось частот."""
        coef_values = self(freq_axis)
        unit_pa_per_fs = unit_devide(UREG.Pa, UREG.FS)
        return FreqFunc(values=(coef_values, unit_pa_per_fs), axis=freq_axis)

        
class FlatResponseModel(CalibrationModel):
    def __init__(self, sensitivity_pa: float = 20.0):
        self.sensitivity_pa = sensitivity_pa

    def __call__(self, freq_hz: Float[ArrayLike, 'freq']) -> Float[ArrayLike, 'freq']:
        ctx = ma.ArrayContext.from_array(freq_hz)
        return ma.full(freq_hz.shape, self.sensitivity_pa, ctx)


class PetterssonM500Model(CalibrationModel):
    def __init__(self, base_pa_per_fs: float = 15.0, rolloff_start_hz: float = 20_000.0,
                 rolloff_rate: float = 5.0 / 10_000.0):
        self.base_pa_per_fs   = base_pa_per_fs
        self.rolloff_start_hz = rolloff_start_hz
        self.rolloff_rate     = rolloff_rate

    def __call__(self, freq_hz: Float[ArrayLike, 'freq']) -> Float[ArrayLike, 'freq']:
        boost = ma.clamp_min(freq_hz - self.rolloff_start_hz, 0.0) * self.rolloff_rate
        return self.base_pa_per_fs + boost


class ResonanceMicModel(CalibrationModel):
    def __init__(self, base_pa: float = 25.0, resonance_hz: float = 40_000.0,
                 resonance_width_hz: float = 5_000.0, resonance_depth_pa: float = 15.0):
        self.base_pa            = base_pa
        self.resonance_hz       = resonance_hz
        self.resonance_width_hz = resonance_width_hz
        self.resonance_depth_pa = resonance_depth_pa

    def __call__(self, freq_hz: Float[ArrayLike, 'freq']) -> Float[ArrayLike, 'freq']:
        exponent   = -0.5 * ((freq_hz - self.resonance_hz) / self.resonance_width_hz) ** 2
        resonance  = self.resonance_depth_pa * ma.exp(exponent)
        return self.base_pa - resonance


# =====================================================================
# 2. Главная функция: TimeFunc(FS) -> TimeFunc(Pa)
# =====================================================================

def applyСalibration(signal_fs: TimeFunc, 
                     model: CalibrationModel, 
                     window: Window = TEST_HANN_WINODW, 
                     overlap: float = 0.5, 
                     bins: int = 300) -> TimeFunc:
    """
    Переводит сырой цифровой сигнал (FS) в физические Паскали (Pa),
    учитывая частотно-зависимую калибровку (АЧХ) прибора.
    Работает с NumPy, PyTorch, TensorFlow и другими фреймворками.
    """
    _, signal_u = signal_fs.values
    if not signal_u.is_compatible_with(UREG.FS):
        raise ValueError(f"Ожидался сигнал в единицах FS, получено: {signal_u}")

    complex_spec = makeComplexSpec(signal_fs, window=window, overlap=overlap, bins=bins)
    freq_axis, _ = complex_spec.freq
    calib_curve  = model.generate_calibration_curve(freq_axis)

    spec_matrix, spec_u = complex_spec.values  # Форма: (... freq time)
    coef_vector, coef_u = calib_curve.values   # Форма: (freq)

    # Добавляем измерение для broadcast'инга с помощью ma.reshape: (freq) -> (freq, 1)
    new_shape = coef_vector.shape + (1,)
    coef_vector_broadcasted = ma.reshape(coef_vector, new_shape)
    
    # Broadcasting (умножение (... freq time) на (freq, 1)) работает одинаково во всех фреймворках
    calibrated_matrix = spec_matrix * coef_vector_broadcasted
    new_spec_unit = unit_mul(spec_u, coef_u)

    time_axis, _ = complex_spec.time
    calibrated_spec = SpecFunc(
        matrix=(calibrated_matrix, new_spec_unit),
        freq=freq_axis,
        time=time_axis,
    )

    return inverseComplexSpec(calibrated_spec, window, overlap=overlap)

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Visualize/spec2d_visualizer.py" encoding="utf-8"
import sys
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QPointF

# --- Внедряем наш новый универсальный фреймворк ---
import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType

# --- Твои импорты ---
from BatSpec.QtUp.Reactive.reactive_dict import ReactiveDict
from BatSpec.QtUp.Selector import Selector
from BatSpec.QtUp.Image import AdaptiveImageItem

from BatSpec.Core.Functions import Function2D, SpecFunc, CorrelFunc
from BatSpec.Core.Physical.Units import UREG


# =====================================================================
# 1. ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ ДАННЫХ И API
# =====================================================================

plot2d_store: ReactiveDict[str, Function2D] = ReactiveDict()

def update_spec2d(name: str, func_obj: Function2D) -> None:
    plot2d_store[name] = func_obj

def remove_spec2d(name: str) -> None:
    if name in plot2d_store:
        del plot2d_store[name]

# =====================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ
# =====================================================================

class Syncer(QtCore.QObject):
    """Синхронизирует регион на Minimap и область видимости MainGraph."""
    def __init__(self, minimap_region: pg.LinearRegionItem, graph_view: pg.ViewBox, parent=None):
        super().__init__(parent)
        self._region = minimap_region
        self._view = graph_view
        self._updating = False

        self._region.sigRegionChanged.connect(self._sync_main_from_mini)
        self._view.sigRangeChanged.connect(self._sync_mini_from_main)

    def _sync_main_from_mini(self):
        if self._updating: return
        self._updating = True
        self._view.setXRange(*self._region.getRegion(), padding=0)
        self._updating = False

    def _sync_mini_from_main(self):
        if self._updating: return
        self._updating = True
        self._region.setRegion(self._view.viewRange()[0])
        self._updating = False

class HoverTracker(QtWidgets.QLabel):
    """Отображает координаты и значение функции 2D под курсором."""
    def __init__(self, view_box: pg.ViewBox, parent=None):
        super().__init__("No data", parent)
        self.setMinimumWidth(350)
        
        self._view_box = view_box
        self._matrix: np.ndarray | None = None   
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 0.0
        self._unit = ""
        self._x_label = "X"
        self._y_label = "Y"

        scene = self._view_box.scene()
        if scene is not None:
            scene.sigMouseMoved.connect(self._on_mouse_moved)

    def set_data(self, matrix_yx: np.ndarray, x0: float, x1: float, y0: float, y1: float, 
                 unit: str, x_label: str, y_label: str):
        self._matrix = matrix_yx 
        self._x_min, self._x_max = x0, x1
        self._y_min, self._y_max = y0, y1
        self._unit = unit
        self._x_label = x_label
        self._y_label = y_label
        self.setText("Ready")

    def clear_data(self):
        self._matrix = None
        self.setText("No data")

    @QtCore.Slot(QPointF)
    def _on_mouse_moved(self, scene_pos: QPointF):
        if self._matrix is None:
            return

        if not self._view_box.sceneBoundingRect().contains(scene_pos):
            self.setText("Outside graph area")
            return

        view_pos = self._view_box.mapSceneToView(scene_pos)
        x, y = view_pos.x(), view_pos.y()

        if not (self._x_min <= x <= self._x_max and self._y_min <= y <= self._y_max):
            self.setText(f"{self._x_label}={x:.3f}, {self._y_label}={y:.3f} [outside bounds]")
            return

        y_bins, x_bins = self._matrix.shape
        x_idx = int(np.clip((x - self._x_min) / (self._x_max - self._x_min) * x_bins, 0, x_bins - 1))
        y_idx = int(np.clip((y - self._y_min) / (self._y_max - self._y_min) * y_bins, 0, y_bins - 1))
        
        value = float(self._matrix[y_idx, x_idx])
        self.setText(f"{self._x_label} = {x:.3f} | {self._y_label} = {y:.3f} | val = {value:.2f} {self._unit}")

# =====================================================================
# 3. ОСНОВНОЙ ВИДЖЕТ ВИЗУАЛИЗАТОРА 2D
# =====================================================================

class Spec2DVisualizer(QtWidgets.QGroupBox):
    def __init__(self, title: str = "2D Function Visualizer", parent=None):
        super().__init__(title, parent)
        self._data_loaded = False
        self._is_y_locked = False

        self._setup_ui()
        self._setup_pyqtgraph()
        self._connect_signals()

        self._on_colormap_changed(self.cm_combo.currentText())
        self._redraw_all()

    def _setup_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 8, 4, 4)

        self.controls_layout = QtWidgets.QHBoxLayout()
        
        self.selector = Selector(title="Данные:")
        self.selector.set_dictionary(plot2d_store)
        self.controls_layout.addWidget(self.selector)

        self.controls_layout.addWidget(QtWidgets.QLabel("Цвет:"))
        self.cm_combo = QtWidgets.QComboBox()
        self.cm_combo.addItems(pg.colormap.listMaps())
        if "viridis" in pg.colormap.listMaps():
            self.cm_combo.setCurrentText("viridis")
        self.controls_layout.addWidget(self.cm_combo)

        self.lock_y_cb = QtWidgets.QCheckBox("Lock Y Axis")
        self.controls_layout.addWidget(self.lock_y_cb)
        
        self.controls_layout.addStretch()
        self.main_layout.addLayout(self.controls_layout)

    def _setup_pyqtgraph(self):
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.main_layout.addWidget(self.graphics_layout, stretch=1)

        self.view_box = pg.ViewBox()
        self.spec_plot = self.graphics_layout.addPlot(row=0, col=0, viewBox=self.view_box)
        self.spec_plot.getAxis('left').setWidth(50)
        self.spec_plot.setLabel('bottom', "Ось X")
        self.spec_plot.setLabel('left', "Ось Y")

        self.image_item = AdaptiveImageItem()
        self.image_item.attachTo(self.spec_plot)
        self.spec_plot.addItem(self.image_item)

        self.minimap = self.graphics_layout.addPlot(row=1, col=0)
        self.minimap.setMaximumHeight(60)
        self.minimap.hideAxis('bottom')
        self.minimap.getAxis('left').setWidth(50)
        self.minimap.getAxis('left').setStyle(showValues=False)
        self.minimap.setMouseEnabled(x=False, y=False)

        self.minimap_curve = self.minimap.plot(pen=pg.mkPen('c', width=1), fillLevel=0, brush=(0, 255, 255, 80))
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.minimap.addItem(self.region)

        self.syncer = Syncer(self.region, self.view_box, parent=self)
        
        self.hover_tracker = HoverTracker(self.view_box)
        self.controls_layout.addWidget(self.hover_tracker)

    def _connect_signals(self):
        self.cm_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.lock_y_cb.toggled.connect(self._on_lock_y_toggled)

        self.selector.signals.keySelected.connect(self._redraw_all)
        self.selector.signals.activeValueChanged.connect(self._redraw_all)

    def _redraw_all(self, *args):
        func: Function2D | None = self.selector.active_value()
        
        if func is None:
            self._clear_all()
            return
            
        # 1. Извлекаем сырые данные
        matrix_t, unit_obj = func._matrx_a, func._matrx_u
        y_axis_t, y_unit = func._first_a, func._first_u
        x_axis_t, x_unit = func._sec___a, func._sec___u

        if matrix_t.ndim > 2:
            matrix_t = matrix_t[0]

        # Определяем лейблы на основе типа данных
        if isinstance(func, SpecFunc):
            x_label, y_label = "Время", "Частота"
        elif isinstance(func, CorrelFunc):
            x_label, y_label = "Время", "Задержка"
        else:
            x_label, y_label = "Ось X", "Ось Y"

        self.spec_plot.setLabel('bottom', x_label, units=str(x_unit))
        self.spec_plot.setLabel('left', y_label, units=str(y_unit))

        # 2. Универсальная конвертация в NumPy для PyQtGraph
        matrix_np = ma.to_numpy(matrix_t)
        
        # СУПЕР ВАЖНО ДЛЯ FPS: Гарантируем, что память не фрагментирована (C-Contiguous)
        matrix_np = np.ascontiguousarray(matrix_np)
        
        x_axis = ma.to_numpy(x_axis_t)
        y_axis = ma.to_numpy(y_axis_t)

        unit_str = str(unit_obj)

        x0, x1 = float(x_axis[0]), float(x_axis[-1])
        y0, y1 = float(y_axis[0]), float(y_axis[-1])

        # ImageItem ожидает данные (Y, X). Наша матрица уже имеет эту форму.
        self.image_item.setFullData(matrix_np, (x0, x1), (y0, y1))

        minimap_data = np.mean(matrix_np, axis=0)
        self.minimap_curve.setData(x=x_axis, y=minimap_data)

        self.hover_tracker.set_data(matrix_np, x0, x1, y0, y1, unit_str, x_label, y_label)
        self._update_camera_limits(x0, x1, y0, y1)


    def _update_camera_limits(self, x0: float, x1: float, y0: float, y1: float):
        self.spec_plot.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self.minimap.setLimits(xMin=None, xMax=None)

        self.minimap.setXRange(x0, x1, padding=0)
        self.region.blockSignals(True)
        self.region.setBounds([x0, x1])
        if not self._data_loaded:
            self.region.setRegion([x0, x1])
        self.region.blockSignals(False)

        if not self._data_loaded:
            self.spec_plot.setXRange(x0, x1, padding=0)
            if self._is_y_locked:
                self.spec_plot.setYRange(y0, y1, padding=0)
            self._data_loaded = True
        else:
            current_x = self.view_box.viewRange()[0]
            if current_x[1] <= x0 or current_x[0] >= x1:
                self.spec_plot.setXRange(x0, x1, padding=0)
                self.region.blockSignals(True)
                self.region.setRegion([x0, x1])
                self.region.blockSignals(False)
            
            if self._is_y_locked:
                self.spec_plot.setYRange(y0, y1, padding=0)

        duration = x1 - x0
        # Если шкала X достаточно большая, фиксируем лимиты камеры
        if duration > 1e-3:
            self.spec_plot.setLimits(xMin=x0, xMax=x1, yMin=y0, yMax=y1)
        else:
            self.spec_plot.setLimits(xMin=None, xMax=None, yMin=y0, yMax=y1)
            
        self.minimap.setLimits(xMin=x0, xMax=x1)

    def _clear_all(self):
        self.image_item.clear()
        self.minimap_curve.setData([], [])
        self.spec_plot.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self.minimap.setLimits(xMin=None, xMax=None)
        self._data_loaded = False
        self.hover_tracker.clear_data()

    def _on_colormap_changed(self, cmap_name: str):
        if hasattr(self.image_item, 'setColorMap'):
            self.image_item.setColorMap(cmap_name)
            if hasattr(self.image_item, '_doUpdate'):
                self.image_item._doUpdate()

    def _on_lock_y_toggled(self, checked: bool):
        self._is_y_locked = checked
        self.spec_plot.setMouseEnabled(x=True, y=not checked)
        
        func = self.selector.active_value()
        if checked and func is not None:
            f_tensor = func._first_a
            f0 = float(f_tensor[0])
            f1 = float(f_tensor[-1])
            self.spec_plot.setYRange(f0, f1, padding=0)

# =====================================================================
# 4. ДЕМОНСТРАТОР (ОТЛАДКА)
# =====================================================================

def run_standalone_demo():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    try:
        import torch
        ctx = ArrayContext(Framework.TORCH, DeviceType.CPU, None)
    except ImportError:
        ctx = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)

    # --- СНАЧАЛА создаем данные в NumPy ---
    t_np = np.linspace(0, 10, 4000)
    f_np = np.linspace(0, 100, 200)
    
    T, F = np.meshgrid(t_np, f_np) 
    matrix_np = np.sin(T * 2) * np.exp(-(F - 50)**2 / 50) 
    
    matrix_t = ma.convert_to(matrix_np, ctx)
    time_t = ma.convert_to(t_np, ctx)
    freq_t = ma.convert_to(f_np, ctx)

    spec_func = SpecFunc(
        matrix=(matrix_t, UREG.V),
        freq=freq_t,
        time=time_t
    )
    
    fw_name = ctx.isTorch() and 'Torch' or 'NumPy'
    update_spec2d(f"Test Spectrogram ({fw_name})", spec_func)
    
    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle("Тест Визуализатора 2D (MultiArray edition)")
    main_window.resize(1000, 800)
    
    visualizer = Spec2DVisualizer()
    main_window.setCentralWidget(visualizer)
    main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_standalone_demo()

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Visualize/__init__.py" encoding="utf-8"
import sys
import threading
from PySide6 import QtWidgets
from .func1d_visualizer import Function1DVisualizer, update_function
from .spec2d_visualizer import Spec2DVisualizer, update_spec2d


def run_visualizer(worker_func, *args, **kwargs):
    """
    Создает окна Qt, запускает переданную функцию в фоновом потоке,
    а затем входит в главный цикл обработки событий GUI.
    """
    # 1. Инициализация приложения
    # Проверяем, нет ли уже созданного QApplication (полезно при перезапусках/дебаге)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # 2. Создаем и показываем окна
    # Важно: сохраняем ссылки (window1, window2), чтобы сборщик мусора их не удалил
    window2 = Function1DVisualizer()
    window2.show()
    
    window1 = Spec2DVisualizer()
    window1.show()

    # 3. ЗАПУСК ВАШИХ ВЫЧИСЛЕНИЙ В ФОНЕ
    # daemon=True означает, что как только пользователь закроет окна (app.exec завершится),
    # этот фоновый поток автоматически жестко остановится.
    calc_thread = threading.Thread(
        target=worker_func,
        args=args,       # Передаем позиционные аргументы, если есть
        kwargs=kwargs,   # Передаем именованные аргументы, если есть
        daemon=True      
    )
    calc_thread.start()

    # 4. Вход в главный цикл событий Qt (Блокирующий вызов)
    sys.exit(app.exec())

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Visualize/func1d_visualizer.py" encoding="utf-8"
import sys
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from typing import Any

# --- Внедряем наш новый универсальный фреймворк ---
import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType

# --- Твои реальные импорты ---
from BatSpec.QtUp.Reactive.reactive_dict import ReactiveDict

from BatSpec.Core.Functions import Function1D, TimeFunc, FreqFunc
from BatSpec.Core.Physical.Units import PintUnit, UREG


# =====================================================================
# 1. ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ ДАННЫХ И API
# =====================================================================

plot_store: ReactiveDict[str, Function1D] = ReactiveDict()

def update_function(name: str, func_obj: Function1D) -> None:
    plot_store[name] = func_obj

def remove_function(name: str) -> None:
    if name in plot_store:
        del plot_store[name]

def clear_all_functions() -> None:
    plot_store.clear()


# =====================================================================
# 2. ВИДЖЕТ ВИЗУАЛИЗАЦИИ
# =====================================================================

class Function1DVisualizer(QtWidgets.QGroupBox):
    def __init__(self, title: str = "1D Function Visualizer", parent: QtWidgets.QWidget | None = None):
        super().__init__(title, parent)

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        # --- СПИСОК ЧЕКБОКСОВ ---
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setMaximumHeight(120)  # Чтобы график занимал больше места
        self.main_layout.addWidget(self.list_widget)
        self.list_widget.itemChanged.connect(self._redraw_plot)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(10, 10)) # Настоящая легенда для поддержки множества графиков
        self.main_layout.addWidget(self.plot_widget)

        # Таймер для синхронизации словаря с QListWidget
        self.sync_timer = QtCore.QTimer(self)
        self.sync_timer.timeout.connect(self._sync_store)
        self.sync_timer.start(200)

        # Набор контрастных цветов для разных графиков
        self.colors = [
            '#00FF00', '#FF00FF', '#00FFFF', '#FFFF00', 
            '#FF5555', '#5555FF', '#FFAA00', '#FFFFFF'
        ]

    def _sync_store(self):
        current_keys = set(plot_store.keys())
        list_keys = set(self.list_widget.item(i).text() for i in range(self.list_widget.count()))
        
        added = current_keys - list_keys
        removed = list_keys - current_keys
        
        if added or removed:
            self.list_widget.blockSignals(True)
            
            # Удаляем те, которых больше нет в store
            for key in removed:
                items = self.list_widget.findItems(key, QtCore.Qt.MatchExactly)
                for item in items:
                    self.list_widget.takeItem(self.list_widget.row(item))
            
            # Добавляем новые с чекбоксами
            for key in added:
                item = QtWidgets.QListWidgetItem(key)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Unchecked)
                self.list_widget.addItem(item)
                
            self.list_widget.blockSignals(False)
            
            if removed:
                self._redraw_plot()

    def _redraw_plot(self, *args) -> None:
        self.plot_widget.clear()

        checked_items = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                checked_items.append(item.text())

        if not checked_items:
            self.plot_widget.setTitle("Выберите графики для отображения")
            self.plot_widget.getPlotItem().setLabel('bottom', 'Ось X')
            self.plot_widget.getPlotItem().setLabel('left', 'Значения')
            return

        self.plot_widget.setTitle("")
        
        color_idx = 0
        first_func = None

        for key in checked_items:
            func = plot_store.get(key)
            if func is None:
                continue
            
            if first_func is None:
                first_func = func

            try:
                # Извлекаем тензоры в их исходном формате
                x_tensor = func._axis__a
                y_tensor = func._value_a

                # Обработка батчей: если данные многомерные, берем первый элемент
                if y_tensor.ndim > 1:
                    y_tensor = y_tensor[0]
                
                # Универсально конвертируем данные в NumPy массивы для PyQtGraph
                x_data = ma.to_numpy(x_tensor)
                y_data = ma.to_numpy(y_tensor) 
                
                color = self.colors[color_idx % len(self.colors)]
                
                # Отрисовываем
                curve = self.plot_widget.plot(
                    x=x_data, 
                    y=y_data, 
                    pen=pg.mkPen(color=color, width=1.5),
                    name=key
                )
                curve.setDownsampling(ds=True, auto=True, method='peak')
                curve.setClipToView(True)

                color_idx += 1

            except Exception as e:
                print(f"Ошибка отрисовки графика {key}: {e}")

        # Настраиваем оси X по первому выбранному графику
        if first_func is not None:
            x_unit_str = str(first_func._axis__u)
            if isinstance(first_func, TimeFunc):
                x_label = "Время"
            elif isinstance(first_func, FreqFunc):
                x_label = "Частота"
            else:
                x_label = "Ось X"

            self.plot_widget.getPlotItem().setLabel('bottom', x_label, units=x_unit_str)
            # Y-ось оставляем общей (без единиц измерения), так как графиков может быть несколько разных типов
            self.plot_widget.getPlotItem().setLabel('left', 'Значения')


# =====================================================================
# 3. ДЕМОНСТРАТОР (ОТЛАДКА)
# =====================================================================

class DemoWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Line Function1D Visualizer")
        self.resize(800, 600)

        self.visualizer = Function1DVisualizer()
        self.setCentralWidget(self.visualizer)
        
        try:
            import torch
            self.ctx = ArrayContext(Framework.TORCH, DeviceType.CPU, None)
        except ImportError:
            self.ctx = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)

        t_np = np.linspace(0, 1, 1000)
        y1_np = np.sin(2 * np.pi * 5 * t_np)
        y2_np = np.cos(2 * np.pi * 5 * t_np) * 0.5
        
        t_tensor = ma.convert_to(t_np, self.ctx)
        
        func1 = TimeFunc(values=(ma.convert_to(y1_np, self.ctx), UREG.dimensionless), axis=t_tensor)
        func2 = TimeFunc(values=(ma.convert_to(y2_np, self.ctx), UREG.dimensionless), axis=t_tensor)
        
        update_function("Синус (Амплитуда 1.0)", func1)
        update_function("Косинус (Амплитуда 0.5)", func2)

        # Выбираем первый график при старте
        self.visualizer._sync_store()
        if self.visualizer.list_widget.count() > 0:
            self.visualizer.list_widget.item(0).setCheckState(QtCore.Qt.Checked)


def run_standalone_demo() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DemoWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    run_standalone_demo()
``````

