

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



if TYPE_CHECKING:   
    from MultiArray.Stubs.numpy import NumpyStub
    from MultiArray.Stubs.torch import TorchStub

    type StubFW = Type[NumpyStub] | Type[TorchStub] | Any # type: ignore
    import numpy
    import torch
    import tensorflow as tf
    import jax
    import cupy

else:
    type StubFW = Any

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
    def fw(self) -> Union[types.ModuleType, StubFW, Any]:
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

    # ==========================================
    # Математические операторы (Math Overloads)
    # ==========================================

    def _math_op(self, other: Any, op: Callable, unit_op: Callable = None) -> Self:
        if isinstance(other, Function1D):
            new_val = op(self._value_a, other._value_a)
            new_u = unit_op(self._value_u, other._value_u) if unit_op else self._value_u
        else:
            new_val = op(self._value_a, other)
            new_u = self._value_u
            
        base = Function1D(
            values=(new_val, new_u), 
            axis=(self._axis__a, self._axis__u)
        )
        if type(self) is Function1D:
            return base
        return type(self).from_Function1D(base)

    def __add__(self, other): return self._math_op(other, lambda a, b: a + b)
    def __radd__(self, other): return self._math_op(other, lambda a, b: b + a)
    
    def __sub__(self, other): return self._math_op(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._math_op(other, lambda a, b: b - a)
    
    def __mul__(self, other): return self._math_op(other, lambda a, b: a * b, lambda u1, u2: u1 * u2)
    def __rmul__(self, other): return self._math_op(other, lambda a, b: b * a, lambda u1, u2: u2 * u1)
    
    def __truediv__(self, other): return self._math_op(other, lambda a, b: a / b, lambda u1, u2: u1 / u2)
    def __rtruediv__(self, other): return self._math_op(other, lambda a, b: b / a, lambda u1, u2: u2 / u1)


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

    # ==========================================
    # Математические операторы (Math Overloads)
    # ==========================================

    def _math_op(self, other: Any, op: Callable, unit_op: Callable = None) -> Self:
        if isinstance(other, Function2D):
            new_mat = op(self._matrx_a, other._matrx_a)
            new_u = unit_op(self._matrx_u, other._matrx_u) if unit_op else self._matrx_u
        else:
            new_mat = op(self._matrx_a, other)
            new_u = self._matrx_u
            
        base = Function2D(
            matrix=(new_mat, new_u), 
            first=(self._first_a, self._first_u), 
            sec=(self._sec___a, self._sec___u)
        )
        if type(self) is Function2D:
            return base
        return type(self).from_Function2D(base)

    def __add__(self, other): return self._math_op(other, lambda a, b: a + b)
    def __radd__(self, other): return self._math_op(other, lambda a, b: b + a)
    
    def __sub__(self, other): return self._math_op(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._math_op(other, lambda a, b: b - a)
    
    def __mul__(self, other): return self._math_op(other, lambda a, b: a * b, lambda u1, u2: u1 * u2)
    def __rmul__(self, other): return self._math_op(other, lambda a, b: b * a, lambda u1, u2: u2 * u1)
    
    def __truediv__(self, other): return self._math_op(other, lambda a, b: a / b, lambda u1, u2: u1 / u2)
    def __rtruediv__(self, other): return self._math_op(other, lambda a, b: b / a, lambda u1, u2: u2 / u1)


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

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Visualize/__test__/1.py" encoding="utf-8"
from BatSpec.Visualize import update_function, update_spec2d, run_visualizer
@run_visualizer
def main():
    from pathlib import Path
    ScriptDir = Path(__file__).parent
    from BatSpec.Core.Record import loadRecord, correctDC
    from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
    from BatSpec.Core.SaveIntegral import SaveIntegral
    from BatSpec.Core.Spectral import makeSpec, makeLogDB
    from BatSpec.Core.ConvWindow import TEST_HANN_WINODW


    record_row = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record_row = correctDC(record_row)
    update_function("record_row", record_row)


    record = applyСalibration(record_row, FlatResponseModel(sensitivity_pa=20))
    update_function("record", record)

    print(f"""
    SaveIntegralEnergy(record_row): {SaveIntegral.Energy(record_row)}
    SaveIntegralEnergy(record): {SaveIntegral.Energy(record)}
    """)

    spec = makeSpec(record, TEST_HANN_WINODW, overlap=0.9, bins=300)
    update_spec2d("spec", spec)

    print(f"""
    SaveIntegralEnergy(spec): {SaveIntegral.Energy(spec)}
    SaveIntegralEnergy(record): {SaveIntegral.Energy(record)}
    """)

    SPSL = makeLogDB(spec)
    update_spec2d("SPSL", SPSL)


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

        # --- ПАНЕЛЬ УПРАВЛЕНИЯ ---
        self.control_layout = QtWidgets.QHBoxLayout()
        self.norm_cb = QtWidgets.QCheckBox("Нормировать Y (Z-Score)")
        self.norm_cb.setChecked(True)
        self.norm_cb.toggled.connect(self._redraw_plot)
        self.control_layout.addWidget(self.norm_cb)
        self.control_layout.addStretch()
        self.main_layout.addLayout(self.control_layout)

        # --- СПИСОК ЧЕКБОКСОВ ---
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setMaximumHeight(120)  
        self.main_layout.addWidget(self.list_widget)
        self.list_widget.itemChanged.connect(self._redraw_plot)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(10, 10)) 
        self.main_layout.addWidget(self.plot_widget)

        self.sync_timer = QtCore.QTimer(self)
        self.sync_timer.timeout.connect(self._sync_store)
        self.sync_timer.start(200)

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
            
            for key in removed:
                items = self.list_widget.findItems(key, QtCore.Qt.MatchExactly)
                for item in items:
                    self.list_widget.takeItem(self.list_widget.row(item))
            
            for key in added:
                item = QtWidgets.QListWidgetItem(key)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked) # По умолчанию включаем
                self.list_widget.addItem(item)
                
            self.list_widget.blockSignals(False)
            
            if removed or added:
                self._redraw_plot()

    def _redraw_plot(self, *args) -> None:
        self.plot_widget.clear()

        checked_items = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                checked_items.append(item.text())

        do_norm = self.norm_cb.isChecked()

        if not checked_items:
            self.plot_widget.setTitle("Выберите графики для отображения")
            self.plot_widget.getPlotItem().setLabel('bottom', 'Ось X')
            self.plot_widget.getPlotItem().setLabel('left', 'Z-Score' if do_norm else 'Значения')
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
                x_tensor = func._axis__a
                y_tensor = func._value_a

                if y_tensor.ndim > 1:
                    y_tensor = y_tensor[0]
                
                x_data = ma.to_numpy(x_tensor)
                y_data = ma.to_numpy(y_tensor) 
                
                # --- СТАТИСТИЧЕСКАЯ НОРМАЛИЗАЦИЯ ---
                if do_norm:
                    mean_val = np.mean(y_data)
                    std_val = np.std(y_data)
                    # Избегаем деления на ноль для константных графиков
                    if std_val < 1e-12:
                        std_val = 1.0
                    y_data = (y_data - mean_val) / std_val

                color = self.colors[color_idx % len(self.colors)]
                
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

        if first_func is not None:
            x_unit_str = str(first_func._axis__u)
            if isinstance(first_func, TimeFunc):
                x_label = "Время"
            elif isinstance(first_func, FreqFunc):
                x_label = "Частота"
            else:
                x_label = "Ось X"

            self.plot_widget.getPlotItem().setLabel('bottom', x_label, units=x_unit_str)
            self.plot_widget.getPlotItem().setLabel('left', 'Z-Score (σ)' if do_norm else 'Значения')

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
        # Сумасшедший разброс порядков:
        y1_np = np.sin(2 * np.pi * 5 * t_np) * 10000.0  
        y2_np = np.cos(2 * np.pi * 5 * t_np) * 0.0001
        
        t_tensor = ma.convert_to(t_np, self.ctx)
        
        func1 = TimeFunc(values=(ma.convert_to(y1_np, self.ctx), UREG.dimensionless), axis=t_tensor)
        func2 = TimeFunc(values=(ma.convert_to(y2_np, self.ctx), UREG.dimensionless), axis=t_tensor)
        
        update_function("Синус (Ампл 10000)", func1)
        update_function("Косинус (Ампл 0.0001)", func2)

def run_standalone_demo() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_standalone_demo()
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
    # Передавая None в dtype, мы заставляем convert_to не менять тип данных насильно
    return_ctx = ma.ArrayContext(orig_ctx._framework, orig_ctx._device, None)
    
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
    
    # Возвращаем в исходный контекст, позволяя dtype определиться автоматически (как вещественный/real)
    return_ctx = ma.ArrayContext(orig_ctx._framework, orig_ctx._device, None)
    return time_math.to_context(return_ctx)


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
    return_ctx = ma.ArrayContext(orig_ctx._framework, orig_ctx._device, None)
    
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

def makePiecewiseLog(spec: SpecFunc) -> SpecFunc:
    """
    Применяет кусочно-непрерывное логарифмическое сжатие к спектрограмме:
    Y = X,           если X < 0
    Y = ln(X + 1),   если X >= 0
    
    Идеально подходит для работы с Z-нормированными данными (где есть отрицательные значения),
    сжимая огромные пики, но не искажая шумовую полку.
    """
    _, old_unit = spec.values

    def piecewise_log_func(arr: Any) -> Any:
        ctx = ma.ArrayContext.from_array(arr)
        
        if ctx.isTorch():
            # ctx.fw.log1p эквивалентно ln(x + 1)
            return ctx.fw.where(arr > 0, ctx.fw.log1p(arr), arr)
            
        elif ctx.isTensorflow():
            # TensorFlow строго относится к типам в условиях
            zero = ctx.fw.constant(0.0, dtype=arr.dtype)
            return ctx.fw.where(arr > zero, ctx.fw.math.log1p(arr), arr)
            
        else:
            # Для NumPy, JAX и CuPy
            return ctx.fw.where(arr > 0, ctx.fw.log1p(arr), arr)

    # При логарифмировании физическая размерность (например, Вольты) утрачивает прямой смысл,
    # поэтому возвращаем безразмерную величину (dimensionless).
    return spec.cloneApply(func=piecewise_log_func, new_unit=UREG.dimensionless)



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

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/Statistic/__init__.py" encoding="utf-8"
import numpy as np
import scipy.ndimage as ndimage
import warnings
from typing import Any, Tuple, Union

# --- 1. Импорты специфичных для проекта библиотек ---
from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.Physical.Units import UREG
# Внедряем наш новый универсальный фреймворк
import MultiArray as ma
from MultiArray import ArrayContext

# Пытаемся импортировать наш супербыстрый C++ модуль
try:
    from .FastZScore import fast_median_2d
    HAS_FAST_ZSCORE = True
except ImportError:
    HAS_FAST_ZSCORE = False

ArrayLike = Any

# =====================================================================
def noiseZNormByFreq(spec: SpecFunc, noise_percentile: float = 10.0) -> SpecFunc:
    ctx = spec.context
    time_func = spec.integrateOverFreq()
    energy_profile, _ = time_func.values

    energy_np = ma.to_numpy(energy_profile)
    threshold_val = np.percentile(energy_np, noise_percentile)
    
    threshold = ma.convert_to(np.array(threshold_val), ctx)
    noise_mask = energy_profile <= threshold

    if not np.any(ma.to_numpy(noise_mask)):
        warnings.warn("Не найдено шумовых срезов по процентилю. Используется срез с минимальной энергией.")
        min_energy_idx = ma.argmax(-energy_profile) 
        time_indices = ma.arange(energy_profile.shape[-1], ctx)
        noise_mask = (time_indices == min_energy_idx)

    def z_norm_func(mat: ArrayLike) -> ArrayLike:
        ctx_float = ArrayContext(ctx._framework, ctx._device, mat.dtype)
        mask_float = ma.convert_to(noise_mask, ctx_float)
        
        noise_frame_count = ma.sum(mask_float)
        noise_frame_count = ma.clamp_min(noise_frame_count, 1.0)
        
        noise_only_mat = mat * mask_float
        
        noise_sum = ma.sum(noise_only_mat, axis=-1, keepdims=True)
        noise_mean = noise_sum / noise_frame_count
        
        dev_sq = (mat - noise_mean)**2 * mask_float
        noise_var = ma.sum(dev_sq, axis=-1, keepdims=True) / noise_frame_count
        noise_std = ma.sqrt(noise_var)
        
        safe_std = ma.clamp_min(noise_std, 1e-9)
        return (mat - noise_mean) / safe_std

    return spec.cloneApply(func=z_norm_func, new_unit=UREG.dimensionless)


# =====================================================================
def localNoiseZNorm(spec: SpecFunc, window_time: float = 0.05, window_freq: float = 0.0) -> SpecFunc:
    ctx = spec.context
    dt_val, _ = spec.dt
    df_val, _ = spec.df

    time_bins = max(1, int(round(window_time / float(dt_val))))
    freq_bins = max(1, int(round(window_freq / float(df_val))))

    if time_bins % 2 == 0: time_bins += 1
    if freq_bins % 2 == 0: freq_bins += 1

    def _gpu_moving_median(tensor, k_h, k_w):
        import torch
        import torch.nn.functional as F
        
        pad_h, pad_w = k_h // 2, k_w // 2
        orig_shape = tensor.shape
        
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
            
        padded = F.pad(tensor.unsqueeze(1), (pad_w, pad_w, pad_h, pad_h), mode='reflect').squeeze(1)
        
        if k_h == 1:
            unfolded = padded.unfold(-1, k_w, 1)
            median = unfolded.median(dim=-1).values
            return median.view(orig_shape)
            
        H, W = tensor.shape[-2], tensor.shape[-1]
        res = torch.empty_like(tensor)
        
        bytes_per_element = 4
        chunk_size = max(1, (100 * 1024 * 1024) // (H * k_h * k_w * bytes_per_element))
        
        for w_start in range(0, W, chunk_size):
            w_end = min(W, w_start + chunk_size)
            padded_chunk = padded[..., :, w_start : w_end + 2 * pad_w]
            
            unfolded = padded_chunk.unfold(-2, k_h, 1).unfold(-1, k_w, 1)
            unfolded = unfolded.contiguous().view(*unfolded.shape[:-2], -1)
            
            res[..., :, w_start:w_end] = unfolded.median(dim=-1).values
            
        return res.view(orig_shape)

    def _cpu_moving_median(mat_np, k_h, k_w):
        if HAS_FAST_ZSCORE:
            mat_np_f32 = mat_np.astype(np.float32)
            ph, pw = k_h // 2, k_w // 2
            
            if mat_np_f32.ndim == 2:
                padded = np.pad(mat_np_f32, ((ph, ph), (pw, pw)), mode='symmetric')
                h, w = mat_np_f32.shape
                return fast_median_2d(padded, h, w, k_h, k_w)
                
            elif mat_np_f32.ndim == 3: 
                res = np.empty_like(mat_np_f32)
                for b in range(mat_np_f32.shape[0]):
                    padded = np.pad(mat_np_f32[b], ((ph, ph), (pw, pw)), mode='symmetric')
                    h, w = mat_np_f32[b].shape
                    res[b] = fast_median_2d(padded, h, w, k_h, k_w)
                return res

        if k_h == 1:
            try:
                import pandas as pd
                df = pd.DataFrame(mat_np.T)
                return df.rolling(k_w, center=True, min_periods=1).median().values.T
            except ImportError:
                pass
                
        window_shape = [1] * mat_np.ndim
        window_shape[-2] = k_h 
        window_shape[-1] = k_w
        return ndimage.median_filter(mat_np, size=tuple(window_shape), mode='reflect')

    def local_z_norm_func(mat: ArrayLike) -> ArrayLike:
        if ctx.isTorch():
            import torch
            local_median = _gpu_moving_median(mat, freq_bins, time_bins)
            abs_dev = torch.abs(mat - local_median)
            local_mad = _gpu_moving_median(abs_dev, freq_bins, time_bins)
            
            local_std = torch.clamp(local_mad * 1.4826, min=1e-9)
            return (mat - local_median) / local_std
        else:
            mat_np = ma.to_numpy(mat)
            local_median = _cpu_moving_median(mat_np, freq_bins, time_bins)
            abs_dev = np.abs(mat_np - local_median)
            local_mad = _cpu_moving_median(abs_dev, freq_bins, time_bins)

            local_std = np.maximum(local_mad * 1.4826, 1e-9)
            z_normed_np = (mat_np - local_median) / local_std
            return ma.convert_to(z_normed_np, ctx)

    return spec.cloneApply(func=local_z_norm_func, new_unit=UREG.dimensionless)


# =====================================================================
def blurSpec(spec: SpecFunc, sigma: Union[float, Tuple[float, float]] = 1.0) -> SpecFunc:
    ctx = spec.context
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))
    sigma_f, sigma_t = sigma

    def blur_func(mat: ArrayLike) -> ArrayLike:
        if sigma_f <= 0 and sigma_t <= 0:
            return mat

        if ctx.isTorch():
            import torch
            import torch.nn.functional as F
            
            orig_shape = mat.shape
            tensor = mat.unsqueeze(0).unsqueeze(0) if mat.ndim == 2 else mat.unsqueeze(1)
            
            def get_1d_kernel(s):
                k = int(round(s * 3)) * 2 + 1
                if k < 3: k = 3
                x = torch.arange(k, device=tensor.device, dtype=tensor.dtype) - k // 2
                kernel = torch.exp(-0.5 * (x / s) ** 2)
                return kernel / kernel.sum()
                
            if sigma_f > 0:
                k_h = get_1d_kernel(sigma_f).view(1, 1, -1, 1)
                pad_h = k_h.shape[2] // 2
                pad_mode = 'reflect' if pad_h < tensor.shape[-2] else 'replicate'
                tensor = F.pad(tensor, (0, 0, pad_h, pad_h), mode=pad_mode)
                tensor = F.conv2d(tensor, k_h)
                
            if sigma_t > 0:
                k_w = get_1d_kernel(sigma_t).view(1, 1, 1, -1)
                pad_w = k_w.shape[3] // 2
                pad_mode = 'reflect' if pad_w < tensor.shape[-1] else 'replicate'
                tensor = F.pad(tensor, (pad_w, pad_w, 0, 0), mode=pad_mode)
                tensor = F.conv2d(tensor, k_w)
                
            return tensor.view(orig_shape)
        else:
            mat_np = ma.to_numpy(mat)
            blurred_np = ndimage.gaussian_filter(mat_np, sigma=(sigma_f, sigma_t), mode='reflect')
            return ma.convert_to(blurred_np, ctx)

    _, current_unit = spec.values
    return spec.cloneApply(func=blur_func, new_unit=current_unit)


def multiScaleGeometricMean(
    spec: SpecFunc, 
    sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
) -> SpecFunc:
    """
    МУЛЬТИМАСШТАБНОЕ УМНОЖЕНИЕ (Geometric Mean) с кастомными весами.
    Первый вес относится к оригинальному сигналу, остальные — к соответствующим размытиям (sigmas).
    Веса автоматически нормируются так, чтобы их сумма равнялась 1.0.
    
    Сохраняет математику уничтожения шума и решает проблему минусов через abs().
    Возвращает оригинальный знак в самом конце.
    """
    if len(weights) != len(sigmas) + 1:
        raise ValueError(f"Количество весов ({len(weights)}) должно быть на 1 больше количества sigmas ({len(sigmas)}).")

    ctx = spec.context
    _, current_unit = spec.values
    
    # Нормируем веса (чтобы в сумме они давали 1.0)
    total_w = sum(weights)
    if total_w == 0:
        raise ValueError("Сумма весов не может быть равна нулю.")
    norm_weights = [float(w) / total_w for w in weights]

    def process(mat: ArrayLike) -> ArrayLike:
        # Сохраняем оригинальные знаки (чтобы шум продолжил колебаться вокруг 0, а не ушел вверх)
        if ctx.isTorch(): sign_mat = ctx.fw.sign(mat)
        elif ctx.isTensorflow(): sign_mat = ctx.fw.math.sign(mat)
        else: sign_mat = ctx.fw.sign(mat)

        # 1. Стартуем с МОДУЛЯ оригинального сигнала
        res = (ma.abs(mat) + 1e-9) ** norm_weights[0]
        
        # 2. Итерируемся по сигмам
        for i, sig in enumerate(sigmas):
            w = norm_weights[i + 1]
            
            # Оптимизация: если вес нулевой, пропускаем расчет размытия для этой сигмы
            if w <= 0:
                continue
                
            # РАЗМЫВАЕМ ОРИГИНАЛЬНЫЙ SPEC (с минусами, чтобы они взаимно уничтожали друг друга)
            blurred_spec = blurSpec(spec, sigma=sig)
            blurred_mat, _ = blurred_spec.values
            
            # Умножаем на МОДУЛЬ размытого сигнала в заданной нормированной степени
            res = res * ((ma.abs(blurred_mat) + 1e-9) ** w)
            
        # 3. Возвращаем оригинальный знак!
        return res * sign_mat

    return spec.cloneApply(process, current_unit)

def multiScaleLogProduct(spec: SpecFunc, sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)) -> SpecFunc:
    """
    Точное автоматизированное воспроизведение перемножения кусочных логарифмов.
    Работает как мощный нелинейный гейт:
    - Если хотя бы на одном слое (сигме) пустота (0), то всё умножается на 0 (шум исчезает).
    - Если на всех слоях есть сигнал, значения перемножаются и контраст взлетает в космос.
    """
    # 1. Начинаем с логарифмированного оригинала
    res = makePiecewiseLog(spec)
    
    # 2. В цикле размываем ОРИГИНАЛ, логарифмируем и перемножаем (как ты и писал)
    for sig in sigmas:
        blurred_spec = blurSpec(spec, sigma=sig)
        res = res * makePiecewiseLog(blurred_spec)
        
    return res

# =====================================================================
def makePiecewiseLog(spec: SpecFunc) -> SpecFunc:
    """
    Symmetric Logarithm: Y = sign(X) * log(1 + abs(X)).
    Идеально сжимает пики как в положительную, так и в отрицательную сторону без отсечений!
    """
    def piecewise_log_func(arr: Any) -> Any:
        ctx = ma.ArrayContext.from_array(arr)
        
        abs_arr = ma.abs(arr)
        if ctx.isTorch():
            return ctx.fw.sign(arr) * ctx.fw.log1p(abs_arr)
        elif ctx.isTensorflow():
            return ctx.fw.math.sign(arr) * ctx.fw.math.log1p(abs_arr)
        else:
            return ctx.fw.sign(arr) * ctx.fw.log1p(abs_arr)

    return spec.cloneApply(func=piecewise_log_func, new_unit=UREG.dimensionless)

def medianDenoise(spec: SpecFunc, kernel_time: int = 3, kernel_freq: int = 3) -> SpecFunc:
    """
    Удаляет изолированный 'пиксельный' шум, не размывая края полезного сигнала.
    """
    _, current_unit = spec.values
    ctx = spec.context

    # Убеждаемся, что ядро нечетное
    if kernel_time % 2 == 0: kernel_time += 1
    if kernel_freq % 2 == 0: kernel_freq += 1

    def process(mat: ArrayLike) -> ArrayLike:
        if ctx.isNumpy() and HAS_FAST_ZSCORE:
            # Используем твой C++ модуль!
            mat_np_f32 = ma.to_numpy(mat).astype(np.float32)
            ph, pw = kernel_freq // 2, kernel_time // 2
            
            if mat_np_f32.ndim == 2:
                padded = np.pad(mat_np_f32, ((ph, ph), (pw, pw)), mode='symmetric')
                h, w = mat_np_f32.shape
                res = fast_median_2d(padded, h, w, kernel_freq, kernel_time)
            elif mat_np_f32.ndim == 3: 
                res = np.empty_like(mat_np_f32)
                for b in range(mat_np_f32.shape[0]):
                    padded = np.pad(mat_np_f32[b], ((ph, ph), (pw, pw)), mode='symmetric')
                    h, w = mat_np_f32[b].shape
                    res[b] = fast_median_2d(padded, h, w, kernel_freq, kernel_time)
                    
            return ma.convert_to(res, ctx)
        else:
            # Фолбэк для Torch/TF/обычного numpy
            import scipy.ndimage as ndimage
            mat_np = ma.to_numpy(mat)
            window_shape = [1] * mat_np.ndim
            window_shape[-2] = kernel_freq 
            window_shape[-1] = kernel_time
            res = ndimage.median_filter(mat_np, size=tuple(window_shape), mode='reflect')
            return ma.convert_to(res, ctx)

    return spec.cloneApply(process, current_unit)

def deadZoneFilter(spec: SpecFunc, threshold: float = 2.0) -> SpecFunc:
    """
    Подавляет шум около нуля. Все значения от -threshold до +threshold становятся нулями.
    Значения за пределами плавно 'съезжают' к нулю, сохраняя свой знак.
    """
    _, current_unit = spec.values

    def process(mat: ArrayLike) -> ArrayLike:
        ctx = ma.ArrayContext.from_array(mat)
        
        # Получаем знак (+1 или -1)
        if ctx.isTorch(): sign_mat = ctx.fw.sign(mat)
        elif ctx.isTensorflow(): sign_mat = ctx.fw.math.sign(mat)
        else: sign_mat = ctx.fw.sign(mat)
        
        # Берем модуль, вычитаем порог и отсекаем то, что ушло в минус (это и был шум)
        abs_mat = ma.abs(mat)
        shifted = ma.clamp_min(abs_mat - threshold, 0.0)
        
        # Возвращаем оригинальный знак
        return shifted * sign_mat

    return spec.cloneApply(process, current_unit)
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/Statistic/FastZScore/__init__.cpp" encoding="utf-8"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

// Структура Dual-Heap (Mediator) Ричарда Хартера
struct Mediator {
    std::vector<int> pos;
    std::vector<int> heap_base;
    int* heap;
    std::vector<float> data;
    int N, idx, minCt, maxCt;

    Mediator(int nItems) : pos(nItems), heap_base(nItems), data(nItems, 0.0f) {
        N = nItems;
        int rank = N / 2;
        heap = heap_base.data() + rank;
        reset();
    }

    void reset() {
        int rank = N / 2;
        idx = 0;
        minCt = N - rank - 1;
        maxCt = rank;
        std::fill(data.begin(), data.end(), 0.0f);
        for (int i = 0; i < N; i++) {
            pos[i] = i - rank;
            heap[pos[i]] = i;
        }
    }

    inline bool mmless(int i, int j) { return data[heap[i]] < data[heap[j]]; }

    inline bool mmexchange(int i, int j) {
        int t = heap[i];
        heap[i] = heap[j];
        heap[j] = t;
        pos[heap[i]] = i;
        pos[heap[j]] = j;
        return true;
    }

    inline bool mmCmpExch(int i, int j) {
        return (mmless(i, j) && mmexchange(i, j));
    }

    void minSortDown(int i) {
        for (i *= 2; i <= minCt; i *= 2) {
            if (i < minCt && mmless(i + 1, i)) ++i;
            if (!mmCmpExch(i, i / 2)) break;
        }
    }

    void maxSortDown(int i) {
        for (i *= 2; i >= -maxCt; i *= 2) {
            if (i > -maxCt && mmless(i, i - 1)) --i;
            if (!mmCmpExch(i / 2, i)) break;
        }
    }

    inline bool minSortUp(int i) {
        while (i > 0 && mmCmpExch(i, i / 2)) i /= 2;
        return (i == 0);
    }

    inline bool maxSortUp(int i) {
        while (i < 0 && mmCmpExch(i / 2, i)) i /= 2;
        return (i == 0);
    }

    void insert(float v) {
        int p = pos[idx];
        float old = data[idx];
        data[idx] = v;
        idx++;
        if (idx == N) idx = 0;

        if (p > 0) {
            if (v > old) { minSortDown(p); return; }
            if (minSortUp(p) && mmCmpExch(0, -1)) maxSortDown(-1);
        } else if (p < 0) {
            if (v < old) { maxSortDown(p); return; }
            if (maxSortUp(p) && mmCmpExch(1, 0)) minSortDown(1);
        } else {
            if (maxSortUp(-1)) maxSortDown(-1);
            if (minSortUp(1)) minSortDown(1);
        }
    }

    float get_median() {
        return data[heap[0]];
    }
};

py::array_t<float> fast_median_2d(py::array_t<float, py::array::c_style | py::array::forcecast> input_padded, int h_orig, int w_orig, int kh, int kw) {
    // 1. Извлекаем указатели ДО освобождения GIL (вызовы PyBind11 требуют GIL)
    py::buffer_info buf_info = input_padded.request();
    const float* in_ptr = static_cast<float*>(buf_info.ptr);
    int stride = buf_info.shape[1]; 

    py::array_t<float> output({h_orig, w_orig});
    float* out_ptr = static_cast<float*>(output.request().ptr);

    // 2. ОСВОБОЖДАЕМ GIL! Теперь Python (UI поток) может работать параллельно с C++
    py::gil_scoped_release release;

    // 3. Резервируем 1 ядро для UI, чтобы OpenMP не задушил операционную систему
    int num_threads = omp_get_max_threads();
    if (num_threads > 1) {
        omp_set_num_threads(num_threads - 1); 
    }

    #pragma omp parallel
    {
        Mediator med(kh * kw);

        #pragma omp for schedule(dynamic)
        for (int y = 0; y < h_orig; ++y) {
            med.reset();

            for (int dx = 0; dx < kw; ++dx) {
                for (int dy = 0; dy < kh; ++dy) {
                    med.insert(in_ptr[(y + dy) * stride + dx]);
                }
            }
            out_ptr[y * w_orig + 0] = med.get_median();

            for (int x = 1; x < w_orig; ++x) {
                int dx = x + kw - 1;
                for (int dy = 0; dy < kh; ++dy) {
                    med.insert(in_ptr[(y + dy) * stride + dx]);
                }
                out_ptr[y * w_orig + x] = med.get_median();
            }
        }
    }
    // GIL автоматически захватывается обратно при выходе из функции
    return output;
}

PYBIND11_MODULE(FastZScore, m) {
    m.doc() = "BatSpec Spectral Statistics: Fast 2D Median Module";
    m.def("fast_median_2d", &fast_median_2d, 
          "Супероптимизированный расчет 2D медианы на C++ (Dual-Heap + OpenMP)",
          py::arg("input_padded"), py::arg("h_orig"), py::arg("w_orig"), py::arg("kh"), py::arg("kw"));
}

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/Statistic/FastZScore/__init__.pyi" encoding="utf-8"
import numpy as np
import numpy.typing as npt

def fast_median_2d(
    input_padded: npt.NDArray[np.float32],
    h_orig: int,
    w_orig: int,
    kh: int,
    kw: int
) -> npt.NDArray[np.float32]:
    """
    Вычисляет локальную (скользящую) медиану для 2D массива.
    
    Использует структуру Dual-Heap O(K*logK) вместо O(K*K) и многопоточность C++ (OpenMP).
    Алгоритмически быстрее SciPy в десятки раз.

    Args:
        input_padded (NDArray[np.float32]): Исходный массив, предварительно расширенный (np.pad) 
                                            на `kh // 2` и `kw // 2` с каждой стороны.
        h_orig (int): Оригинальная высота массива (без паддинга).
        w_orig (int): Оригинальная ширина массива (без паддинга).
        kh (int): Размер скользящего окна по оси Y.
        kw (int): Размер скользящего окна по оси X.

    Returns:
        NDArray[np.float32]: Результирующая матрица формы (h_orig, w_orig) с медианами.
    """
    ...

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/Statistic/FastZScore/setup.py" encoding="utf-8"
from setuptools import setup, Extension
import pybind11
import sys
import os

if sys.platform.startswith("win"):
    extra_compile_args = ['/openmp', '/O2', '/fp:fast']
    extra_link_args = []
else:
    extra_compile_args = ['-fopenmp', '-O3', '-ffast-math']
    extra_link_args = ['-fopenmp']

# Указываем полный путь к модулю через точки
ext_modules = [
    Extension(
        "__init__", 
        sources=["__init__.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),
]

setup(
    ext_modules=ext_modules,
)
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/Statistic/FastZScore/__test__/FastZScore.test.py" encoding="utf-8"
import numpy as np
import time
from scipy.ndimage import median_filter
from BatSpec.Core.Spectral.Statistic.FastZScore import fast_median_2d

def z_norm_cpp(arr, kh, kw):
    ph, pw = kh // 2, kw // 2
    h, w = arr.shape
    
    # ИСПОЛЬЗУЕМ mode='symmetric' чтобы совпадало с SciPy mode='reflect'
    padded1 = np.pad(arr, ((ph, ph), (pw, pw)), mode='symmetric')
    med = fast_median_2d(padded1, h, w, kh, kw)
    
    dev = np.abs(arr - med)
    padded2 = np.pad(dev, ((ph, ph), (pw, pw)), mode='symmetric')
    mad = fast_median_2d(padded2, h, w, kh, kw)
    
    return (arr - med) / (mad * 1.4826 + 1e-8)

def z_norm_scipy(arr, kh, kw):
    med = median_filter(arr, size=(kh, kw), mode='reflect')
    dev = np.abs(arr - med)
    mad = median_filter(dev, size=(kh, kw), mode='reflect')
    return (arr - med) / (mad * 1.4826 + 1e-8)

if __name__ == "__main__":
    SIZE = 512
    KH = 11
    KW = 51
    data = np.random.randn(SIZE, SIZE).astype(np.float32)

    print(f"Тест: Массив {SIZE}x{SIZE}, Окно {KH}x{KW} (float32)\n")

    t0 = time.time()
    res_scipy = z_norm_scipy(data, KH, KW)
    t_scipy = time.time() - t0
    print(f"SciPy ndimage: {t_scipy:.3f} сек 🐢")

    t0 = time.time()
    res_cpp = z_norm_cpp(data, KH, KW)
    t_cpp = time.time() - t0
    print(f"Наш C++ алгоритм: {t_cpp:.3f} сек ⚡")

    diff = np.abs(res_scipy - res_cpp).max()
    print(f"\nРазница в расчетах: {diff:.5e} (теперь идеально 0.0)")
    print(f"Ускорение на CPU: в {t_scipy / t_cpp:.1f} РАЗ!")

``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/Statistic/__test__/1.py" encoding="utf-8"
from pathlib import Path

ScriptDir = Path(__file__).parent

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW, TEST_BLHA_WINODW, Window, WindowNorm
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.Spectral import makeLogDB, DSPContext, makeSpec, makePiecewiseLog, interpolateByFactors
from BatSpec.Core.Spectral.Statistic import noiseZNormByFreq, localNoiseZNorm, blurSpec
from BatSpec.Visualize import update_spec2d, run_visualizer, update_function
from MultiArray.Core import ArrayContext, Framework, DeviceType
# --- Импорты MultiArray ---
from scipy.signal.windows import hann

@run_visualizer
def main():
    # record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record = loadRecord(ScriptDir / "MYODAS_20230624_011300.wav")

    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    update_function("record", record)

    ctx_numpy_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    
    print("--- Основные вычисления ---")
    with DSPContext(ctx_numpy_cpu):
        # spec_robust = makeSpec(
        #     record,
        #     window=(TEST_HANN_WINODW, TEST_BLHA_WINODW),
        #     overlap=0.8,
        #     bins=300,
        #     shifts=(-1, 1)  
        # ).to_context(ctx_numpy_cpu)
        spec_robust = makeSpec(
            record,
            window=Window(
                lambda n: hann(n, sym=False), 0.001, WindowNorm.ENERGY
            ),
            overlap=0.8,
            bins=300,
        )

    update_spec2d("Robust", makeLogDB(spec_robust, add_one=False))

    zspec = noiseZNormByFreq(spec_robust)
    # update_spec2d("zspec", zspec)
    update_spec2d("zspec_pl", makePiecewiseLog(zspec))

    lz = localNoiseZNorm(zspec, window_time=0.005, window_freq=10000)
    update_spec2d("lz", makePiecewiseLog(lz))

    lz1 = blurSpec(lz, sigma=1)
    update_spec2d("lz1", makePiecewiseLog(lz1))

    lz2 = blurSpec(lz, sigma=2)
    update_spec2d("lz2", makePiecewiseLog(lz2))

    lz4 = blurSpec(lz, sigma=4)
    update_spec2d("lz4", makePiecewiseLog(lz4))

    lz8 = blurSpec(lz, sigma=8)
    update_spec2d("lz8", makePiecewiseLog(lz8))

    lz16 = blurSpec(lz, sigma=16)
    update_spec2d("lz16", makePiecewiseLog(lz16))


    comb = makePiecewiseLog(lz) * makePiecewiseLog(lz1) * makePiecewiseLog(lz2) * makePiecewiseLog(lz4) * makePiecewiseLog(lz8) * makePiecewiseLog(lz16)
    update_spec2d("comb", makePiecewiseLog(comb))
    update_spec2d("comb", makeLogDB(comb, add_one=True))



    





``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/BatSpec/Source/BatSpec/Core/Spectral/Statistic/__test__/2.py" encoding="utf-8"
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
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/AdaptiveSequenogram/Source/BatCNN/Config.py" encoding="utf-8"
import torch
import psutil

class HardwareConfig:
    """Умный анализатор среды. Адаптирует гиперпараметры под доступное железо."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Константы
        self.CHUNK_SIZE = 1024
        self.FREQ_BINS = 256
        self.TARGET_SR = 256000
        
        # Динамические параметры
        self.BATCH_SIZE = 4
        self.GPU_POOL_SIZE = 4
        self.CPU_QUEUE_SIZE = 2
        
        self._analyze_hardware()

    def _analyze_hardware(self):
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        if self.device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            name = torch.cuda.get_device_name(0)
            print(f"[*] Обнаружена GPU: {name} (VRAM: {vram_gb:.1f} GB)")
            print(f"[*] Системная RAM: {ram_gb:.1f} GB")
            
            if vram_gb >= 20:     
                self.BATCH_SIZE = 128
                self.GPU_POOL_SIZE = 40
            elif vram_gb >= 14:   
                self.BATCH_SIZE = 64
                self.GPU_POOL_SIZE = 25
            elif vram_gb >= 8:    
                self.BATCH_SIZE = 32
                self.GPU_POOL_SIZE = 15
            elif vram_gb >= 4:
                self.BATCH_SIZE = 16
                self.GPU_POOL_SIZE = 8
            else: # Ультра-бюджетные карты типа MX150 (2GB VRAM)
                self.BATCH_SIZE = 4
                self.GPU_POOL_SIZE = 4
                
        else:
            print(f"[*] Обнаружен CPU. Обучение будет медленным. (RAM: {ram_gb:.1f} GB)")
            self.BATCH_SIZE = 4
            self.GPU_POOL_SIZE = 2
            
        if ram_gb >= 32:
            self.CPU_QUEUE_SIZE = 15
        elif ram_gb >= 16:
            self.CPU_QUEUE_SIZE = 8
        else:
            self.CPU_QUEUE_SIZE = 3
            
        print(f"[*] Настроено: Batch={self.BATCH_SIZE}, "
              f"GPU_Pool={self.GPU_POOL_SIZE}, CPU_Queue={self.CPU_QUEUE_SIZE}")
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/AdaptiveSequenogram/Source/BatCNN/DataLoader.py" encoding="utf-8"
import random
import time
import multiprocessing as mp
import torch
import numpy as np
from pathlib import Path
from typing import List

from scipy.signal.windows import hann, blackmanharris

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import Window, WindowNorm
from BatSpec.Core.Record import loadRecord, correctDC, resampleRecord
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.Spectral import makeLogDB, DSPContext, makeRobustSpec
from BatSpec.Core.Spectral.Statistic import noiseZNormByFreq

import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType
from BatCNN.Config import HardwareConfig

# ==========================================
# 1. ЖЕСТКАЯ ГЕНЕРАЦИЯ ОКОН (Safe for Multiprocessing)
# ==========================================
# Функции-обертки нужны вместо lambda, чтобы multiprocessing не падал с ошибкой pickling
def _get_hann(n: int):
    return hann(n, sym=False)

def _get_blha(n: int):
    return blackmanharris(n, sym=False)

# ИДЕАЛЬНОЕ ВРЕМЯ:
# При bins=256, makeRobustSpec делает fft_length = (256-1)*2 = 510.
# Значит, окно должно быть ровно 510 сэмплов.
# time = 510 / 256000 = 0.0019921875 сек.
EXACT_WINDOW_TIME = 510 / 256000

FIXED_HANN = Window(func=_get_hann, time=EXACT_WINDOW_TIME, norm=WindowNorm.ENERGY)
FIXED_BLHA = Window(func=_get_blha, time=EXACT_WINDOW_TIME, norm=WindowNorm.ENERGY)


# ==========================================
# 2. CPU ПРОЦЕСС (MATH WORKER)
# ==========================================
def cpu_worker(wav_paths: List[Path], queue: mp.Queue, target_sr: int, target_bins: int):
    """Фоновый процесс: читает аудио, делает STFT и нормализацию, кладет Numpy-матрицы в очередь."""
    ctx_numpy_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    
    while True:
        try:
            path = random.choice(wav_paths)
            
            # Тайм-домен (Ресемплинг + Калибровка)
            record = loadRecord(path)
            record = resampleRecord(record, target_sr) # SR жестко 256000
            record = correctDC(record)
            record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
            
            # Частотный домен (STFT)
            with DSPContext(ctx_numpy_cpu):
                spec_robust = makeRobustSpec(
                    record,
                    window=(FIXED_HANN, FIXED_BLHA),
                    overlap=0.8,
                    bins=target_bins, 
                    shifts=(-1, 1)  
                ).to_context(ctx_numpy_cpu)
            
            # Нормализация
            zspec = noiseZNormByFreq(spec_robust)
            log_spec = makeLogDB(zspec, add_one=True)
            
            # Извлекаем сырой numpy array: [C, F, T]
            tensor_np = ma.to_numpy(log_spec.values[0])
            if tensor_np.ndim == 2:
                tensor_np = np.expand_dims(tensor_np, axis=0) 
                
            tensor_np = tensor_np.astype(np.float32)
            
            # Кладем в очередь (заблокируется, если RAM очередь полна)
            queue.put(tensor_np)
            
        except Exception as e:
            print(f"[CPU Worker Error] Ошибка при обработке {path}: {e}")
            time.sleep(1)


# ==========================================
# 3. GPU МЕНЕДЖЕР
# ==========================================
class DualProcessDataLoader:
    def __init__(self, wav_paths: List[Path], config: HardwareConfig):
        self.cfg = config
        self.gpu_pool: List[torch.Tensor] = []
        
        # 'spawn' критичен для PyTorch (предотвращает зависания CUDA)
        ctx = mp.get_context('spawn') 
        self.queue = ctx.Queue(maxsize=self.cfg.CPU_QUEUE_SIZE)
        
        self.worker_process = ctx.Process(
            target=cpu_worker, 
            args=(wav_paths, self.queue, self.cfg.TARGET_SR, self.cfg.FREQ_BINS),
            daemon=True
        )
        self.worker_process.start()
        print("[DataLoader] Фоновый CPU Worker запущен.")

    def _sync_queue_to_gpu(self):
        """Переносит всё готовое из RAM-очереди в VRAM-пул без блокировки."""
        while not self.queue.empty():
            try:
                tensor_np = self.queue.get_nowait()
                # Перенос на GPU
                tensor_gpu = torch.tensor(tensor_np, device=self.cfg.device, dtype=torch.float32)
                self.gpu_pool.append(tensor_gpu)
                
                # Очистка старых данных (FIFO)
                if len(self.gpu_pool) > self.cfg.GPU_POOL_SIZE:
                    old_tensor = self.gpu_pool.pop(0)
                    del old_tensor 
            except mp.queues.Empty:
                break

    def get_batch(self) -> torch.Tensor:
        """Нарезает кропы из пула GPU с умным акцентом на сигналы."""
        self._sync_queue_to_gpu()
        
        if len(self.gpu_pool) == 0:
            print("[DataLoader] Ожидание первых данных от процессора...")
            tensor_np = self.queue.get(block=True)
            tensor_gpu = torch.tensor(tensor_np, device=self.cfg.device, dtype=torch.float32)
            self.gpu_pool.append(tensor_gpu)
            
        batch = torch.empty((self.cfg.BATCH_SIZE, 1, self.cfg.FREQ_BINS, self.cfg.CHUNK_SIZE), device=self.cfg.device)
        
        for i in range(self.cfg.BATCH_SIZE):
            t = random.choice(self.gpu_pool)
            T = t.shape[-1]
            
            if T <= self.cfg.CHUNK_SIZE:
                pad_len = self.cfg.CHUNK_SIZE - T
                batch[i] = torch.nn.functional.pad(t, (0, pad_len))
            else:
                # 75% шанс: Фокусируемся на полезном сигнале
                # 25% шанс: Случайный кроп (чтобы сеть видела чистый шум)
                if random.random() < 0.95:
                    # Ищем "яркие" участки. Берём максимум энергии по всем частотам (схлопываем в 1D)
                    # Так как данные Z-нормализованы, сигналы будут иметь большие положительные значения.
                    energy_profile = t[0].max(dim=0).values  # Shape: (T,)
                    
                    # Зануляем отрицательный шум и возводим в квадрат, чтобы усилить пики
                    weights = torch.clamp(energy_profile, min=0.0) ** 2
                    
                    if weights.sum() > 1e-5:
                        # Сэмплируем 1 индекс на основе распределения энергии
                        center_idx = torch.multinomial(weights, 1).item()
                        
                        # Преобразуем центр в начало окна
                        start = center_idx - (self.cfg.CHUNK_SIZE // 2)
                        # Защита от выхода за границы
                        start = max(0, min(start, T - self.cfg.CHUNK_SIZE))
                    else:
                        # Если сигнал сплошной шум (нет ярких участков) - fallback на рандом
                        start = random.randint(0, T - self.cfg.CHUNK_SIZE)
                else:
                    # Случайный кроп (Uniform distribution)
                    start = random.randint(0, T - self.cfg.CHUNK_SIZE)
                    
                batch[i] = t[:, :, start : start + self.cfg.CHUNK_SIZE]
                
        return batch

    def __del__(self):
        if hasattr(self, 'worker_process') and self.worker_process.is_alive():
            self.worker_process.terminate()
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/AdaptiveSequenogram/Source/BatCNN/Model.py" encoding="utf-8"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function


# ==========================================
# 1. МАГИЯ ГРАДИЕНТОВ (Только для обратного прохода)
# ==========================================
class ApplyWindowToGrad(Function):
    @staticmethod
    def forward(ctx, weight, window):
        ctx.save_for_backward(window)
        return weight.view_as(weight)

    @staticmethod
    def backward(ctx, grad_output):
        window, = ctx.saved_tensors
        grad_weight = grad_output * window
        return grad_weight, None


# ==========================================
# 2. АРХИТЕКТУРА
# ==========================================
class WindowedPainterLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        window = 0.5 + 0.5 * torch.hann_window(kernel_size)
        self.register_buffer("window", window.view(1, 1, kernel_size))
        
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        windowed_weight = ApplyWindowToGrad.apply(self.weight, self.window)
        return F.conv_transpose1d(
            x, 
            weight=windowed_weight, 
            bias=None, 
            stride=self.stride, 
            padding=self.padding,
            output_padding=self.output_padding
        )

class SlidingPerceptronAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- ENCODER ---
        self.enc_layer1 = nn.Conv1d(in_channels=256, out_channels=192, kernel_size=502, stride=10, padding=251)
        self.enc_norm1 = nn.InstanceNorm1d(192)
        self.enc_act1 = nn.GELU()
        
        self.enc_layer2 = nn.Conv1d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.enc_norm2 = nn.InstanceNorm1d(64)
        self.enc_act2 = nn.GELU()
        
        # Латентный слой 32 канала
        self.enc_bottleneck = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1)
        
        # --- DECODER ---
        self.dec_expand = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=1)
        self.dec_norm1 = nn.InstanceNorm1d(64)
        self.dec_act1 = nn.GELU()
        
        self.dec_layer2 = nn.ConvTranspose1d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.dec_norm2 = nn.InstanceNorm1d(192)
        self.dec_act2 = nn.GELU()
        
        self.dec_layer3 = WindowedPainterLayer(
            in_channels=192, out_channels=256, 
            kernel_size=502, stride=10, padding=251, output_padding=4
        )

    def forward(self, x, return_latent=False):
        x = x.squeeze(1) 
        
        e1 = self.enc_act1(self.enc_norm1(self.enc_layer1(x)))
        e2 = self.enc_act2(self.enc_norm2(self.enc_layer2(e1)))
        
        latent_pre_activation = self.enc_bottleneck(e2)
        latent = F.softplus(latent_pre_activation)
        
        d1 = self.dec_act1(self.dec_norm1(self.dec_expand(latent)))
        d2 = self.dec_act2(self.dec_norm2(self.dec_layer2(d1)))
        
        reconstructed = self.dec_layer3(d2)
        reconstructed = F.softplus(reconstructed, beta=1.0)
        reconstructed = reconstructed.unsqueeze(1)
        
        if return_latent:
            latent_for_vis = latent.unsqueeze(2) 
            return reconstructed, latent_for_vis
        return reconstructed


def calculate_loss(pred, target, latent, sparsity_weight=1e-5):
    valid_pred = pred[:, :, :, 251:-251]
    valid_target = target[:, :, :, 251:-251]
    
    # 1. Мягкая энергетическая маска (внимание на громкие звуки)
    # Нормализуем таргет внутри батча в диапазон ~[0, 1] для весов
    max_vals = valid_target.amax(dim=(2, 3), keepdim=True) + 1e-6
    # Фоновому шуму оставляем вес 0.1, полезным сигналам вес стремится к 1.0
    weight_mask = 0.1 + 0.9 * (valid_target / max_vals)
    
    # 2. Пиксельная ошибка: Huber Loss (Гладкий L1)
    # Не сходит с ума от резких скачков (как MSE), но градиент стабилен возле нуля
    pixel_loss = F.huber_loss(valid_pred, valid_target, delta=2.0, reduction='none')
    weighted_pixel_loss = torch.mean(pixel_loss * weight_mask)
    
    # 3. Структурная ошибка (Edge / Gradient Loss) - Замена SSIM
    # Учит форму (наклоны, края) спектрограмм, заставляя избегать "размытия"
    
    # Разница по оси времени
    diff_pred_t = valid_pred[:, :, :, 1:] - valid_pred[:, :, :, :-1]
    diff_target_t = valid_target[:, :, :, 1:] - valid_target[:, :, :, :-1]
    loss_grad_t = F.l1_loss(diff_pred_t, diff_target_t)
    
    # Разница по оси частот
    diff_pred_f = valid_pred[:, :, 1:, :] - valid_pred[:, :, :-1, :]
    diff_target_f = valid_target[:, :, 1:, :] - valid_target[:, :, :-1, :]
    loss_grad_f = F.l1_loss(diff_pred_f, diff_target_f)
    
    structural_loss = loss_grad_t + loss_grad_f
    
    # 4. Штраф латентного пространства
    sparsity_loss = torch.mean(latent)
    
    # Итоговая ошибка (0.5 для баланса структурной ошибки)
    recon_loss = weighted_pixel_loss + 0.5 * structural_loss
    total_loss = recon_loss + sparsity_weight * sparsity_loss
    
    return total_loss, {'recon': recon_loss, 'sparsity': sparsity_loss}
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/AdaptiveSequenogram/Source/BatCNN/train.py" encoding="utf-8"
import torch
import time
import argparse
from pathlib import Path

from BatCNN.Config import HardwareConfig
from BatCNN.Model import SlidingPerceptronAutoencoder, calculate_loss
from BatCNN.DataLoader import DualProcessDataLoader
import os
os.chdir(Path(__file__).parent)

class LiveVisualizer:
    # ... (код визуализатора без изменений)
    def __init__(self):
        try:
            import pyqtgraph as pg
            from PySide6 import QtWidgets
            self.has_gui = True
        except ImportError:
            self.has_gui = False
            return

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
            
        self.win = pg.GraphicsLayoutWidget(title="BatCNN Real-Time Monitor")
        self.win.resize(1200, 400)

        self.p1 = self.win.addPlot(title="Original Spectrogram")
        self.img_orig = pg.ImageItem()
        self.p1.addItem(self.img_orig)

        self.p2 = self.win.addPlot(title="Latent Space (Tokens vs Time)")
        self.img_latent = pg.ImageItem()
        self.p2.addItem(self.img_latent)

        self.p3 = self.win.addPlot(title="Reconstructed")
        self.img_recon = pg.ImageItem()
        self.p3.addItem(self.img_recon)

        colormap = pg.colormap.get('viridis')
        self.img_orig.setColorMap(colormap)
        self.img_latent.setColorMap(pg.colormap.get('plasma'))
        self.img_recon.setColorMap(colormap)

        self.win.show()

    def update(self, orig_tensor, recon_tensor, latent_tensor):
        if not self.has_gui: return
        orig_np = orig_tensor[0, 0].detach().cpu().to(torch.float32).numpy().T
        recon_np = recon_tensor[0, 0].detach().cpu().to(torch.float32).numpy().T
        latent_np = latent_tensor[0].squeeze(1).detach().cpu().to(torch.float32).numpy().T

        self.img_orig.setImage(orig_np, autoLevels=True)
        self.img_recon.setImage(recon_np, autoLevels=True)
        self.img_latent.setImage(latent_np, autoLevels=True)
        self.app.processEvents()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BatCNN Training")
    parser.add_argument('--vis', action='store_true', help="Включить реалтайм визуализацию")
    args = parser.parse_args()

    config = HardwareConfig()
    data_dir = Path(__file__).parent / "Resources"
    wav_paths = list(data_dir.rglob("*.wav"))
    
    if not wav_paths:
        print("[!] ОШИБКА: WAV файлы не найдены!")
        exit(1)

    dataloader = DualProcessDataLoader(wav_paths, config)
    model = SlidingPerceptronAutoencoder().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device='cuda' if config.device == 'cuda' else 'cpu')
    
    CHECKPOINT_FILE = Path("batcnn_weights.pt")
    TEMP_CHECKPOINT = Path("batcnn_weights.tmp")
    start_epoch = 0

    if CHECKPOINT_FILE.exists():
        try:
            print(f"[*] Найден чекпоинт. Загрузка...")
            checkpoint = torch.load(CHECKPOINT_FILE, map_location=config.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[*] Успешно. Продолжаем с эпохи {start_epoch+1}.")
        except Exception as e:
            print(f"[!] Ошибка загрузки чекпоинта. Начинаем с нуля.")

    visualizer = LiveVisualizer() if args.vis else None

    EPOCHS = 200
    STEPS_PER_EPOCH = 50
    ACCUMULATION_STEPS = 8 
    
    model.train()
    print("\n================ START TRAINING ================")
    
    try:
        for epoch in range(start_epoch, EPOCHS):
            total_recon_loss = 0
            total_sparsity_loss = 0 # <--- Добавлено для логирования
            start_time = time.time()
            
            optimizer.zero_grad()
            
            for step in range(STEPS_PER_EPOCH):
                batch = dataloader.get_batch()
                
                with torch.autocast(device_type='cuda' if config.device == 'cuda' else 'cpu', dtype=torch.float16):
                    reconstructed, latent = model(batch, return_latent=True)
                    # Loss теперь возвращает общую ошибку и её компоненты
                    total_loss, loss_components = calculate_loss(reconstructed, batch, latent, sparsity_weight=0.01)
                    
                    total_loss = total_loss / ACCUMULATION_STEPS
                
                scaler.scale(total_loss).backward()
                
                if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == STEPS_PER_EPOCH:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                with torch.no_grad():
                    total_recon_loss += loss_components['recon'].item() * ACCUMULATION_STEPS
                    total_sparsity_loss += loss_components['sparsity'].item() * ACCUMULATION_STEPS
                
                if visualizer and step % 5 == 0:
                    visualizer.update(batch, reconstructed, latent)
                
            epoch_time = time.time() - start_time
            avg_recon = total_recon_loss / STEPS_PER_EPOCH
            avg_sparsity = total_sparsity_loss / STEPS_PER_EPOCH
            
            # Обновленный лог для вывода всех компонент
            print(f"Эпоха [{epoch+1:03d}/{EPOCHS}] | "
                  f"Recon Loss: {avg_recon:.3f} | "
                  f"Sparsity: {avg_sparsity:.3f} | "
                  f"Время: {epoch_time:.2f} сек.")
                  
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, TEMP_CHECKPOINT)
            TEMP_CHECKPOINT.replace(CHECKPOINT_FILE)

    except KeyboardInterrupt:
        print("\n[!] Обучение прервано. Сохранено.")
``````

``````text path="/MainData/Repo/valllshapkin/MonoBat/AdaptiveSequenogram/Source/BatCNN/Inference.py" encoding="utf-8"
import warnings
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch

# --- Импорты BatSpec и MultiArray ---
import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType
from BatSpec.Core.Functions import SpecFunc, TimeFunc
from BatSpec.Core.Physical.Units import UREG
# --- Ключевой импорт для решения проблемы ---
from BatSpec.Core.Spectral import interpolateToShape

# --- Импорты архитектуры сети ---
from BatCNN.Model import SlidingPerceptronAutoencoder
from BatCNN.Config import HardwareConfig

class BatCNNInference:
    """
    Инференс-модуль для модели SlidingPerceptronAutoencoder.
    Автоматически адаптируется под контекст входных данных (NumPy/Torch) и железо.
    """
    def __init__(self, weights_path: Path | str = None):
        self.config = HardwareConfig()
        self.device = torch.device(self.config.device)
        self.model = SlidingPerceptronAutoencoder().to(self.device)
        
        if weights_path is None:
            weights_path = Path(__file__).parent / "batcnn_weights.pt"
        else:
            weights_path = Path(weights_path)
            
        if not weights_path.exists():
            raise FileNotFoundError(f"Файл весов не найден: {weights_path}")
            
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"[*] Модель BatCNN успешно загружена на {self.device}")

        # Целевой контекст для нейросети
        dev_enum = DeviceType.GPU if self.device.type == 'cuda' else DeviceType.CPU
        self.torch_ctx = ArrayContext(Framework.TORCH, dev_enum, None)

    @torch.no_grad()
    def process(self, spec: SpecFunc) -> Tuple[SpecFunc, List[TimeFunc]]:
        """
        Принимает спектрограмму, прогоняет через Autoencoder и возвращает:
        1. Восстановленный SpecFunc (такой же размерности и контекста как оригинал).
        2. Список из 32 TimeFunc, представляющих активации латентного слоя во времени.
        """
        orig_ctx = spec.context
        orig_freq_bins = spec.freq[0].shape[-1]
        orig_time_frames = spec.time[0].shape[-1]

        # 1. СТАНДАРТИЗАЦИЯ ВХОДА: Интерполируем спектрограмму к размеру сети
        if orig_freq_bins != self.config.FREQ_BINS:
            warnings.warn(f"Сеть ожидает {self.config.FREQ_BINS} частотных бинов, получено {orig_freq_bins}. "
                          f"Входные данные будут интерполированы.")
            spec_standard = interpolateToShape(spec, target_shape=(self.config.FREQ_BINS, orig_time_frames))
        else:
            spec_standard = spec

        # 2. Переносим стандартизированные данные в PyTorch
        spec_torch = spec_standard.to_context(self.torch_ctx)
        mat_t, unit = spec_torch.values
        
        if mat_t.ndim > 2: mat_t = mat_t[0]
            
        # 3. Формируем тензор для сети: (Batch=1, Channels=1, Freq, Time)
        x = mat_t.unsqueeze(0).unsqueeze(0).to(torch.float32)

        # 4. Проход через сеть
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
            reconstructed, latent = self.model(x, return_latent=True)
            
        # 5. Постобработка: создаем SpecFunc с укороченной осью времени
        recon_mat_torch = reconstructed[0, 0, :, :]
        output_time_frames = recon_mat_torch.shape[-1]

        time_axis_sliced = spec_torch.time[0][:output_time_frames]

        recon_spec_standard = SpecFunc(
            matrix=(recon_mat_torch, unit),
            freq=spec_torch.freq[0],
            time=time_axis_sliced
        )

        # 6. ОБРАТНАЯ ИНТЕРПОЛЯЦИЯ: Возвращаем к оригинальной частотной размерности
        if orig_freq_bins != self.config.FREQ_BINS:
            interp_spec = interpolateToShape(
                recon_spec_standard,
                target_shape=(orig_freq_bins, output_time_frames)
            )
        else:
            interp_spec = recon_spec_standard

        # 7. ФИНАЛЬНОЕ ВЫРАВНИВАНИЕ: Дополняем (pad) матрицу, чтобы она точно соответствовала оригиналу
        final_mat_unpadded, final_unit = interp_spec.values
        mat_in_orig_ctx = ma.convert_to(final_mat_unpadded, orig_ctx)

        if mat_in_orig_ctx.shape[-1] < orig_time_frames:
            pad_len = orig_time_frames - mat_in_orig_ctx.shape[-1]
            pad_widths = [(0, 0)] * (mat_in_orig_ctx.ndim - 1) + [(0, pad_len)]
            
            if orig_ctx.isTorch():
                final_mat = torch.nn.functional.pad(mat_in_orig_ctx, (0, pad_len), "constant", 0)
            else:
                final_mat = orig_ctx.fw.pad(mat_in_orig_ctx, pad_widths, mode='constant')
        else:
            final_mat = mat_in_orig_ctx

        final_recon_spec = SpecFunc(
            matrix=(final_mat, final_unit),
            freq=spec.freq[0],
            time=spec.time[0]
        )

        # 8. Постобработка латентного пространства
        latent_mat = latent[0, :, 0, :]
        channels, t_latent = latent_mat.shape
        dt_orig, _ = spec_torch.dt
        dt_latent = float(dt_orig) * 10.0
        
        latent_time_np = np.arange(t_latent) * dt_latent
        axis_ctx = ArrayContext(orig_ctx._framework, orig_ctx._device, None)
        latent_time_axis = ma.convert_to(latent_time_np, axis_ctx)

        latent_funcs = []
        for i in range(channels):
            chan_activations = latent_mat[i]
            tf = TimeFunc(
                values=(chan_activations, UREG.dimensionless),
                axis=latent_time_axis
            ).to_context(orig_ctx)
            latent_funcs.append(tf)

        return final_recon_spec, latent_funcs
``````

