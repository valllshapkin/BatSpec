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