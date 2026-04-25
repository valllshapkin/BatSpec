from types import ModuleType
from typing import Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy
    import numpy.typing
    import torch
    import tensorflow

TensorLike = Union['numpy.typing.NDArray[Any]', 'torch.Tensor', 'tensorflow.Tensor', Any]

def convert_to_framework(
    data: TensorLike, 
    framework: Union[str, ModuleType], 
    device: str | None = None
) -> TensorLike:
    """
    Конвертирует массив/тензор в указанный фреймворк и переносит на нужный девайс.
    """
    
    # 1. Определяем целевой фреймворк
    if isinstance(framework, str):
        fw_name = framework.lower()
    else:
        fw_name = framework.__name__.lower()
        
    if fw_name in ['np', 'numpy']: fw_name = 'numpy'
    elif fw_name in ['tf', 'tensorflow']: fw_name = 'tensorflow'
    elif fw_name in ['torch', 'pytorch']: fw_name = 'torch'
    else: raise ValueError(f"Неизвестный фреймворк: {framework}")

    # 2. Определяем исходный фреймворк данных
    type_str = str(type(data)).lower()
    is_torch = 'torch.tensor' in type_str
    is_tf = 'tensorflow' in type_str and hasattr(data, 'numpy')

    # --- 3. БЫСТРЫЙ ПУТЬ (ОПТИМИЗАЦИЯ) ---
    # Если тензор уже в нужном фреймворке, просто меняем девайс (если нужно)
    if fw_name == 'torch' and is_torch:
        if device is not None:
            device_str = str(device).lower().replace('gpu', 'cuda')
            return data.to(device_str)
        return data
        
    if fw_name == 'tensorflow' and is_tf:
        if device is not None:
            import tensorflow as tf
            device_str = str(device).upper().replace('CUDA', 'GPU')
            if not device_str.startswith('/'):
                device_str = f'/{device_str}'
            with tf.device(device_str):
                return tf.identity(data)
        return data


    # --- 4. Предобработка: Кросс-фреймворковая конвертация ---
    # Извлекаем данные в numpy, ТОЛЬКО ЕСЛИ мы меняем фреймворк
    if is_torch:
        data = data.detach().cpu().numpy()
    elif is_tf:
        data = data.numpy()

    # --- 5. Создание нового тензора ---
    if fw_name == 'numpy':
        import numpy as np
        if device is not None and 'cpu' not in str(device).lower():
            import warnings
            warnings.warn(f"NumPy поддерживает только CPU. Аргумент device='{device}' проигнорирован.")
        return np.array(data)

    elif fw_name == 'torch':
        import torch
        tensor = torch.as_tensor(data)
        if device is not None:
            device_str = str(device).lower().replace('gpu', 'cuda')
            tensor = tensor.to(device_str)
        return tensor

    elif fw_name == 'tensorflow':
        import tensorflow as tf
        if device is not None:
            device_str = str(device).upper().replace('CUDA', 'GPU')
            if not device_str.startswith('/'):
                device_str = f'/{device_str}'
            with tf.device(device_str):
                return tf.convert_to_tensor(data)
        else:
            return tf.convert_to_tensor(data)