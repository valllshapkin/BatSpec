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
