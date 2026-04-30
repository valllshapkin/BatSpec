from typing import Tuple, Union, Any, List

# --- Импорты из вашей экосистемы ---
from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.Spectral.Filters import (
    get_analytical_gaussian_filter, 
    get_analytical_neg_laplacian_filter, 
    get_analytical_directional_ridge_filter
)

# Внедряем новый фреймворк
import MultiArray as ma
from MultiArray.Core import ArrayContext
import MultiArray.FFT as mfft

# ==========================================
# ВНУТРЕННИЕ ХЕЛПЕРЫ
# ==========================================

def _is_complex(tensor: Any) -> bool:
    """Кросс-фреймворковая проверка на комплексный тип данных."""
    return 'complex' in str(getattr(tensor, 'dtype', type(tensor))).lower()

# ==========================================
# ФИЛЬТРЫ И СГЛАЖИВАНИЯ
# ==========================================

def negativeLaplacianSpec(spec: 'SpecFunc', sigma: Union[float, Tuple[float, float], None] = None) -> 'SpecFunc':
    """
    Вычисляет отрицательный Лапласиан спектрограммы (выделение краев, локальных максимумов).
    """
    if sigma is not None and isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))
        
    _, current_unit = spec.values
    ctx = spec.context

    def apply_laplacian(mat: Any) -> Any:
        h, w = mat.shape[-2], mat.shape[-1]
        is_cplx = _is_complex(mat)
        
        H_filter = get_analytical_neg_laplacian_filter(h, w, is_cplx, ctx)
        
        if sigma is not None:
            H_gauss = get_analytical_gaussian_filter(h, w, sigma, is_complex=is_cplx, ctx=ctx)
            H_filter = H_filter * H_gauss
        
        if is_cplx:
            fft_mat = mfft.fft2(mat, axes=(-2, -1))
            res = mfft.ifft2(fft_mat * H_filter, axes=(-2, -1))
        else:
            fft_mat = mfft.rfft2(mat, axes=(-2, -1))
            res = mfft.irfft2(fft_mat * H_filter, s=(h, w), axes=(-2, -1))
            
        return res

    return spec.cloneApply(func=apply_laplacian, new_unit=current_unit)

from typing import Any
import MultiArray as ma
from MultiArray.Core import ArrayContext
from MultiArray import FFT as mfft

def _is_complex(tensor: Any) -> bool:
    """Хелпер для проверки комплексности тензора."""
    ctx = ArrayContext.from_array(tensor)
    if ctx.isTorch():
        return tensor.is_complex()
    elif ctx.isTensorflow():
        return tensor.dtype.is_complex
    else:
        import numpy as np
        return np.iscomplexobj(ma.to_numpy(tensor))


def curveEnhanceSpec(spec: 'SpecFunc', 
                     num_angles: int = 12,
                     sigma_len: float = 3.0, 
                     sigma_width: float = 1.0, 
                     blend: float = 1.0,
                     derivative_order: int = 2) -> 'SpecFunc':
    """
    Выделяет на спектрограмме кривые (свисты, щелчки) произвольной формы.
    Оптимизировано через Batched FFT.
    """
    _, current_unit = spec.values
    ctx = spec.context

    def apply_curve_enhancement(mat: Any) -> Any:
        h, w = mat.shape[-2], mat.shape[-1]
        
        # Хелпер проверки комплексности прямо здесь (чтобы не тянуть внешние)
        ctx_mat = ArrayContext.from_array(mat)
        if ctx_mat.isTorch():
            is_cplx = mat.is_complex()
        elif ctx_mat.isTensorflow():
            is_cplx = mat.dtype.is_complex
        else:
            import numpy as np
            is_cplx = np.iscomplexobj(ma.to_numpy(mat))
        
        # 1. Прямое FFT
        if is_cplx:
            fft_mat = mfft.fft2(mat, axes=(-2, -1))
        else:
            fft_mat = mfft.rfft2(mat, axes=(-2, -1))

        # ИСПРАВЛЕНИЕ РАЗМЕРНОСТИ: берем размеры матрицы уже ПОСЛЕ Фурье (у rfft ширина обрезается)
        filter_spatial_shape = fft_mat.shape[-2:]
        target_shape = (1,) * (fft_mat.ndim - 2) + filter_spatial_shape

        angles = [i * (180.0 / num_angles) for i in range(num_angles)]
        filters_list = []

        # 2. Генерируем фильтры
        for angle_deg in angles:
            H_filter = get_analytical_directional_ridge_filter(
                h, w, angle_deg, sigma_len, sigma_width, is_cplx, ctx, derivative_order
            )
            
            # Подгоняем размерности под fft_mat и добавляем ось (1) для конкатенации
            H_filter_reshaped = ma.reshape(H_filter, (1,) + target_shape)
            filters_list.append(H_filter_reshaped)

        # 3. Собираем все фильтры в один батч
        filters_tensor = ma.concatenate(filters_list, axis=0)

        # 4. Умножаем матрицу разом на все фильтры
        fft_mat_expanded = ma.reshape(fft_mat, (1,) + fft_mat.shape)
        fft_filtered_batched = fft_mat_expanded * filters_tensor

        # 5. Одно обратное IFFT для всех углов сразу
        if is_cplx:
            all_responses = mfft.ifft2(fft_filtered_batched, axes=(-2, -1))
        else:
            all_responses = mfft.irfft2(fft_filtered_batched, s=(h, w), axes=(-2, -1))
            
        # 6. Векторизованный Max Pooling по нулевой оси
        if is_cplx:
            abs_responses = ma.abs(all_responses)
            max_abs = ma.max(abs_responses, axis=0, keepdims=True)
            
            mask = (abs_responses == max_abs)
            ctx_float = ArrayContext(ctx._framework, ctx._device, abs_responses.dtype)
            mask_float = ma.convert_to(mask, ctx_float)
            
            mask_float = mask_float / ma.clamp_min(ma.sum(mask_float, axis=0, keepdims=True), 1.0)
            max_response = ma.sum(all_responses * mask_float, axis=0)
        else:
            max_response = ma.max(all_responses, axis=0)

        if blend < 1.0:
            final_result = (1.0 - blend) * mat + blend * max_response
        else:
            final_result = max_response
            
        return final_result

    return spec.cloneApply(func=apply_curve_enhancement, new_unit=current_unit)