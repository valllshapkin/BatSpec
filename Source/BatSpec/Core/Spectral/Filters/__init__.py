"""
Модуль для аналитической генерации фильтров в частотной области.
Все функции являются кросс-фреймворковыми благодаря использованию MultiArray.
"""
import math
from typing import Tuple, Any, Union

import MultiArray as ma
import MultiArray.FFT as mfft
from MultiArray.Core import ArrayContext


def get_analytical_gaussian_filter(h: int, w: int, 
                                   sigma: Union[float, Tuple[float, float]],
                                   is_complex: bool, ctx: ArrayContext) -> Any:
    """
    Аналитически генерирует 2D Гауссовский фильтр в частотной области.
    """
    if isinstance(sigma, (float, int)):
        sigma_v, sigma_u = float(sigma), float(sigma)
    else:
        sigma_v, sigma_u = sigma
        
    v = mfft.fftfreq(h, d=1.0, ctx=ctx)
    if not is_complex:
        u = mfft.rfftfreq(w, d=1.0, ctx=ctx)
    else:
        u = mfft.fftfreq(w, d=1.0, ctx=ctx)

    v_part = ma.reshape(v, (h, 1))
    u_part = ma.reshape(u, (1, u.shape[0]))

    term_u = (sigma_u * u_part)**2
    term_v = (sigma_v * v_part)**2
    
    exponent = -2.0 * (math.pi**2) * (term_u + term_v)
    H_gauss = ma.exp(exponent)
    
    return H_gauss


def get_analytical_neg_laplacian_filter(h: int, w: int, is_complex: bool, ctx: ArrayContext) -> Any:
    """
    Аналитически генерирует частотный отклик дискретного отрицательного Лапласиана.
    """
    if not is_complex:
        u = mfft.rfftfreq(w, d=1.0, ctx=ctx)
        u_part = ma.reshape(u, (1, w // 2 + 1))
    else:
        u = mfft.fftfreq(w, d=1.0, ctx=ctx)
        u_part = ma.reshape(u, (1, w))
        
    v = mfft.fftfreq(h, d=1.0, ctx=ctx)
    v_part = ma.reshape(v, (h, 1))
    
    H_filter = 4.0 - 2.0 * ma.cos(2.0 * math.pi * u_part) - 2.0 * ma.cos(2.0 * math.pi * v_part)
    return H_filter


def get_analytical_directional_ridge_filter(h: int, w: int, 
                                            angle_deg: float, 
                                            sigma_len: float, 
                                            sigma_width: float, 
                                            is_complex: bool, ctx: ArrayContext,
                                            derivative_order: int = 2) -> Any:
    """
    Аналитически генерирует частотный отклик направленного детектора кривых (Ridge Filter).
    
    Args:
        derivative_order (int): Порядок производной поперек кривой.
            0: Просто Гауссовское размытие (направленный "блик").
            2: Вторая производная (Горб с канавками по краям - классика).
            4: Четвертая производная (Более острый горб с глубокими резкими канавками).
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    
    if not is_complex:
        u = ma.reshape(mfft.rfftfreq(w, d=1.0, ctx=ctx), (1, w // 2 + 1))
    else:
        u = ma.reshape(mfft.fftfreq(w, d=1.0, ctx=ctx), (1, w))
        
    v = ma.reshape(mfft.fftfreq(h, d=1.0, ctx=ctx), (h, 1))
    
    # Вращение сетки частот
    u_rot = u * cos_t + v * sin_t
    v_rot = -u * sin_t + v * cos_t
    
    # Базовая Гауссиана (основа фильтра)
    gauss_exp = -2.0 * math.pi**2 * ((sigma_len**2) * (u_rot**2) + (sigma_width**2) * (v_rot**2))
    H_gauss = ma.exp(gauss_exp)
    
    if derivative_order == 0:
        # Просто направленное сглаживание
        H_ridge = H_gauss
        
    elif derivative_order == 2:
        # Вторая производная (умножение на частоту в квадрате)
        H_ridge = (4.0 * math.pi**2 * (v_rot**2)) * H_gauss
        # Нормализация для сохранения амплитуды на пике
        if sigma_width > 1e-6:
            max_gain = 2.0 / ((sigma_width**2) * math.e)
            H_ridge = H_ridge * (1.0 / max_gain)
            
    elif derivative_order == 4:
        # Четвертая производная (умножение на частоту в 4-й степени)
        H_ridge = (16.0 * math.pi**4 * (v_rot**4)) * H_gauss
        # Нормализация аналитического экстремума
        if sigma_width > 1e-6:
            max_gain = 16.0 / ((sigma_width**4) * math.exp(2.0))
            H_ridge = H_ridge * (1.0 / max_gain)
            
    else:
        raise ValueError(f"Поддерживаются только derivative_order 0, 2 и 4. Получено {derivative_order}")
        
    return H_ridge