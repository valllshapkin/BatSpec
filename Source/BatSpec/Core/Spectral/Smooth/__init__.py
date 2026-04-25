from typing import Tuple, Union, Any, List
import math
import torch
# --- Импорты из вашей экосистемы (предполагается, что они доступны) ---
from BatSpec.Core.Physical.Arrays import convert_to_framework
from BatSpec.Core.Functions import SpecFunc
# Предполагаем, что эти хелперы импортированы из вашего модуля Spectral
from BatSpec.Core.Spectral import _get_fw_info, _get_analytical_gaussian_filter


def _get_analytical_neg_laplacian_filter(h: int, w: int, 
                                         device: Any, dtype: Any, is_complex: bool) -> Any:
    """
    Аналитически генерирует частотный отклик дискретного отрицательного Лапласиана.
    """
    if not is_complex:
        u = torch.fft.rfftfreq(w, d=1.0, device=device)
    else:
        u = torch.fft.fftfreq(w, d=1.0, device=device)
    v = torch.fft.fftfreq(h, d=1.0, device=device)
    v_part = v.unsqueeze(-1)
    u_part = u.unsqueeze(0)
    H_filter = 4.0 - 2.0 * torch.cos(2.0 * math.pi * u_part) - 2.0 * torch.cos(2.0 * math.pi * v_part)
    return H_filter.to(dtype)


def negativeLaplacianSpec(spec: 'SpecFunc', sigma: Union[float, Tuple[float, float], None] = None) -> 'SpecFunc':
    """
    Вычисляет отрицательный Лапласиан спектрограммы (выделение краев, локальных максимумов).
    """
    if sigma is not None and isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))
        
    _, current_unit = spec.values

    def apply_laplacian(mat: Any) -> Any:
        fw_target, dev_target = _get_fw_info(mat)
        if fw_target != 'torch': mat = convert_to_framework(mat, 'torch')
        
        device = mat.device
        dtype = mat.real.dtype if mat.is_complex() else mat.dtype
        h, w = mat.shape[-2], mat.shape[-1]
        is_cplx = mat.is_complex()
        
        H_filter = _get_analytical_neg_laplacian_filter(h, w, device, dtype, is_cplx)
        
        if sigma is not None:
            H_gauss = _get_analytical_gaussian_filter(h, w, sigma, device, dtype, is_cplx)
            H_filter = H_filter * H_gauss
        
        if is_cplx:
            fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
            res = torch.fft.ifft2(fft_mat * H_filter, dim=(-2, -1))
        else:
            fft_mat = torch.fft.rfft2(mat, dim=(-2, -1))
            res = torch.fft.irfft2(fft_mat * H_filter, s=(h, w), dim=(-2, -1))
            
        if fw_target != 'torch': res = convert_to_framework(res, fw_target, dev_target)
        return res

    return spec.cloneApply(func=apply_laplacian, new_unit=current_unit)


def _get_analytical_directional_ridge_filter(h: int, w: int, 
                                             angle_deg: float, 
                                             sigma_len: float, 
                                             sigma_width: float, 
                                             device: Any, dtype: Any, is_complex: bool) -> Any:
    """
    Аналитически генерирует частотный отклик направленного детектора кривых (Ridge Filter).
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    
    if not is_complex:
        u = torch.fft.rfftfreq(w, d=1.0, device=device).unsqueeze(0)
    else:
        u = torch.fft.fftfreq(w, d=1.0, device=device).unsqueeze(0)
        
    v = torch.fft.fftfreq(h, d=1.0, device=device).unsqueeze(-1)
    
    u_rot = u * cos_t + v * sin_t
    v_rot = -u * sin_t + v * cos_t
    
    gauss_exp = -2.0 * math.pi**2 * ((sigma_len**2) * (u_rot**2) + (sigma_width**2) * (v_rot**2))
    H_gauss = torch.exp(gauss_exp)
    
    H_ridge = (4.0 * math.pi**2 * (v_rot**2)) * H_gauss
    
    if sigma_width > 1e-6:
        max_gain = 1.0 / (2.0 * math.pi**2 * sigma_width**2 * math.e)
        H_ridge = H_ridge * (1.0 / max_gain)
        
    return H_ridge.to(dtype)


def curveEnhanceSpec(spec: 'SpecFunc', 
                     num_angles: int = 12,
                     sigma_len: float = 3.0, 
                     sigma_width: float = 1.0, 
                     blend: float = 1.0) -> 'SpecFunc':
    """
    Выделяет на спектрограмме кривые (свисты, щелчки) произвольной формы.
    """
    _, current_unit = spec.values

    def apply_curve_enhancement(mat: Any) -> Any:
        fw_target, dev_target = _get_fw_info(mat)
        if fw_target != 'torch': 
            mat = convert_to_framework(mat, 'torch')
        
        device = mat.device
        dtype = mat.real.dtype if mat.is_complex() else mat.dtype
        h, w = mat.shape[-2], mat.shape[-1]
        is_cplx = mat.is_complex()
        
        if is_cplx:
            fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
        else:
            fft_mat = torch.fft.rfft2(mat, dim=(-2, -1))

        angle_responses = []
        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
        angles = torch.linspace(0, 180, num_angles, endpoint=False, device=device)

        for angle_deg in angles:
            H_filter = _get_analytical_directional_ridge_filter(
                h, w, angle_deg.item(), sigma_len, sigma_width, device, dtype, is_cplx
            )
            
            fft_filtered = fft_mat * H_filter
            if is_cplx:
                response = torch.fft.ifft2(fft_filtered, dim=(-2, -1))
            else:
                response = torch.fft.irfft2(fft_filtered, s=(h, w), dim=(-2, -1))
            
            angle_responses.append(response)

        all_responses = torch.stack(angle_responses, dim=0)
        
        if is_cplx:
            abs_responses = torch.abs(all_responses)
            max_indices = torch.argmax(abs_responses, dim=0, keepdim=True)
            # Индексы max_indices должны иметь тот же размер, что и all_responses, кроме dim=0
            # для gather. Нам нужно расширить его для комплексного измерения.
            # Но т.к. gather работает с float, проще сделать так:
            max_response = all_responses.gather(0, max_indices).squeeze(0)
        else:
            max_response, _ = torch.max(all_responses, dim=0)

        if blend < 1.0:
            final_result = (1.0 - blend) * mat + blend * max_response
        else:
            final_result = max_response
            
        if fw_target != 'torch': 
            final_result = convert_to_framework(final_result, fw_target, dev_target)
            
        return final_result

    return spec.cloneApply(func=apply_curve_enhancement, new_unit=current_unit)