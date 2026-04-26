
import math

def _get_analytical_gaussian_filter(h: int, w: int, sigma: Tuple[float, float], 
                                    device: Any, dtype: Any, is_complex: bool) -> Any:
    """Аналитически генерирует частотный отклик Гауссова фильтра."""
    import torch
    
    # Частоты по оси X (ширина/время) - циклов на пиксель
    if not is_complex:
        u = torch.fft.rfftfreq(w, d=1.0, device=device) # Только положительные частоты
    else:
        u = torch.fft.fftfreq(w, d=1.0, device=device)  # Все частоты
        
    # Частоты по оси Y (высота/частота)
    v = torch.fft.fftfreq(h, d=1.0, device=device)
    
    # Формула Фурье-образа Гауссианы: H(v, u) = exp(-2 * pi^2 * (sigma_y^2 * v^2 + sigma_x^2 * u^2))
    v_part = (sigma[0] ** 2) * (v ** 2)
    u_part = (sigma[1] ** 2) * (u ** 2)
    
    v_part = v_part.unsqueeze(-1) # Форма (h, 1)
    u_part = u_part.unsqueeze(0)  # Форма (1, w_rfft)
    
    exponent = -2.0 * math.pi**2 * (v_part + u_part)
    return torch.exp(exponent).to(dtype)


def blurSpec(spec: 'SpecFunc', sigma: Union[float, Tuple[float, float]] = 1.0) -> 'SpecFunc':
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))
    _, current_unit = spec.values

    def apply_blur(mat: Any) -> Any:
        fw_target, dev_target = _get_fw_info(mat)
        if fw_target != 'torch': mat = convert_to_framework(mat, 'torch')
        import torch
        
        device = mat.device
        dtype = mat.real.dtype if mat.is_complex() else mat.dtype
        h, w = mat.shape[-2], mat.shape[-1]
        is_cplx = mat.is_complex()
        
        # Сразу получаем идеальный частотный фильтр (без FFT окна!)
        H_filter = _get_analytical_gaussian_filter(h, w, sigma, device, dtype, is_cplx)
        
        if is_cplx:
            fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
            res = torch.fft.ifft2(fft_mat * H_filter, dim=(-2, -1))
        else:
            fft_mat = torch.fft.rfft2(mat, dim=(-2, -1))
            res = torch.fft.irfft2(fft_mat * H_filter, s=(h, w), dim=(-2, -1))
            
        if fw_target != 'torch': res = convert_to_framework(res, fw_target, dev_target)
        return res
    return spec.cloneApply(func=apply_blur, new_unit=current_unit)


def highPassSpec(spec: 'SpecFunc', sigma: Union[float, Tuple[float, float]] = 1.0) -> 'SpecFunc':
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))
    _, current_unit = spec.values

    def apply_hpf(mat: Any) -> Any:
        fw_target, dev_target = _get_fw_info(mat)
        if fw_target != 'torch': mat = convert_to_framework(mat, 'torch')
        import torch
        
        device = mat.device
        dtype = mat.real.dtype if mat.is_complex() else mat.dtype
        h, w = mat.shape[-2], mat.shape[-1]
        is_cplx = mat.is_complex()
        
        H_filter = _get_analytical_gaussian_filter(h, w, sigma, device, dtype, is_cplx)
        
        if is_cplx:
            fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
            res = torch.fft.ifft2(fft_mat * (1.0 - H_filter), dim=(-2, -1))
        else:
            fft_mat = torch.fft.rfft2(mat, dim=(-2, -1))
            res = torch.fft.irfft2(fft_mat * (1.0 - H_filter), s=(h, w), dim=(-2, -1))
            
        if fw_target != 'torch': res = convert_to_framework(res, fw_target, dev_target)
        return res
    return spec.cloneApply(func=apply_hpf, new_unit=current_unit)


def sharpenSpec(spec: 'SpecFunc', sigma: Union[float, Tuple[float, float]] = 1.0, amount: float = 1.0) -> 'SpecFunc':
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))
    _, current_unit = spec.values

    def apply_sharpen(mat: Any) -> Any:
        fw_target, dev_target = _get_fw_info(mat)
        if fw_target != 'torch': mat = convert_to_framework(mat, 'torch')
        import torch
        
        device = mat.device
        dtype = mat.real.dtype if mat.is_complex() else mat.dtype
        h, w = mat.shape[-2], mat.shape[-1]
        is_cplx = mat.is_complex()
        
        H_filter = _get_analytical_gaussian_filter(h, w, sigma, device, dtype, is_cplx)
        H_sharpen = 1.0 + amount * (1.0 - H_filter)
        
        if is_cplx:
            fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
            res = torch.fft.ifft2(fft_mat * H_sharpen, dim=(-2, -1))
        else:
            fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
            res = torch.fft.ifft2(fft_mat * H_sharpen, s=(h, w), dim=(-2, -1))
            
        if fw_target != 'torch': res = convert_to_framework(res, fw_target, dev_target)
        return res
    return spec.cloneApply(func=apply_sharpen, new_unit=current_unit)

