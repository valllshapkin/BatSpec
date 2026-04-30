from typing import Tuple, Optional, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import NMF
from tqdm.auto import tqdm

# --- Внедряем наш универсальный фреймворк ---
import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType

from BatSpec.Core.Functions import SpecFunc, TimeFunc
from BatSpec.Core.Physical.Units import UREG

# --- Подключаем мощное тензорное ядро ---
from BatSpec.Core.ConvFFT import tensor_match1d_time, tensor_match2d_ft


# =====================================================================
# БЛОК 1: NMF Декомпозиция
# =====================================================================

class _FFTSpecConvDecomposer(nn.Module):
    def __init__(self, num_patterns: int, num_freqs: int, win_width: int, seq_len: int, dtype: torch.dtype):
        super().__init__()
        self.num_patterns = num_patterns
        self.W = seq_len
        self.W_win = win_width
        self.basis = nn.Parameter(torch.abs(torch.randn(num_patterns, num_freqs, win_width, dtype=dtype)) * 0.1)
        self.seq = nn.Parameter(torch.abs(torch.randn(1, num_patterns, seq_len, dtype=dtype)) * 0.01)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        active_seq = F.relu(self.seq)
        fft_size = self.W + self.W_win - 1
        seq_f = torch.fft.rfft(active_seq, n=fft_size)
        basis_f = torch.fft.rfft(self.basis, n=fft_size)
        canvas_f = torch.einsum('b n f, n h f -> b h f', seq_f, basis_f)
        canvas = torch.fft.irfft(canvas_f, n=fft_size)
        return canvas[..., :self.W], active_seq


def extract_spec_patterns_with_callback(
    spec: SpecFunc,
    num_patterns: int = 6,
    num_learn: int = 12,
    window_duration_ms: float = 100.0,
    epochs: int = 2000,
    learning_rate: float = 0.05,
    lambda_l1: float = 0.15,
    lambda_ortho: float = 1.0,
    lambda_center: float = 0.1,  
    device: Optional[torch.device] = None,
    callback: Optional[Callable[[SpecFunc, TimeFunc, SpecFunc, SpecFunc], None]] = None,
    callback_interval_percent: Optional[float] = 1.0
) -> Tuple[SpecFunc, TimeFunc, SpecFunc, SpecFunc]:
    
    # --- Бесшовная миграция данных в PyTorch ---
    orig_ctx = spec.context
    pt_ctx = ArrayContext(Framework.TORCH, DeviceType.GPU if orig_ctx.isGPU() and torch.cuda.is_available() else DeviceType.CPU, None)
    spec_pt = spec.to_context(pt_ctx)
    
    S_tensor, S_unit = spec_pt.values
    freq_axis, _ = spec_pt.freq
    time_axis, _ = spec_pt.time
    
    device = S_tensor.device if device is None else device
    dtype = S_tensor.dtype
    target_canvas = S_tensor.unsqueeze(0) if S_tensor.dim() == 2 else S_tensor
    
    dt_val, _ = spec_pt.dt
    sr = 1.0 / float(dt_val)
    num_freqs, seq_len = target_canvas.shape[-2], target_canvas.shape[-1]
    win_width = max(3, int(round((window_duration_ms / 1000.0) * sr)))
    
    model = _FFTSpecConvDecomposer(num_learn, num_freqs, win_width, seq_len, dtype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    identity = torch.eye(num_learn, dtype=dtype, device=device)

    target_center = (win_width - 1) / 2.0
    time_indices = torch.arange(win_width, dtype=dtype, device=device)

    callback_epoch_interval = 0
    if callback and callback_interval_percent is not None and callback_interval_percent > 0 and epochs > 0:
        callback_epoch_interval = max(1, int(epochs * (callback_interval_percent / 100.0)))

    print(f"Запуск NMF разложения: {num_patterns} паттернов. Окно: {win_width} сэмплов.")
        
    def _post_process_and_invoke_callback(
        current_model: _FFTSpecConvDecomposer, 
        reconstructed_canvas: torch.Tensor, 
        activations: torch.Tensor
    ):
        current_model.eval()
        with torch.no_grad():
            seq_np = activations[0].cpu().numpy()
            nmf = NMF(n_components=num_patterns, init='nndsvda', max_iter=500)
            nmf_projections = nmf.fit_transform(seq_np.T)
            nmf_timelines = torch.tensor(nmf_projections.T, dtype=dtype, device=device)

            nmf_weights = nmf.components_ 
            flat_bases_np = current_model.basis.view(num_learn, -1).cpu().numpy()
            nmf_bases_flat = np.dot(nmf_weights, flat_bases_np)
            nmf_bases = torch.tensor(nmf_bases_flat.reshape(num_patterns, num_freqs, win_width), dtype=dtype, device=device)

            fft_size = seq_len + win_width - 1
            time_f = torch.fft.rfft(nmf_timelines, n=fft_size)  
            basis_f = torch.fft.rfft(nmf_bases, n=fft_size)     
            canvas_components_f = torch.einsum('n f, n h f -> n h f', time_f, basis_f)
            components_canvas = torch.fft.irfft(canvas_components_f, n=fft_size)[..., :seq_len]
            total_reconstructed = components_canvas.sum(dim=0) 
            
            basis_time_axis = torch.arange(win_width, dtype=dtype, device=device) * float(dt_val)
            
            bases_spec = SpecFunc((nmf_bases, S_unit), freq_axis, basis_time_axis).to_context(orig_ctx)
            timelines_func = TimeFunc((nmf_timelines, UREG.dimensionless), time_axis).to_context(orig_ctx)
            reconstructed_spec = SpecFunc((total_reconstructed, S_unit), freq_axis, time_axis).to_context(orig_ctx)
            components_spec = SpecFunc((components_canvas, S_unit), freq_axis, time_axis).to_context(orig_ctx)
            
            if callback:
                callback(bases_spec, timelines_func, reconstructed_spec, components_spec)

        current_model.train()
        return bases_spec, timelines_func, reconstructed_spec, components_spec
            
    pbar = tqdm(range(epochs), desc="Оптимизация базисов")
    latest_results = None

    for epoch in pbar:
        optimizer.zero_grad()
        
        with torch.no_grad():
            model.basis.data.clamp_(min=0.0)
            norms = torch.norm(model.basis, p=2, dim=(1, 2), keepdim=True)
            model.basis.data.div_(norms + 1e-8)

        pred_canvas, active_seq = model.forward()
        
        loss_recon = F.mse_loss(pred_canvas, target_canvas)
        seq_l2_norms = torch.sqrt(torch.sum(active_seq**2, dim=1) + 1e-8)
        loss_sparse = lambda_l1 * torch.mean(seq_l2_norms)
        
        flat_basis = model.basis.view(num_learn, -1)
        gram_matrix = torch.mm(flat_basis, flat_basis.T)
        loss_ortho = lambda_ortho * F.mse_loss(gram_matrix, identity)

        density = model.basis.sum(dim=1) 
        mass = density.sum(dim=1) + 1e-8
        com = (density * time_indices).sum(dim=1) / mass
        loss_center = lambda_center * F.mse_loss(com, torch.full_like(com, target_center))
        
        loss = loss_recon + loss_sparse + loss_ortho + loss_center
        loss.backward()
        optimizer.step()

        pbar.set_postfix(Loss=f"{loss.item():.3f}", Rec=f"{loss_recon.item():.3f}", Cntr=f"{loss_center.item():.3f}")

        is_last_epoch = (epoch == epochs - 1)
        if callback_epoch_interval > 0 and ((epoch + 1) % callback_epoch_interval == 0 or is_last_epoch):
            latest_results = _post_process_and_invoke_callback(model, pred_canvas, active_seq)

    if latest_results is None:
        latest_results = _post_process_and_invoke_callback(model, *model.forward())

    return latest_results


# =====================================================================
# БЛОК 2: ZNCC Pattern Discovery (через ядро ConvFFT)
# =====================================================================

def _total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    tv_h = torch.sum(torch.abs(img[..., :-1, :] - img[..., 1:, :]))
    tv_w = torch.sum(torch.abs(img[..., :, :-1] - img[..., :, 1:]))
    return tv_h + tv_w

def discover_best_pattern_with_callback(
    spec: SpecFunc,
    window_duration_ms: float = 100.0,
    anchor_percent: float = 50.0,
    epochs: int = 500,
    learning_rate: float = 0.05,
    top_k_peaks: int = 20,
    lambda_tv: float = 0.01,
    lambda_center: float = 0.1,
    mode: str = '1d',  # '1d' для TimeFunc отклика, '2d' для SpecFunc отклика
    device: Optional[torch.device] = None,
    callback: Optional[Callable] = None,
    callback_interval_percent: Optional[float] = 2.0
) -> Tuple[SpecFunc, Union[TimeFunc, SpecFunc]]:
    
    # --- 1. Подготовка данных ---
    orig_ctx = spec.context
    pt_ctx = ArrayContext(Framework.TORCH, DeviceType.GPU if orig_ctx.isGPU() and torch.cuda.is_available() else DeviceType.CPU, None)
    spec_pt = spec.to_context(pt_ctx)
    
    S, S_unit = spec_pt.values
    freq_axis, _ = spec_pt.freq
    time_axis, _ = spec_pt.time
    
    device = S.device if device is None else device
    dtype = S.dtype
    target_spec = S.unsqueeze(0) if S.dim() == 2 else S
    
    dt_val, _ = spec_pt.dt
    sr = 1.0 / float(dt_val)
    num_freqs, _ = target_spec.shape[-2], target_spec.shape[-1]
    win_width = max(3, int(round((window_duration_ms / 1000.0) * sr)))
    
    W_raw = nn.Parameter(torch.randn(1, num_freqs, win_width, device=device, dtype=dtype))
    optimizer = torch.optim.Adam([W_raw], lr=learning_rate)
    
    target_center_index = (win_width - 1) * (anchor_percent / 100.0)
    time_indices = torch.arange(win_width, device=device, dtype=dtype)
    
    print(f"Поиск паттерна: окно {win_width} сэмплов. Целевой центр: {target_center_index:.1f}")
    
    callback_epoch_interval = 0
    if callback and callback_interval_percent is not None and callback_interval_percent > 0:
        callback_epoch_interval = max(1, int(epochs * (callback_interval_percent / 100.0)))

    def _invoke_callback(current_W_raw: torch.Tensor, zncc_map_tensor: torch.Tensor):
        with torch.no_grad():
            W_norm = current_W_raw - current_W_raw.mean()
            W_norm /= (torch.norm(W_norm) + 1e-8)
            
            pattern_time_axis = torch.arange(win_width, device=device, dtype=dtype) * float(dt_val)
            found_pattern_spec = SpecFunc((W_norm.squeeze(0), S_unit), freq_axis, pattern_time_axis).to_context(orig_ctx)
            
            if mode == '1d':
                corr_func = TimeFunc((zncc_map_tensor.squeeze(0), UREG.dimensionless), time_axis).to_context(orig_ctx)
            else:
                corr_func = SpecFunc((zncc_map_tensor.squeeze(0), UREG.dimensionless), freq_axis, time_axis).to_context(orig_ctx)
            
            if callback:
                callback(found_pattern_spec, corr_func)
        return found_pattern_spec, corr_func

    # --- 2. Цикл обучения ---
    pbar = tqdm(range(epochs), desc=f"Поиск {mode}-паттерна")
    latest_results = None
    eps = 1e-8

    for epoch in pbar:
        optimizer.zero_grad()
        
        W_zero_mean = W_raw - W_raw.mean()
        W = W_zero_mean / (torch.norm(W_zero_mean) + eps)
        
        # Дифференцируемый ZNCC через математическое ядро ConvFFT
        if mode == '1d':
            numerator = tensor_match1d_time(target_spec, W)
            U = torch.ones_like(W); N = W.numel()
            Sum_S = tensor_match1d_time(target_spec, U)
            Sum_S2 = tensor_match1d_time(target_spec**2, U)
            local_variance = (Sum_S2 - (Sum_S**2) / N).clamp_(min=0.0)
            denominator = torch.sqrt(local_variance) + eps
            zncc_map_full = numerator / denominator
            zncc_map = zncc_map_full.sum(dim=-2)  # Схлопываем по частоте
        else:
            numerator = tensor_match2d_ft(target_spec, W)
            U = torch.ones_like(W); N = W.numel()
            Sum_S = tensor_match2d_ft(target_spec, U)
            Sum_S2 = tensor_match2d_ft(target_spec**2, U)
            local_variance = (Sum_S2 - (Sum_S**2) / N).clamp_(min=0.0)
            denominator = torch.sqrt(local_variance) + eps
            zncc_map = numerator / denominator

        top_peaks, _ = torch.topk(zncc_map.flatten(), k=top_k_peaks)
        loss_match = -torch.mean(top_peaks) 
        loss_tv = lambda_tv * _total_variation_loss(W_raw)
        
        energy_density = W_raw**2
        time_density = energy_density.sum(dim=-2)
        total_mass = time_density.sum() + eps
        current_com = (time_density * time_indices).sum() / total_mass
        
        loss_center = lambda_center * F.mse_loss(current_com, torch.full_like(current_com, target_center_index))
        
        total_loss = loss_match + loss_tv + loss_center
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(Loss=f"{total_loss.item():.3f}", Match=f"{-loss_match.item():.3f}", Cntr=f"{loss_center.item():.3f}")

        is_last_epoch = (epoch == epochs - 1)
        if callback_epoch_interval > 0 and ((epoch + 1) % callback_epoch_interval == 0 or is_last_epoch):
            latest_results = _invoke_callback(W_raw, zncc_map)

    if latest_results is None:
        with torch.no_grad():
            W_zero_mean = W_raw - W_raw.mean()
            W = W_zero_mean / (torch.norm(W_zero_mean) + eps)
            if mode == '1d':
                numerator = tensor_match1d_time(target_spec, W)
                U = torch.ones_like(W); N = W.numel()
                local_variance = (tensor_match1d_time(target_spec**2, U) - (tensor_match1d_time(target_spec, U)**2)/N).clamp_(min=0.0)
                final_zncc = (numerator / (torch.sqrt(local_variance) + eps)).sum(dim=-2)
            else:
                numerator = tensor_match2d_ft(target_spec, W)
                U = torch.ones_like(W); N = W.numel()
                local_variance = (tensor_match2d_ft(target_spec**2, U) - (tensor_match2d_ft(target_spec, U)**2)/N).clamp_(min=0.0)
                final_zncc = numerator / (torch.sqrt(local_variance) + eps)
            latest_results = _invoke_callback(W_raw, final_zncc)

    return latest_results


# =====================================================================
# ТЕСТИРОВАНИЕ ОБОИХ АЛГОРИТМОВ
# =====================================================================
if __name__ == "__main__":
    F_BINS, TIME_STEPS = 64, 2000
    spec_tensor = torch.randn(F_BINS, TIME_STEPS) * 0.1 

    pattern_width = 100
    y_coords = torch.arange(F_BINS).unsqueeze(1)
    x_coords = torch.arange(pattern_width).unsqueeze(0)
    center_freq = F_BINS / 2
    chirp = torch.exp(-((y_coords - (center_freq + 15 * torch.sin(x_coords * 0.2)))**2) / 10)
    
    spec_tensor[:, 200:200+pattern_width] += chirp * 5.0
    spec_tensor[:, 900:900+pattern_width] += chirp * 3.0
    spec_tensor[:, 1500:1500+pattern_width] += chirp * 8.0

    freq_axis = torch.arange(F_BINS)
    time_axis = torch.arange(TIME_STEPS) * 0.01 
    
    test_spec = SpecFunc((spec_tensor, UREG.dimensionless), freq_axis, time_axis)

    # --- ТЕСТ 1: ZNCC (1D Режим - Поиск паттерна по времени) ---
    def plot_zncc(found_pattern, correlation_func):
        try:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(15, 4))
            
            pattern_img, _ = found_pattern.values
            axs[0].imshow(pattern_img.numpy(), aspect='auto', origin='lower', cmap='viridis')
            axs[0].set_title('ZNCC: Найденный паттерн (W)')
            
            corr_data, _ = correlation_func.values
            if corr_data.dim() == 1:
                axs[1].plot(correlation_func.time[0].numpy(), corr_data.numpy())
                axs[1].set_title('1D Временная карта корреляций')
            else:
                axs[1].imshow(corr_data.numpy(), aspect='auto', origin='lower', cmap='coolwarm')
                axs[1].set_title('2D Карта корреляций ZNCC')
                
            plt.tight_layout()
            plt.show()
        except ImportError: pass

    print("\n--- Запуск ZNCC (1D Режим) ---")
    discover_best_pattern_with_callback(
        spec=test_spec,
        window_duration_ms=1000.0,
        mode='1d',
        epochs=100,
        learning_rate=0.05,
        callback=plot_zncc,
        callback_interval_percent=100.0
    )

    # --- ТЕСТ 2: NMF Разложение ---
    def plot_nmf(bases, timelines, rec, comp):
        try:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            axs[0].imshow(bases.values[0][0].numpy(), aspect='auto', origin='lower')
            axs[0].set_title('NMF Базис 1')
            axs[1].plot(timelines.time[0].numpy(), timelines.values[0][0].numpy())
            axs[1].set_title('NMF Активации 1')
            axs[2].imshow(rec.values[0].numpy(), aspect='auto', origin='lower')
            axs[2].set_title('NMF Реконструкция')
            plt.tight_layout()
            plt.show()
        except ImportError: pass

    print("\n--- Запуск NMF Разложения ---")
    extract_spec_patterns_with_callback(
        spec=test_spec,
        num_patterns=2,
        num_learn=4,
        window_duration_ms=1000.0,
        epochs=200,
        learning_rate=0.05,
        callback=plot_nmf,
        callback_interval_percent=100.0
    )