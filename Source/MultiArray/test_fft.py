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
