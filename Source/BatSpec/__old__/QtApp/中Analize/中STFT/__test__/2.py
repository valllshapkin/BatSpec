import matplotlib.pyplot as plt
import scipy.signal

from typing import Tuple
import tensorflow as tf
import numpy as np

def make_spec(
    signal: np.ndarray, 
    window_array: np.ndarray, 
    frame_step: int, 
    sample_rate: int  # Добавлен параметр для расчета времени и частоты
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    frame_length = len(window_array)

    def win_func(length, dtype=tf.float32):
        assert length == frame_length
        return tf.convert_to_tensor(window_array, dtype=dtype)

    # 1. Вычисляем STFT
    stft = tf.signal.stft(
        signal,
        frame_length=frame_length,
        frame_step=frame_step,
        window_fn=win_func 
    )

    # 2. Конвертация в numpy и получение амплитудной спектрограммы (Magnitude)
    # stft возвращает комплексные числа. Берем модуль (abs), чтобы получить амплитуду.
    # Транспонируем (.T), чтобы получить форму [частоты, время] — это стандарт для отрисовки.
    spectrogram = np.abs(stft.numpy()).T 

    # 3. Генерация оси частот (Y-axis) в Герцах
    # По умолчанию TF использует fft_length равный ближайшей степени 2 >= frame_length
    fft_length = 2 ** int(np.ceil(np.log2(frame_length)))
    
    # np.fft.rfftfreq идеально подходит, так как генерирует частоты для 
    # уникальных бинов (от 0 до Найквиста), что совпадает с выводом TF
    freq_axis = np.fft.rfftfreq(fft_length, d=1.0/sample_rate)

    # 4. Генерация оси времени (X-axis) в Секундах
    # stft.shape[0] - количество получившихся фреймов (окон)
    num_frames = stft.shape[0]
    
    # Время начала каждого фрейма
    time_axis = (np.arange(num_frames) * frame_step) / sample_rate

    return spectrogram, time_axis, freq_axis

# Создадим тестовый сигнал: 1 секунда, 16000 Гц (синус 1000 Гц + шум)
sr = 16000
t = np.linspace(0, 1, sr, endpoint=False)
test_signal = np.sin(2 * np.pi * 1000 * t) + np.random.normal(0, 0.5, sr)

# Окно Ханна длиной 512 (32 мс) и шаг 256 (16 мс)
win_length = 512
step = 256
window = scipy.signal.windows.hann(win_length)

# Получаем спектрограмму и оси
spec, t_axis, f_axis = make_spec(test_signal, window, step, sr)

# Переводим в децибелы (логарифмический масштаб) для лучшей видимости
spec_db = 20 * np.log10(np.maximum(spec, 1e-10))

# Отрисовка
plt.figure(figsize=(10, 4))
plt.pcolormesh(t_axis, f_axis, spec_db, shading='gouraud', cmap='magma')
plt.title('Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()