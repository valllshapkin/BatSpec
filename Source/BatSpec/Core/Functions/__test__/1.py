import numpy as np

from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.Physical.Units import UREG

# Пытаемся импортировать torch для тестов, если он установлен
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False
    print("PyTorch не установлен. Тесты конвертации будут пропущены.")

print("=== ЗАПУСК ТЕСТОВ ===")

# 1. Генерируем тестовые данные на NumPy
batch_size = 3
num_freqs = 5
num_times = 10

freq_np = np.linspace(10, 50, num_freqs)       # Ось частот
time_np = np.linspace(0.0, 1.0, num_times)     # Ось времени

# 2D матрица: [batch, freq, time]
matrix_np = np.random.rand(batch_size, num_freqs, num_times) 

# Допустим, наши значения измеряются в Вольтах
# Используем hasattr на случай, если в твоем UREG нет volt
dummy_unit = UREG.volt if hasattr(UREG, 'volt') else UREG.dimensionless

# --- ТЕСТ 1: Инициализация и базовые методы (NumPy) ---
print("\n--- Тест 1: Инициализация SpecFunc (NumPy) ---")
spec_np = SpecFunc(matrix=(matrix_np, dummy_unit), freq=freq_np, time=time_np)
print(f"Тип матрицы: {type(spec_np.values[0])}")
print(f"Размерность матрицы: {spec_np.values[0].shape}")
print(f"Шаг по времени (dt): {spec_np.dt[0]:.4f} {spec_np.dt[1]}")
print(f"Шаг по частоте (df): {spec_np.df[0]:.4f} {spec_np.df[1]}")

# --- ТЕСТ 2: Фреймворк-агностичное интегрирование (NumPy) ---
print("\n--- Тест 2: Интегрирование (NumPy) ---")
integrated_time_np = spec_np.integrateOverFreq()
print(f"Результат integrateOverFreq -> Тип: {type(integrated_time_np).__name__}")
print(f"Размерность после интеграла: {integrated_time_np.values[0].shape}")

# --- ТЕСТ 3: Конвертация фреймворков (NumPy <-> PyTorch) ---
if has_torch:
    print("\n--- Тест 3: Конвертация в PyTorch ---")
    
    # Конвертируем весь объект SpecFunc в PyTorch
    spec_torch = spec_np.to_framework('torch')
    print(f"Успешная конвертация в {type(spec_torch).__name__}")
    print(f"Тип матрицы внутри: {type(spec_torch.values[0])}")
    
    # Проверяем, работает ли интегрирование для Torch-версии объекта
    print("\n--- Тест 4: Интегрирование (PyTorch) ---")
    integrated_time_torch = spec_torch.integrateOverTime()
    print(f"Результат integrateOverTime -> Тип: {type(integrated_time_torch).__name__}")
    print(f"Тип матрицы результата: {type(integrated_time_torch.values[0])}")
    print(f"Размерность после интеграла: {integrated_time_torch.values[0].shape}")

    # Конвертируем обратно в NumPy
    print("\n--- Тест 5: Обратная конвертация (PyTorch -> NumPy) ---")
    spec_back_np = spec_torch.to_framework('numpy')
    print(f"Тип матрицы после возврата: {type(spec_back_np.values[0])}")

# --- ТЕСТ 6: Срезы (Slicing) и Итерация ---
print("\n--- Тест 6: Магические методы (__getitem__, __iter__) ---")
single_spec = spec_np[0] # Берем первый элемент из батча
print(f"Размерность после среза spec_np[0]: {single_spec.values[0].shape}")

print("Итерация по батчу:")
for i, item in enumerate(spec_np):
    print(f"  Элемент {i} имеет форму {item.values[0].shape}")
    
print("\n=== ТЕСТЫ УСПЕШНО ЗАВЕРШЕНЫ ===")