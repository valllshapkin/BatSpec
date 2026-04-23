from pathlib import Path

from BatSpec.Core.Record import correctDC, loadRecord, resampleRecord, trimRecord
ScriptDir = Path(__file__).parent

import torch
device = torch.device('cuda:0')

# 1. Загрузили (под капотом вернется быстрый NumPy на CPU)
signal = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")

# 2. Можем порезать или отфильтровать прямо в NumPy 
# (это не забивает память GPU лишними сырыми данными!)
signal_trimmed = trimRecord(signal, t_start=1.5)
signal_dc = correctDC(signal_trimmed)

# 3. А вот теперь, когда данные готовы, отправляем их на GPU!
signal_gpu = signal_dc.to_framework(torch, device=device)

# 4. Если вызовем resampleRecord для GPU тензора, он "под капотом" 
# сходит в CPU, поменяет частоту и САМ вернет тензор на `cuda:0`!
signal_resampled_gpu = resampleRecord(signal_gpu, new_sr=16000)

print(type(signal_resampled_gpu.values[0])) # <class 'torch.Tensor'>
print(signal_resampled_gpu.values[0].device) # cuda:0