import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

# =====================================================================
# 1. ФУНКЦИИ ИЗ ПРОШЛЫХ ШАГОВ (Тензорное ядро и NCC)
# =====================================================================
def tensor_conv2d_ft(sig_v, ker_v, anchor_f, anchor_t):
    F_sig, T_sig = sig_v.shape[-2], sig_v.shape[-1]; F_ker, T_ker = ker_v.shape[-2], ker_v.shape[-1]
    N_f, N_t = F_sig + F_ker - 1, T_sig + T_ker - 1
    Sig_f = torch.fft.rfft2(sig_v, s=(N_f, N_t), dim=(-2, -1)); Ker_f = torch.fft.rfft2(ker_v, s=(N_f, N_t), dim=(-2, -1))
    return torch.fft.irfft2(Sig_f * Ker_f, s=(N_f, N_t), dim=(-2, -1))[..., anchor_f:anchor_f+F_sig, anchor_t:anchor_t+T_sig]

def moving_sum_1d(x, window_size, anchor):
    pad = window_size - 1; x_pad = torch.nn.functional.pad(x, (pad, pad))
    cs = torch.zeros(x_pad.shape[-1] + 1, dtype=torch.float64, device=x.device); cs[1:] = torch.cumsum(x_pad.to(torch.float64), dim=-1)
    start_idx = torch.arange(x.shape[-1], device=x.device) + anchor; end_idx = start_idx + window_size
    return (cs[end_idx] - cs[start_idx]).to(x.dtype)

def fast_ncc_1d(W, S, anchor_f, anchor_t):
    H, W_W = W.shape; N = H * W_W
    # W и S должны требовать градиенты, PyTorch FFT поддерживает backprop!
    conv2d_out = tensor_conv2d_ft(S, torch.flip(W, dims=(-2, -1)), anchor_f, anchor_t)
    numerator = conv2d_out[(H - 1) - anchor_f, :] / N
    var_S = torch.clamp(moving_sum_1d((S**2).sum(dim=-2), W_W, anchor_t)/N - (moving_sum_1d(S.sum(dim=-2), W_W, anchor_t)/N)**2, min=1e-8)
    return numerator / torch.sqrt(var_S)

def get_masks(shape):
    H, W_W = shape; Y, X = torch.meshgrid(torch.arange(H), torch.arange(W_W), indexing='ij'); cy, cx = H // 2, W_W // 2
    R_c = min(H, W_W) * 0.4; M_circle = (((Y - cy)**2 + (X - cx)**2) <= R_c**2).float()
    R_s = min(H, W_W) * 0.35; M_square = ((torch.abs(Y - cy) <= R_s) & (torch.abs(X - cx) <= R_s)).float()
    return M_circle, M_square

# =====================================================================
# 2. ГЕНЕРАЦИЯ БОЛЬШОГО ДАТАСЕТА (Длинная спектрограмма)
# =====================================================================
torch.manual_seed(42)
H, W_W, W_S = 41, 41, 4000
N_pixels = H * W_W
anchor_f, anchor_t = (H - 1) // 2, (W_W - 1) // 2

M_circle, M_square = get_masks((H, W_W))
S_target = 2.0 + torch.rand(H, W_S) * 0.5  # Фон

# Расставляем 15 кругов и 15 квадратов, в основном одиночных
num_objects = 15
circle_positions = torch.randint(W_W, W_S - W_W, (num_objects,))
square_positions = torch.randint(W_W, W_S - W_W, (num_objects,))

for pos in circle_positions: S_target[:, pos - anchor_t : pos + (W_W - anchor_t)] += M_circle * torch.empty(1).uniform_(5.0, 9.0).item()
for pos in square_positions: S_target[:, pos - anchor_t : pos + (W_W - anchor_t)] += M_square * torch.empty(1).uniform_(5.0, 9.0).item()

# Переводим в float32 для градиентов
S_target = S_target.float()

# =====================================================================
# 3. МОДЕЛЬ ОБУЧЕНИЯ (Dictionary Learning)
# =====================================================================
class TemplateLearner(nn.Module):
    def __init__(self, H, W_W):
        super().__init__()
        # Инициализация: Вертикальные линии в центре + легкий шум для нарушения симметрии
        V1_init = torch.randn(H, W_W) * 0.01
        V1_init[:, W_W // 2 - 1 : W_W // 2 + 2] = 1.0  
        
        V2_init = torch.randn(H, W_W) * 0.01
        V2_init[:, W_W // 2 - 1 : W_W // 2 + 2] = 1.0  
        
        # Сырые веса, которые будет обновлять оптимизатор
        self.V1 = nn.Parameter(V1_init)
        self.V2 = nn.Parameter(V2_init)

    def get_normalized_W(self):
        """Дифференцируемая проекция сырых весов V на условия <W, B>=0 и <W, W>=1"""
        def normalize(V):
            V_centered = V - V.mean()
            return V_centered / torch.sqrt((V_centered**2).sum() / (H * W_W))
        return normalize(self.V1), normalize(self.V2)

# =====================================================================
# 4. ЦИКЛ ОБУЧЕНИЯ
# =====================================================================
model = TemplateLearner(H, W_W)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Гиперпараметры обучения
K_peaks = 7          # Сколько пиков градиент будет пытаться максимизировать
repulsion_weight = 3 # Сила расталкивания шаблонов
num_epochs = 400

history_loss = []

print("Начинаем градиентный поиск шаблонов...")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # 1. Получаем нормализованные шаблоны (backprop пройдет через это!)
    W1, W2 = model.get_normalized_W()
    
    # 2. Считаем NCC по всей спектрограмме (backprop пройдет через FFT!)
    ncc1 = fast_ncc_1d(W1, S_target, anchor_f, anchor_t)
    ncc2 = fast_ncc_1d(W2, S_target, anchor_f, anchor_t)
    
    # Игнорируем края
    ncc1[:anchor_t] = 0; ncc1[-(W_W - anchor_t):] = 0
    ncc2[:anchor_t] = 0; ncc2[-(W_W - anchor_t):] = 0
    
    # 3. Находим лучшие K пиков (наше внимание)
    topk_vals1, _ = torch.topk(ncc1, K_peaks)
    topk_vals2, _ = torch.topk(ncc2, K_peaks)
    
    # Цель: максимизировать среднее значение корреляции на этих пиках
    loss_acc = - (topk_vals1.mean() + topk_vals2.mean())
    
    # =====================================================================
    # 4. СДВИГ-ИНВАРИАНТНЫЙ ШТРАФ ЗА ПОХОЖЕСТЬ (Shift-Invariant Repulsion)
    # =====================================================================
    # Подготавливаем размерности для F.conv2d: (batch, channels, H, W)
    w1_tensor = W1.unsqueeze(0).unsqueeze(0)
    w2_tensor = W2.unsqueeze(0).unsqueeze(0)
    
    # Паддинг W1, чтобы W2 мог "скользить" по нему во всех возможных положениях
    # Формат F.pad: (left, right, top, bottom)
    pad_h = H - 1
    pad_w = W_W - 1
    w1_padded = F.pad(w1_tensor, (pad_w, pad_w, pad_h, pad_h))
    
    # Считаем 2D кросс-корреляцию
    # Это матрица скалярных произведений при ВСЕХ возможных сдвигах
    corr_map = F.conv2d(w1_padded, w2_tensor)
    
    # Нормируем на число пикселей N. 
    # Максимум корреляции (абсолютный) не должен быть большим.
    # Берем абсолют, так как перевернутый паттерн (-W) нам тоже не нужен.
    max_overlap = torch.max(torch.abs(corr_map)) / N_pixels
    
    # Штрафуем квадрат максимального перекрытия
    loss_repel = repulsion_weight * (max_overlap ** 2)
    
    # Итоговый loss
    loss = loss_acc + loss_repel
    
    # Шаг градиента
    loss.backward()
    optimizer.step()
    
    history_loss.append(loss.item())
    
    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Peak1: {topk_vals1.mean().item():.3f} | Peak2: {topk_vals2.mean().item():.3f} | Overlap: {max_overlap.item():.3f}")

# =====================================================================
# 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =====================================================================
W1_learned, W2_learned = model.get_normalized_W()

fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# График Loss
axs[0, 0].plot(history_loss, color='purple')
axs[0, 0].set_title("Training Loss")
axs[0, 0].grid(True)

# Исходные маски (для сравнения)
axs[0, 1].imshow(M_circle.numpy(), cmap='magma')
axs[0, 1].set_title("Ground Truth: Circle")
axs[0, 2].imshow(M_square.numpy(), cmap='magma')
axs[0, 2].set_title("Ground Truth: Square")

# Инициализация (что было в начале)
init_v = torch.randn(H, W_W) * 0.01; init_v[:, W_W//2-1:W_W//2+2] = 1.0
init_w = (init_v - init_v.mean()) / torch.sqrt(((init_v - init_v.mean())**2).sum() / N_pixels)
axs[1, 0].imshow(init_w.numpy(), cmap='coolwarm')
axs[1, 0].set_title("Initialization (Vertical Line)")

# Выученные шаблоны
axs[1, 1].imshow(W1_learned.detach().numpy(), cmap='coolwarm')
axs[1, 1].set_title("Learned Pattern 1")
axs[1, 2].imshow(W2_learned.detach().numpy(), cmap='coolwarm')
axs[1, 2].set_title("Learned Pattern 2")

plt.tight_layout()
plt.show()