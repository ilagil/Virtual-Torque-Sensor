import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import scipy.io
# Настройки
device = torch.device('cuda')
WINDOW_SIZE = 60 

# --- 1. ГЕНЕРАЦИЯ ДАННЫХ: ТРИ СКАНДИНАВСКИХ ПРЫЖКА ---ия
t = np.linspace(0, 0.8, 8000) 
torque_true = np.where((t > 0.1) & (t < 0.2), 0.1, 0.02) # Малая нагрузка
torque_true += np.where((t > 0.3) & (t < 0.45), 0.15, 0.0) # Средняя нагрузка (суммируем к базе)
torque_true += np.where((t > 0.6) & (t < 0.65), 0.18, 0.0) # Импульсная перегрузка

# шум + наводка 50Гц 
noise = np.random.normal(0, 0.6, 8000) + 0.3 * np.sin(2 * np.pi * 50 * t)
# Простая линейная модель связи: ток = момент * К + шум
i_measured = torque_true * 15 + noise 

# ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ (EMA) - чуть агрессивнее
i_smooth = [i_measured[0]]
alpha = 0.1 # Сделали фильтр чуть быстрее для коротких импульсов
for n in range(1, len(i_measured)):
    i_smooth.append(alpha * i_measured[n] + (1 - alpha) * i_smooth[-1])
i_smooth = np.array(i_smooth)

# --- 2. ПОДГОТОВКА ---
scaler = StandardScaler()
x_s = scaler.fit_transform(i_smooth.reshape(-1, 1)).astype(np.float32)
y_s = torque_true.reshape(-1, 1).astype(np.float32)

X, Y = [], []
for i in range(len(x_s) - WINDOW_SIZE):
    X.append(x_s[i:i+WINDOW_SIZE])
    Y.append(y_s[i+WINDOW_SIZE])
X, Y = torch.tensor(np.array(X)).to(device), torch.tensor(np.array(Y)).to(device)

# --- 3. МОДЕЛЬ (Оставляем LSTM) ---
class LightObserver(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

model = LightObserver().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# --- 4. ОБУЧЕНИЕ ---
print("Обучение пошло...")
model.train()
for epoch in range(501):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, Y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# --- 5. ФИНАЛЬНЫЙ ОТЧЕТ (Две оси Y) ---
model.eval()
with torch.no_grad():
    final_pred = model(X).cpu().numpy()

fig, ax1 = plt.subplots(figsize=(12, 7))

# Левая ось: Крутящий момент (Нм)
ax1.set_xlabel('Время (сек)', fontsize=12)
ax1.set_ylabel('Крутящий момент (Нм)', color='red', fontsize=12)
ax1.plot(t[WINDOW_SIZE:], torque_true[WINDOW_SIZE:], 'r', label='Цель (Real Torque)', lw=2.5, alpha=0.7)
ax1.plot(t[WINDOW_SIZE:], final_pred, 'b--', label='AI Датчик (Soft Sensor)', lw=2)
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_ylim(-0.02, 0.4) # Даем запас сверху для наглядности
ax1.grid(True, which='both', linestyle='--', alpha=0.4)

# Правая ось: Ток (А)
ax2 = ax1.twinx() 
ax2.set_ylabel('Ток фазы (Амперы)', color='green', fontsize=12)
# Отрисуем сырой ток и сглаженный EMA
ax2.plot(t, i_measured, color='gray', alpha=0.15, label='Сырой ток (Шум)')
ax2.plot(t, i_smooth, color='green', alpha=0.5, label='Ток после EMA-фильтра')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-3, 10) # Масштабируем ток отдельно от момента

# Собираем легенду
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("Стресс-тест Soft Sensor: Три скачка нагрузки", fontsize=15, pad=20)
fig.tight_layout()
plt.show()
# --- 6. СОХРАНЕНИЕ МОДЕЛИ ---
import torch
import torch.nn as nn
import os

# 1. Описание архитектуры (чтобы Python знал, что загружать)
class TorqueLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(TorqueLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


model = TorqueLSTM()

pth_path = r"C:\Users\ilaga\Documents\Wesa\torque_model_Suc.pth"
onnx_path = r"C:\Users\ilaga\Documents\Wesa\torque_model.onnx"

# 3. Сохранение и Экспорт

import torch
import scipy.io
import numpy as np

# 1. СТРОГО определяем пути в самом начале
wesa_path = r"C:\Users\ilaga\Documents\Wesa"
onnx_path = wesa_path + r"\torque_model.onnx"
mat_file_path = wesa_path + r"\data_for_matlab.mat"

# 2. Экспорт ONNX
import scipy.io as sio
import numpy as np

# Вытаскиваем веса
weights = {k.replace('.', '_'): v.detach().cpu().numpy().astype(np.float32) 
           for k, v in model.state_dict().items()}

# Сохраняем в папку Wesa
sio.savemat(r'C:\Users\ilaga\Documents\Wesa\model_weights.mat', weights)
print("--- Веса успешно выгружены в model_weights.mat ---")