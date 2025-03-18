import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 환경 설정
T = 10  # 최적화 기간 (연속시간)
dt = 0.01  # 시간 간격
r = 0.04  # 안전 자산 수익률
mu = 0.07  # 위험 자산 기대 수익률
sigma = 0.15  # 위험 자산 변동성
gamma = 0.95  # 할인율
delta = 1.05  # 소비 결정 매개변수
learning_rate = 0.01  # 학습률 증가
alpha = 0.1  # 투자 성과에 대한 보상 비중

# 로그 효용 함수
def utility(c):
    return torch.log(c + 1e-8)  # log(0) 방지

# 연속시간 정책 네트워크 (Merton Model)
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)  # Wealth W 입력
        self.fc2 = nn.Linear(32, 32)
        self.w_head = nn.Linear(32, 1)
        self.c_head = nn.Linear(32, 1)

    def forward(self, W):
        x = torch.relu(self.fc1(W))
        x = torch.relu(self.fc2(x))
        w = torch.sigmoid(self.w_head(x))  # 투자 비율 [0,1] 제한
        c = torch.relu(self.c_head(x))  # 소비는 양수
        return w.squeeze(), c.squeeze()

# PG-DPO 학습
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=1e-4)

W_trajectory, C_trajectory, w_trajectory = [], [], []

def stochastic_dynamics(t, W, w, c):
    """ 연속시간 자산 변화 SDE """
    dW = (r * (1 - w) + w * mu) * W - c
    return dW

for episode in range(50000):
    W = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)  # 초기 부
    time_steps = np.arange(0, T, dt)
    
    log_probs, rewards = [], []
    W_ep, C_ep, w_ep = [], [], []
    
    for t in time_steps:
        W_ep.append(W.item())
        W_input = W.reshape(1, 1)
        w_t, C_t = policy(W_input)
        C_ep.append(C_t.item())
        w_ep.append(w_t.item())
        
        # SDE에 따른 부 변화 (Euler-Maruyama 방법)
        dZ = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(np.sqrt(dt)))
        dW = (r * (1 - w_t) + w_t * mu) * W * dt + w_t * sigma * W * dZ - C_t * dt
        W_new = torch.max(W + dW, torch.tensor(1e-3))  # 부가 0 이하로 떨어지는 것 방지
        
        # 보상 계산
        reward = utility(C_t)
        rewards.append(reward.unsqueeze(0))
        
        W = W_new.detach()
    
    W_trajectory.append(W_ep)
    C_trajectory.append(C_ep)
    w_trajectory.append(w_ep)
    
    returns = torch.cat(rewards).flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,))
    loss = -returns.sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 학습된 정책 시각화
policy.eval()
W_values = np.linspace(0.1, 10, 100)
W_test = torch.tensor(W_values, dtype=torch.float32).reshape(-1, 1)
w_pred, c_pred = policy(W_test)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(W_values, w_pred.detach().numpy(), label='Learned Investment Ratio', linestyle='dashed')
plt.xlabel('Wealth (W)')
plt.ylabel('Investment Ratio (w)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(W_values, c_pred.detach().numpy(), label='Learned Consumption', linestyle='dashed')
plt.xlabel('Wealth (W)')
plt.ylabel('Consumption (C)')
plt.legend()

plt.tight_layout()
plt.show()

# 투자 비율 및 소비 히트맵 추가
T_grid = np.arange(0, T, dt)
W_grid = np.linspace(0.1, 10, len(W_trajectory[0]))
T_mesh, W_mesh = np.meshgrid(T_grid, W_grid)

w_values = np.array(w_trajectory).T
c_values = np.array(C_trajectory).T

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(W_mesh, T_mesh, w_values, levels=50, cmap='jet')
plt.colorbar(label='Investment Ratio (w)')
plt.xlabel('Wealth (W)')
plt.ylabel('T-t')
plt.title('Investment Ratio Heatmap')

plt.subplot(1, 2, 2)
plt.contourf(W_mesh, T_mesh, c_values, levels=50, cmap='jet')
plt.colorbar(label='Consumption (C)')
plt.xlabel('Wealth (W)')
plt.ylabel('T-t')
plt.title('Consumption Heatmap')

plt.tight_layout()
plt.show()
