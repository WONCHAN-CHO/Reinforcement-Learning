# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 09:28:27 2025

@author: WONCHAN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 환경 설정
T = 10  # 기간
r = 0.04  # 안전 자산 수익률
mu = 0.07  # 위험 자산 기대 수익률
sigma = 0.15  # 위험 자산 변동성
gamma = 0.95  # 할인율 (감속 효과 추가)
delta = 1.05  # 소비 결정 매개변수
learning_rate = 0.01  # 학습률 증가
alpha = 0.1  # 투자 성과에 대한 보상 비중

# 이론적 최적 투자 비율
w_theoretical = 0.555

# 로그 효용 함수
def utility(c):
    return torch.log(c + 1e-8)  # log(0) 방지

# Hamiltonian 기반 정책 네트워크
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)  # (W, lambda) 입력
        self.fc2 = nn.Linear(32, 32)
        self.w_head = nn.Linear(32, 1)

    def forward(self, W, lam):
        x = torch.cat((W, lam), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        w = torch.tanh(self.w_head(x)) * 0.555  # 투자 비율 제한
        return w.squeeze()

# PG-DPO 학습
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=1e-4)

W_trajectory, C_trajectory, w_trajectory = [], [], []

for episode in range(100000):
    W = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)  # 초기 부
    lambda_t = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)  # Costate 변수
    
    log_probs, rewards = [], []
    W_ep, C_ep, w_ep = [], [], []
    
    for t in range(T):
        W_ep.append(W.item())
        
        C_t = (1 - 1 / delta) * W  # 소비 결정
        C_ep.append(C_t.item())
        
        W_input = W.reshape(1, 1)
        lambda_input = lambda_t.reshape(1, 1)
        w_t = policy(W_input, lambda_input)
        
        Z_t = torch.exp(torch.normal(mean=torch.tensor(mu), std=torch.tensor(sigma)))
        W_new = (1 + r) * (1 - w_t) * W + w_t * W * Z_t
        W_new = torch.max(W_new, torch.tensor(1e-3))  # 부가 0 이하로 떨어지는 것 방지
        
        # Costate (Adjoint State) 업데이트: Hamiltonian을 고려한 업데이트
        lambda_new = lambda_t * gamma * ((1 + r) * (1 - w_t) + w_t * Z_t)
        
        # 보상 계산: Pontryagin의 원리를 반영
        reward = utility(C_t) + alpha * utility(W_new)
        
        rewards.append(reward.unsqueeze(0))
        w_ep.append(w_t.item())
        
        W, lambda_t = W_new.detach(), lambda_new.detach()
    
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
lambda_test = torch.ones_like(W_test)
w_pred = policy(W_test, lambda_test).detach().numpy().flatten()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(W_values, np.full_like(W_values, w_theoretical), label='Theoretical investment rate')
plt.plot(W_values, w_pred, label='Learned Investment Ratio', linestyle='dashed')
plt.xlabel('Time t')
plt.ylabel('Investment Ratio (w)')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(T), np.mean(C_trajectory, axis=0), label='Theoretical consumption')
plt.plot(range(T), np.mean(C_trajectory, axis=0), label='Learned consumption', linestyle='dashed')
plt.xlabel('Time t')
plt.ylabel('Consumption (C)')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(range(T), np.mean(W_trajectory, axis=0), label='Theoretical wealth')
plt.plot(range(T), np.mean(W_trajectory, axis=0), label='Learned wealth', linestyle='dashed')
plt.xlabel('Time t')
plt.ylabel('Wealth (W)')
plt.legend()

plt.tight_layout()
plt.show()

# 추가: 투자 비율과 소비의 분포를 히트맵으로 시각화
w_hist, w_xedges, w_yedges = np.histogram2d(np.tile(range(T), len(w_trajectory)), np.array(w_trajectory).flatten(), bins=50)
plt.figure(figsize=(12, 4))
plt.imshow(w_hist.T, origin='lower', aspect='auto', cmap='jet', extent=[w_xedges[0], w_xedges[-1], w_yedges[0], w_yedges[-1]])
plt.colorbar(label='Counts')
plt.xlabel('Time t')
plt.ylabel('Investment Ratio (w)')
plt.title('Investment Ratio Heatmap')
plt.show()

c_hist, c_xedges, c_yedges = np.histogram2d(np.tile(range(T), len(C_trajectory)), np.array(C_trajectory).flatten(), bins=50)
plt.figure(figsize=(12, 4))
plt.imshow(c_hist.T, origin='lower', aspect='auto', cmap='jet', extent=[c_xedges[0], c_xedges[-1], c_yedges[0], c_yedges[-1]])
plt.colorbar(label='Counts')
plt.xlabel('Time t')
plt.ylabel('Consumption (C)')
plt.title('Consumption Heatmap')
plt.show()
