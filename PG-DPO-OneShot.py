# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:17:57 2025

@author: WONCHAN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulation parameters
T = 10  # Time horizon
dt = 0.1  # Time step
N = int(T / dt)  # Number of time steps
num_paths = 100000  # Number of Monte Carlo simulations

# Economic parameters
rho = 0.04  # Discount rate
gamma = 2  # CRRA risk aversion parameter
r = 0.03  # Risk-free rate
mu = 0.08  # Expected return on risky asset
sigma = 0.2  # Volatility of risky asset
X0 = 1.0  # Initial wealth
C_min = 0.005  # Minimum consumption floor
C_smooth = 0.05  # Smoothing factor for consumption

# Pontryagin-Guided Adjoint Calculation
def compute_adjoint(X, t):
    lambda_t = torch.exp(-rho * t) * (gamma / rho) * X.pow(-gamma)  # Costate variable
    d_lambda_dx = -gamma * lambda_t / X  # Derivative w.r.t. X
    return lambda_t, d_lambda_dx

# OneShot Policy Computation
def oneshot_policy(X, t, prev_C):
    lambda_t, d_lambda_dx = compute_adjoint(X, t)
    
    C_opt = (rho / gamma) * X  # Optimal consumption
    C_opt = torch.clamp(C_opt, min=C_min)  # Enforce consumption floor
    
    # Apply soft smoothing
    C_opt = C_smooth * prev_C + (1 - C_smooth) * C_opt
    
    pi_opt = ((mu - r) / (gamma * sigma ** 2)) * torch.ones_like(X, device=X.device)
    pi_opt = torch.clamp(pi_opt, min=0.0, max=1.0)
    
    return pi_opt, C_opt

# Compute theoretical paths
def theoretical_wealth(t):
    return X0 * np.exp((r + ((mu - r) ** 2) / (2 * sigma ** 2 * gamma)) * t)

def theoretical_consumption(t):
    return (rho / gamma) * theoretical_wealth(t)

# Warm-up Policy Neural Network
class PolicyNN(nn.Module):
    def __init__(self):
        super(PolicyNN, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3_pi = nn.Linear(32, 1)
        self.fc3_c = nn.Linear(32, 1)

    def forward(self, t, X):
        x = torch.cat((t, X), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        pi = torch.sigmoid(self.fc3_pi(x))  # Portfolio allocation
        C = torch.clamp((rho / gamma) * X, min=C_min)
        return pi, C

# Train PolicyNN for Warm-up Phase
policy_net = PolicyNN().to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

num_warmup_epochs = 500  # Warm-up iterations
for epoch in range(num_warmup_epochs):
    t_values = torch.linspace(0, T, N, device=device).repeat(num_paths, 1).unsqueeze(-1)
    X = X0 * torch.ones(num_paths, 1, device=device, requires_grad=True)  # Ensure grad tracking
    loss = 0
    prev_C = torch.zeros_like(X, device=device)
    for t_idx in range(N):
        t = t_values[:, t_idx, :]
        pi, C = policy_net(t, X)
        loss += loss_fn(C, (rho / gamma) * X)  # Ensure loss requires grad
    
    # Ensure loss requires grad before backward
    if loss.requires_grad:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        print("Warning: loss does not require grad. Check computation graph.")
    
    if epoch % 100 == 0:
        print(f"Warm-up Epoch {epoch}, Loss: {loss.item():.6f}")

# Testing OneShot Policy
X_all = np.zeros((num_paths, N + 1))
C_all = np.zeros((num_paths, N + 1))
X_all[:, 0] = X0

t_values = torch.linspace(0, T, N, device=device).repeat(num_paths, 1).unsqueeze(-1)
X = X0 * torch.ones(num_paths, 1, device=device)
prev_C = torch.zeros_like(X, device=X.device)
for t_idx in range(N):
    t = t_values[:, t_idx, :]
    pi, C = oneshot_policy(X, t, prev_C)
    dW = torch.randn(num_paths, 1, device=device) * np.sqrt(dt)
    X_new = X * (1 + (r + pi * (mu - r)) * dt + pi * sigma * dW) - C * dt
    X_all[:, t_idx + 1] = X_new.cpu().numpy().flatten()
    C_all[:, t_idx] = C.cpu().numpy().flatten()
    X = X_new.clamp(min=1e-3)
    prev_C = C

# Compute mean wealth and consumption
mean_X = np.mean(X_all, axis=0)
mean_C = np.mean(C_all, axis=0)

# Compute theoretical paths
time_grid = np.linspace(0, T, N + 1)
theo_X = theoretical_wealth(time_grid)
theo_C = theoretical_consumption(time_grid)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(time_grid, mean_X, label='Mean Wealth $X_t$', marker='o')
axes[0].plot(time_grid, theo_X, label='Theoretical Wealth', linestyle='dashed', color='red')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Wealth')
axes[0].legend()
axes[0].set_title("OneShot: Wealth Path")
axes[0].grid()

axes[1].plot(time_grid, mean_C, label='Mean Consumption $C_t$', marker='s', color='orange')
axes[1].plot(time_grid, theo_C, label='Theoretical Consumption', linestyle='dashed', color='red')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Consumption')
axes[1].legend()
axes[1].set_title("OneShot: Consumption Path")
axes[1].grid()

plt.tight_layout()
plt.show()
