# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 22:57:30 2025

@author: WONCHAN
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
import scipy.stats as stats

# 1. Kaggle 데이터 다운로드 (S&P 500)
KAGGLE_USERNAME = 'dnjscks'
KAGGLE_KEY = '26186342'
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    f.write(f'{{"username":"{KAGGLE_USERNAME}","key":"{KAGGLE_KEY}"}}')
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

if not os.path.exists("sp-500-stocks.zip"):
    os.system("kaggle datasets download -d andrewmvd/sp-500-stocks")
    with zipfile.ZipFile("sp-500-stocks.zip", 'r') as zip_ref:
        zip_ref.extractall("sp500_data")

# 2. 데이터 불러오기 및 전처리
df = pd.read_csv("sp500_data/sp500_stocks.csv", parse_dates=['Date'])
df = df.rename(columns={"Date": "date", "Symbol": "ticker", "Adj Close": "adj_close"})
df = df.pivot(index='date', columns='ticker', values='adj_close')
df = df.dropna(axis=1)

# 3. 통계량 계산
returns = df.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
num_assets = len(mean_returns)
risk_free_rate = 0.01

# 4. 포트폴리오 성능 함수
def portfolio_performance(weights):
    ret = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, std

# 5. 효율적 투자선 계산
def efficient_frontier(num_points=100):
    results = []
    for r in np.linspace(mean_returns.min(), mean_returns.max(), num_points):
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - r}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        result = minimize(lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))),
                          num_assets * [1. / num_assets],
                          method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            std = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
            results.append((std, r, result.x))
    return results

# 6. Tangency Portfolio 계산 (no short-selling)
def tangency_portfolio():
    def neg_sharpe_ratio(weights):
        ret, std = portfolio_performance(weights)
        return -(ret - risk_free_rate) / std
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets],
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

frontier = efficient_frontier()
tangent = tangency_portfolio()
tan_ret, tan_std = portfolio_performance(tangent.x)

# 7. 무작위 포트폴리오
random_returns, random_risks = [], []
for _ in range(5000):
    weights = np.random.dirichlet(np.ones(num_assets))
    ret, std = portfolio_performance(weights)
    random_returns.append(ret)
    random_risks.append(std)

# 8. CML 계산
x_vals = np.linspace(0, max(random_risks), 100)
tangency_sharpe = (tan_ret - risk_free_rate) / tan_std
cml_y = risk_free_rate + tangency_sharpe * x_vals

# 9. 개별 종목 수익률/리스크
individual_returns = mean_returns.values
individual_stds = np.sqrt(np.diag(cov_matrix))

# 10. 시각화 (종합)
plt.figure(figsize=(12, 7))
plt.scatter(random_risks, random_returns, c='red', s=2, alpha=0.3, label='Random Portfolios')
stds, rets, _ = zip(*frontier)
plt.plot(stds, rets, color='blue', label='Efficient Frontier')
plt.scatter(tan_std, tan_ret, marker='*', color='gold', s=200, label='Tangency Portfolio')
plt.plot(x_vals, cml_y, linestyle='--', color='black', label='CML')
plt.scatter(individual_stds, individual_returns, c='green', s=25, label='Individual Assets')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.title('Efficient Frontier, CML, and Random Portfolios')
plt.tight_layout()
plt.show()

# 11. 히트맵 시각화
xy = np.vstack([random_risks, random_returns])
z = stats.gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = np.array(random_risks)[idx], np.array(random_returns)[idx], z[idx]

plt.figure(figsize=(10, 7))
plt.scatter(x, y, c=np.log(z), cmap='Reds', s=10, alpha=0.8)
plt.colorbar(label="log(Density)")
plt.xlabel('Risk')
plt.ylabel('Return')
plt.title('Density Heatmap of Random Portfolios')
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Tangency 포트폴리오 상위 종목 바 차트
tangent_weights = pd.Series(tangent.x, index=mean_returns.index)
top_assets = tangent_weights[tangent_weights > 0.01].sort_values(ascending=False)
plt.figure(figsize=(10, 4))
sns.barplot(x=top_assets.index, y=top_assets.values, palette="viridis")
plt.title("Tangency Portfolio Weights (>1%)")
plt.ylabel("Weight")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 13. Tangency와 가장 가까운 종목 찾기
distances = np.sqrt((individual_returns - tan_ret)**2 + (individual_stds - tan_std)**2)
closest_index = np.argmin(distances)
closest_asset = mean_returns.index[closest_index]
print("\nTangency Portfolio 주요 종목 (비중 > 1%):")
for ticker, weight in top_assets.items():
    print(f"{ticker}: {weight:.2%}")
print(f"\nTangency Portfolio와 가장 가까운 종목: {closest_asset}")
print(f"    - Return: {individual_returns[closest_index]:.4f}")
print(f"    - StdDev: {individual_stds[closest_index]:.4f}")

# 14. PG-DPO 학습
mu = torch.tensor(mean_returns.values, dtype=torch.float32)
Sigma = torch.tensor(cov_matrix.values, dtype=torch.float32)
w_tangent = torch.tensor(tangent.x, dtype=torch.float32)

class PolicyNet(nn.Module):
    def __init__(self, input_dim, n_assets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_assets)
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

input_vec = torch.cat([mu, torch.diag(Sigma)])
state = input_vec.unsqueeze(0).repeat(128, 1)

def reward_fn(weights, mu, Sigma):
    quad = torch.einsum('bi,ij,bj->b', weights, Sigma, weights)
    linear = torch.einsum('bi,i->b', weights, mu)
    return -0.5 * quad + linear

policy = PolicyNet(input_dim=state.shape[1], n_assets=num_assets)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

for epoch in range(3000):
    weights = policy(state)
    rewards = reward_fn(weights, mu, Sigma)
    loss = -rewards.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"[Epoch {epoch}] Reward: {rewards.mean().item():.4f}")

# 15. PG-DPO 결과 비교
with torch.no_grad():
    w_pg = policy(state[0:1]).squeeze().numpy()
    w_tan = w_tangent.numpy()
    dist = np.linalg.norm(w_pg - w_tan)

    print("\n🔍 Tangency vs PG-DPO 정책 비교 (비중 상위 10개)")
    df_compare = pd.DataFrame({
        "Tangency": w_tan,
        "PG_DPO": w_pg
    }, index=mean_returns.index)
    df_top = df_compare[df_compare["Tangency"] > 0.01].sort_values("Tangency", ascending=False).head(10)
    print(df_top)

    print(f"\n📏 L2 Distance between policies: {dist:.4f}")

    df_top.plot(kind='bar', figsize=(10, 5))
    plt.title("Tangency vs PG-DPO Portfolio Weights (Top 10)")
    plt.ylabel("Weight")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()