# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:54:25 2025

@author: WONCHAN
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats

# Kaggle API 설정
KAGGLE_USERNAME = 'dnjscks'
KAGGLE_KEY = '26186342'
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    f.write(f'{{"username":"{KAGGLE_USERNAME}","key":"{KAGGLE_KEY}"}}')
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

# 데이터 다운로드
if not os.path.exists("sp-500-stocks.zip"):
    os.system("kaggle datasets download -d andrewmvd/sp-500-stocks")
    with zipfile.ZipFile("sp-500-stocks.zip", 'r') as zip_ref:
        zip_ref.extractall("sp500_data")

# 데이터 불러오기
df = pd.read_csv("sp500_data/sp500_stocks.csv", parse_dates=['Date'])
df = df.rename(columns={"Date": "date", "Symbol": "ticker", "Adj Close": "adj_close"})
df = df.pivot(index='date', columns='ticker', values='adj_close')
df = df.dropna(axis=1)

# 수익률 및 통계량
returns = df.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
num_assets = len(mean_returns)
risk_free_rate = 0.01

def portfolio_performance(weights):
    ret = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, std

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

def tangency_portfolio():
    def neg_sharpe_ratio(weights):
        ret, std = portfolio_performance(weights)
        return -(ret - risk_free_rate) / std
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets],
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# 계산
frontier = efficient_frontier()
tangent = tangency_portfolio()
tan_ret, tan_std = portfolio_performance(tangent.x)

# 무작위 포트폴리오
num_portfolios = 5000
random_returns, random_risks = [], []
for _ in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(num_assets))
    ret, std = portfolio_performance(weights)
    random_returns.append(ret)
    random_risks.append(std)

# CML
x_vals = np.linspace(0, max(random_risks), 100)
tangency_sharpe = (tan_ret - risk_free_rate) / tan_std
cml_y = risk_free_rate + tangency_sharpe * x_vals

# 개별 종목 수익률 & 표준편차
individual_returns = mean_returns.values
individual_stds = np.sqrt(np.diag(cov_matrix))

#  시각화
# --- 종합 포트폴리오 시각화 ---
plt.figure(figsize=(12, 7))
plt.scatter(random_risks, random_returns, c='red', s=2, alpha=0.3, label='Random Portfolios')
stds, rets, _ = zip(*frontier)
plt.plot(stds, rets, color='blue', label='Efficient Frontier')
plt.scatter(tan_std, tan_ret, marker='*', color='gold', s=200, label='Tangency Portfolio (T)')
plt.plot(x_vals, cml_y, linestyle='--', color='black', label='Capital Market Line (CML)')
plt.scatter(individual_stds, individual_returns, c='green', s=25, label='Individual Assets')

plt.xlabel('Standard Deviation of Portfolio Returns')
plt.ylabel('Mean of Portfolio Returns')
plt.title('Mean-Variance Efficient Frontier and Random Portfolios')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 밀도 히트맵 (무작위 포트폴리오) ---
xy = np.vstack([random_risks, random_returns])
z = stats.gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = np.array(random_risks)[idx], np.array(random_returns)[idx], z[idx]

plt.figure(figsize=(10, 7))
plt.scatter(x, y, c=np.log(z), cmap='Reds', s=10, alpha=0.8)
plt.colorbar(label="log(Density)")
plt.xlabel('Standard Deviation of Portfolio Returns')
plt.ylabel('Mean of Portfolio Returns')
plt.title('Density Heatmap of Random Portfolios')
plt.grid(True)
plt.tight_layout()
plt.show()

#  Tangency Portfolio 상위 종목 비중 바 차트
tangent_weights = pd.Series(tangent.x, index=mean_returns.index)
top_assets = tangent_weights[tangent_weights > 0.01].sort_values(ascending=False)

# --- Tangency Portfolio 상위 종목 바 차트 ---
plt.figure(figsize=(10, 4))
sns.barplot(x=top_assets.index, y=top_assets.values, palette="viridis")
plt.title("Tangency Portfolio Weights (>1%)")
plt.ylabel("Weight")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Tangency Portfolio와 가장 가까운 단일 종목 찾기
distances = np.sqrt((individual_returns - tan_ret)**2 + (individual_stds - tan_std)**2)
closest_index = np.argmin(distances)
closest_asset = mean_returns.index[closest_index]

print("\n Tangency Portfolio 주요 종목 (비중 > 1%):")
for ticker, weight in top_assets.items():
    print(f"{ticker}: {weight:.2%}")

print(f"\n Tangency Portfolio와 가장 가까운 단일 종목: {closest_asset}")
print(f"    - Return: {individual_returns[closest_index]:.4f}")
print(f"    - StdDev: {individual_stds[closest_index]:.4f}")