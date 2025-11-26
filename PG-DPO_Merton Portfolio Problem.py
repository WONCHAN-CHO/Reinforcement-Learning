import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf


class Config:
    """실험 전반의 설정 값을 관리하는 컨테이너."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 환경 설정
    num_assets: int = 50
    T: float = 1.0
    dt: float = 0.02
    r: float = 0.03
    mu_base: float = 0.08
    discount: float = 0.05
    alpha: float = 0.005  # 거래 비용 비율
    epsilon: float = 1e-6

    # 학습 설정
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 1000
    log_every: int = 100

    def __init__(self) -> None:
        self.time_steps = int(self.T / self.dt)


cfg = Config()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_robust_covariance(num_assets: int, num_observations: int = 120):
    """
    Ledoit-Wolf shrinkage로 안정화된 공분산 행렬을 생성한다.

    Args:
        num_assets: 자산 개수.
        num_observations: 과거 관측치 개수 (노이즈가 많은 과거 데이터를 가정).

    Returns:
        torch.Tensor: PyTorch용 축소 공분산 행렬.
        np.ndarray: 샘플 공분산 행렬.
        np.ndarray: Ledoit-Wolf 적용 후 공분산 행렬.
    """

    # 1. 약한 상관관계를 가진 "진짜" 공분산 행렬 생성
    A = np.random.randn(num_assets, num_assets)
    true_cov = np.dot(A, A.T)

    # 2. 노이즈가 섞인 수익률 데이터 생성
    noisy_returns = np.random.multivariate_normal(
        mean=np.zeros(num_assets), cov=true_cov, size=num_observations
    )

    # 3. Ledoit-Wolf shrinkage 적용
    lw = LedoitWolf()
    shrunk_cov = lw.fit(noisy_returns).covariance_
    sample_cov = np.cov(noisy_returns, rowvar=False)

    print(f"Ledoit-Wolf 적용 완료. Shrinkage 강도: {lw.shrinkage_:.4f}")

    return (
        torch.tensor(shrunk_cov, dtype=torch.float32, device=cfg.device),
        sample_cov,
        shrunk_cov,
    )


# 시드 고정 및 공분산 계산
set_seed(7)
cov_matrix_torch, sample_cov_np, shrunk_cov_np = get_robust_covariance(cfg.num_assets)

# 상관관계가 반영된 하위 삼각행렬 (시뮬레이션용)
L_matrix = torch.linalg.cholesky(cov_matrix_torch)

# 기대수익률 벡터를 약간 무작위로 생성
mu_vec = torch.rand(cfg.num_assets, device=cfg.device) * 0.05 + cfg.mu_base


class PolicyNet(nn.Module):
    """
    Projected PG-DPO 정책 네트워크.

    입력: 시점 t, 현재 부, 현재 포트폴리오 비중.
    출력: 소비 비율 (0~1), 목표 포트폴리오 비중 (simplex 위).
    """

    def __init__(self, num_assets: int) -> None:
        super().__init__()
        input_dim = 1 + 1 + num_assets  # t, wealth, weights

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.c_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.w_head = nn.Sequential(nn.Linear(128, num_assets), nn.Softmax(dim=1))

    def forward(self, t: torch.Tensor, wealth: torch.Tensor, weights: torch.Tensor):
        x = torch.cat([t, wealth, weights], dim=1)
        feat = self.net(x)
        return self.c_head(feat), self.w_head(feat)


def utility(consumption: torch.Tensor) -> torch.Tensor:
    """로그 효용 함수 (0 근처 불안정성 완화)."""

    return torch.log(consumption + cfg.epsilon)


def rollout(model: PolicyNet, batch_size: int, track_gradients: bool = True):
    """
    Ledoit-Wolf 기반 상관관계를 반영한 GBM으로 배치를 시뮬레이션한다.

    Returns:
        total_utility: (batch, 1) 누적 효용.
        wealth: (batch, 1) 최종 부.
    """

    context = torch.enable_grad() if track_gradients else torch.no_grad()
    with context:
        wealth = torch.ones(batch_size, 1, device=cfg.device)
        curr_weights = torch.ones(batch_size, cfg.num_assets, device=cfg.device)
        curr_weights = curr_weights / curr_weights.sum(dim=1, keepdim=True)
        total_utility = torch.zeros(batch_size, 1, device=cfg.device)

        for i in range(cfg.time_steps):
            t_val = torch.full((batch_size, 1), i * cfg.dt, device=cfg.device)
            c_frac, target_weights = model(t_val, wealth, curr_weights)

            turnover = torch.sum(torch.abs(target_weights - curr_weights), dim=1, keepdim=True)
            cost = cfg.alpha * turnover * wealth
            consumption = c_frac * wealth

            wealth_post = torch.clamp(wealth - cost - consumption, min=cfg.epsilon)

            # 상관관계가 있는 잡음 생성
            Z = torch.randn(batch_size, cfg.num_assets, device=cfg.device)
            correlated_noise = torch.matmul(Z, L_matrix.T)

            # 자산 수익률 및 포트폴리오 수익률 계산
            asset_ret = (mu_vec * cfg.dt) + (correlated_noise * np.sqrt(cfg.dt))
            port_ret = torch.sum(target_weights * asset_ret, dim=1, keepdim=True)

            wealth = wealth_post * (1 + port_ret)
            weights_drifted = target_weights * (1 + asset_ret) / (1 + port_ret)
            curr_weights = weights_drifted / weights_drifted.sum(dim=1, keepdim=True)

            total_utility += utility(consumption) * cfg.dt * torch.exp(-cfg.discount * t_val)

        return total_utility, wealth


def train_pg_dpo(model: PolicyNet):
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_history = []
    wealth_history = []

    print("학습 시작...")

    for epoch in range(cfg.epochs):
        optimizer.zero_grad()
        total_utility, wealth = rollout(model, cfg.batch_size, track_gradients=True)
        loss = -total_utility.mean()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        wealth_history.append(wealth.mean().item())

        if epoch % cfg.log_every == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}, Final Wealth {wealth.mean().item():.2f}")

    return loss_history, wealth_history


@torch.no_grad()
def evaluate_policy(model: PolicyNet, eval_batch: int = 512):
    model.eval()
    total_utility, wealth = rollout(model, eval_batch, track_gradients=False)
    avg_utility = total_utility.mean().item()
    avg_wealth = wealth.mean().item()
    return avg_utility, avg_wealth


def plot_results(loss_history, sample_cov_np, shrunk_cov_np, wealth_history):
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.plot(loss_history)
    plt.title("Training Loss (Negative Utility)")
    plt.xlabel("Epoch")

    plt.subplot(1, 4, 2)
    plt.title("Sample Covariance (Noisy)")
    plt.imshow(sample_cov_np[:20, :20], cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title("Ledoit-Wolf Covariance (Stabilized)")
    plt.imshow(shrunk_cov_np[:20, :20], cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.plot(wealth_history)
    plt.title("Mean Wealth During Training")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.show()


def main():
    policy = PolicyNet(cfg.num_assets).to(cfg.device)
    loss_history, wealth_history = train_pg_dpo(policy)

    avg_u, avg_w = evaluate_policy(policy)
    print(f"평균 효용: {avg_u:.4f}, 평균 최종 부: {avg_w:.2f}")

    plot_results(loss_history, sample_cov_np, shrunk_cov_np, wealth_history)


if __name__ == "__main__":
    main()
