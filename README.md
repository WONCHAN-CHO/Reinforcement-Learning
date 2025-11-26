# Reinforcement-Learning Experiments

This repository contains several self-contained experiments. The `PG-DPO_Merton Portfolio Problem.py` script trains a Projected PG-DPO policy for Merton's portfolio problem with 50 assets, transaction costs, and a Ledoit–Wolf–stabilized covariance-driven simulator.

## Quick start for the PG-DPO Merton script

### Where the script lives
- The training script is stored at the **repository root** as `./PG-DPO_Merton Portfolio Problem.py`.
- Run commands from the repository root (the same folder that contains this README) so the filename is resolved correctly.
- If you cloned this repo from GitHub, the file is already in your local checkout at that path. To make it appear on your own
  GitHub account, push your commits after cloning or forking.

1. **Install dependencies (Python 3.9+)**
   ```bash
   pip install torch numpy matplotlib scikit-learn
   ```

2. **Run the experiment**
   In the repository root, run:
   ```bash
   python "PG-DPO_Merton Portfolio Problem.py"
   ```

   The `__main__` block will:
   - Generate noisy historical returns, apply Ledoit–Wolf shrinkage, and build the Cholesky factor for correlated GBM simulation.
   - Train the PG-DPO policy with transaction costs.
   - Print epoch logs and show plots for training loss, wealth evolution, and covariance comparisons.

## Notes
- CUDA is used automatically if available; otherwise the script runs on CPU.
- To adjust experiment settings (time horizon, learning rate, batch size, etc.), edit the `Config` class near the top of the script.
