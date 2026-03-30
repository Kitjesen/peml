# -*- coding: utf-8 -*-
"""
Experiment 1: Duffing Oscillator -- Full reproduction of Section 3.

Reproduces:
  - Fig. 4a: External force error histogram (test set)
  - Fig. 4b: Predicted vs actual force time series (test sample)
  - Fig. 5:  NN-learned spring vs actual nonlinear spring
  - Fig. 7:  Forward dynamics integration (identified params)
  - Fig. 8:  Forward dynamics with modified params (m=2, c=0.08)
"""

import os
import sys
import io
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models import DuffingHybridModel
from duffing_data import (
    generate_dataset, M_TRUE, C_TRUE, K_TRUE, D_TRUE,
    DT, T_TOTAL, N_STEPS, duffing_rhs, generate_random_sinusoidal_load,
)

# Force UTF-8 stdout on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------------------------
# 1. Data generation
# -----------------------------------------------
def prepare_data():
    print("Generating data...")
    train_data = generate_dataset(50, seed=42)
    test_data = generate_dataset(20, seed=123)

    def to_tensors(data):
        return {k: torch.tensor(v, dtype=torch.float32, device=DEVICE)
                for k, v in data.items()}

    return to_tensors(train_data), to_tensors(test_data), train_data, test_data


# -----------------------------------------------
# 2. Training
# -----------------------------------------------
def train_model(model, train, n_sims=50, n_epochs=6000, lr=1e-3):
    """
    Per-simulation mini-batch training (50 optimizer steps per epoch = 300k total).
    Cosine annealing LR for stable late-stage convergence.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    x, xd, xdd, tau = train['x'], train['x_dot'], train['x_ddot'], train['tau']
    sim_indices = list(range(n_sims))

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        np.random.shuffle(sim_indices)

        for i in sim_indices:
            sl = slice(i * N_STEPS, (i + 1) * N_STEPS)
            optimizer.zero_grad()
            tau_pred = model(x[sl], xd[sl], xdd[sl])
            loss = loss_fn(tau_pred, tau[sl])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / n_sims
        losses.append(avg_loss)
        if (epoch + 1) % 1000 == 0:
            m_val = model.m.item()
            c_val = model.c.item()
            cur_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg_loss:.6e} | "
                  f"m={m_val:.7f} | c={c_val:.10f} | lr={cur_lr:.2e}")

    return losses


# -----------------------------------------------
# 3. Evaluation
# -----------------------------------------------
def evaluate(model, test):
    model.eval()
    with torch.no_grad():
        tau_pred = model(test['x'], test['x_dot'], test['x_ddot'])
        error = (tau_pred - test['tau']).cpu().numpy()
        tau_pred_np = tau_pred.cpu().numpy()
        tau_actual_np = test['tau'].cpu().numpy()
    return error, tau_pred_np, tau_actual_np


# -----------------------------------------------
# 4. Forward dynamics integration
# -----------------------------------------------
def integrate_forward(model, tau_func, x0, v0, t_span, t_eval,
                      m_override=None, c_override=None):
    """
    Integrate the identified HM forward in time (Eq. 12).
    Optionally override physical params for design studies (Fig. 8).
    """
    model.eval()
    m_val = m_override if m_override is not None else model.m.item()
    c_val = c_override if c_override is not None else model.c.item()

    def rhs(t, state):
        x, v = state
        x_t = torch.tensor([x], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            g_nn = model.nn_unknown(x_t.unsqueeze(-1)).squeeze(-1).item()
        tau = tau_func(t)
        a = (tau - c_val * v - g_nn) / m_val
        return [v, a]

    sol = solve_ivp(rhs, t_span, [x0, v0], t_eval=t_eval,
                    method='RK45', rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1]


# -----------------------------------------------
# 5. Visualization -- reproduce all paper figures
# -----------------------------------------------
def plot_training_loss(losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    # Raw loss (faint) + smoothed (bold)
    ax.semilogy(losses, alpha=0.15, color='C0', linewidth=0.5)
    # Exponential moving average for smooth curve
    alpha_ema = 0.02
    ema = [losses[0]]
    for l in losses[1:]:
        ema.append(alpha_ema * l + (1 - alpha_ema) * ema[-1])
    ax.semilogy(ema, color='C0', linewidth=1.5, label='Smoothed (EMA)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fig_training_loss.png'), dpi=150)
    plt.close(fig)


def plot_fig4a_error_histogram(error):
    """Fig. 4a: Predicted external force error histogram."""
    mu, sigma = np.mean(error), np.std(error)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(error, bins=80, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_xlabel('External Force error [N]')
    ax.set_ylabel('Count')
    ax.set_title(f'External Force error [N]: mu={mu:.4e}, sigma={sigma:.4e}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fig4a_error_histogram.png'), dpi=150)
    plt.close(fig)
    print(f"  Error stats: mu={mu:.6e}, sigma={sigma:.6e}")


def plot_fig4b_force_comparison(test_data_np, tau_pred_np):
    """Fig. 4b: Predicted vs actual force for one test simulation."""
    idx = 4
    sl = slice(idx * N_STEPS, (idx + 1) * N_STEPS)
    t = test_data_np['t'][sl]
    tau_actual = test_data_np['tau'][sl]
    tau_pred = tau_pred_np[sl]
    err = tau_pred - tau_actual

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(t, tau_actual, 'C1', label='Actual', linewidth=1.2)
    axes[0].plot(t, tau_pred, 'C0--', label='Hybrid Model', linewidth=1.0)
    axes[0].set_ylabel('External force [N]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, err, 'C3', linewidth=0.8, label='Error (HM - Actual)')
    axes[1].set_xlabel('t [s]')
    axes[1].set_ylabel('Error [N]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Fig. 4b -- HM validation on test data')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fig4b_force_comparison.png'), dpi=150)
    plt.close(fig)


def plot_fig5_spring_comparison(model):
    """Fig. 5: Actual nonlinear spring vs NN-learned spring."""
    x_range = np.linspace(-10, 10, 500)
    actual = K_TRUE * x_range + D_TRUE * x_range**3

    x_t = torch.tensor(x_range, dtype=torch.float32, device=DEVICE)
    nn_spring = model.get_nn_spring_force(x_t).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_range, actual, 'C1', linewidth=2.0, label='Actual')
    ax.plot(x_range, nn_spring, 'C0--', linewidth=1.5, label='Hybrid Model')
    ax.set_xlabel('x displacement [m]')
    ax.set_ylabel('Spring force [N]')
    ax.set_title('Fig. 5 -- Nonlinear spring: actual vs NN approximation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fig5_spring_comparison.png'), dpi=150)
    plt.close(fig)


def plot_fig7_forward_dynamics(model):
    """Fig. 7: Forward dynamics integration comparison (identified params)."""
    rng = np.random.default_rng(999)
    A1, f1 = rng.uniform(2, 8), rng.uniform(0.3, 1.5)
    A2, f2 = rng.uniform(1, 5), rng.uniform(0.1, 0.8)
    tau_func = lambda t: A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)

    t_eval = np.linspace(0.01, T_TOTAL, 2000)

    sol_true = solve_ivp(
        duffing_rhs, [0, T_TOTAL], [0.0, 0.0],
        args=(tau_func, M_TRUE, C_TRUE, K_TRUE, D_TRUE),
        t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12,
    )

    t_hm, x_hm, xd_hm = integrate_forward(
        model, tau_func, 0.0, 0.0, [0, T_TOTAL], t_eval,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(sol_true.t, sol_true.y[0], 'C1', linewidth=1.2, label='Actual')
    axes[0].plot(t_hm, x_hm, 'C0--', linewidth=1.0, label='Hybrid Model')
    axes[0].set_ylabel('x [m]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sol_true.t, sol_true.y[1], 'C1', linewidth=1.2, label='Actual')
    axes[1].plot(t_hm, xd_hm, 'C0--', linewidth=1.0, label='Hybrid Model')
    axes[1].set_ylabel('x_dot [m/s]')
    axes[1].set_xlabel('t [s]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Fig. 7 -- Forward dynamics: actual vs identified HM')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fig7_forward_dynamics.png'), dpi=150)
    plt.close(fig)


def plot_fig8_modified_params(model):
    """Fig. 8: Forward dynamics with modified physical params (m=2, c=0.08)."""
    m_new, c_new = 2.0, 0.08

    tau_func = lambda t: (1.0 * np.sin(2 * np.pi * 0.5 * t) +
                          0.5 * np.sin(2 * np.pi * 0.75 * t) +
                          1.2 * np.sin(2 * np.pi * 0.3 * t))

    t_eval = np.linspace(0.01, T_TOTAL, 2000)

    sol_true = solve_ivp(
        duffing_rhs, [0, T_TOTAL], [0.0, 0.0],
        args=(tau_func, m_new, c_new, K_TRUE, D_TRUE),
        t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12,
    )

    t_hm, x_hm, xd_hm = integrate_forward(
        model, tau_func, 0.0, 0.0, [0, T_TOTAL], t_eval,
        m_override=m_new, c_override=c_new,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(sol_true.t, sol_true.y[0], 'C1', linewidth=1.2, label='Actual')
    axes[0].plot(t_hm, x_hm, 'C0--', linewidth=1.0, label='Hybrid Model')
    axes[0].set_ylabel('x [m]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Modified params: m={m_new} kg, c={c_new} Ns/m')

    axes[1].plot(sol_true.t, sol_true.y[1], 'C1', linewidth=1.2, label='Actual')
    axes[1].plot(t_hm, xd_hm, 'C0--', linewidth=1.0, label='Hybrid Model')
    axes[1].set_ylabel('x_dot [m/s]')
    axes[1].set_xlabel('t [s]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Fig. 8 -- Forward dynamics with new physical parameters')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fig8_modified_params.png'), dpi=150)
    plt.close(fig)


def plot_linearization(model):
    """Linearize the NN spring at x=0, extract effective k."""
    x0 = torch.tensor([0.0], dtype=torch.float32, device=DEVICE, requires_grad=True)
    g = model.nn_unknown(x0.unsqueeze(-1)).squeeze(-1)
    g.backward()
    k_lin = x0.grad.item()

    print(f"\n  Linearization at x=0:")
    print(f"    NN spring gradient dk/dx|_0 = {k_lin:.6f}  (true k = {K_TRUE})")

    x_range = np.linspace(-3, 3, 200)
    actual = K_TRUE * x_range + D_TRUE * x_range**3
    x_t = torch.tensor(x_range, dtype=torch.float32, device=DEVICE)
    nn_spring = model.get_nn_spring_force(x_t).cpu().numpy()

    with torch.no_grad():
        g0 = model.nn_unknown(torch.zeros(1, 1, device=DEVICE)).item()
    linearized = g0 + k_lin * x_range

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_range, actual, 'C1', linewidth=2, label='Actual spring')
    ax.plot(x_range, nn_spring, 'C0--', linewidth=1.5, label='NN spring')
    ax.plot(x_range, linearized, 'C2:', linewidth=1.5, label=f'Linearized (k={k_lin:.4f})')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('Spring force [N]')
    ax.set_title('Spring linearization at equilibrium')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'fig_linearization.png'), dpi=150)
    plt.close(fig)


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    print("=" * 60)
    print("PEML Reproduction -- Experiment 1: Duffing Oscillator")
    print("=" * 60)

    # 1. Data
    train, test, train_np, test_np = prepare_data()
    print(f"  Train samples: {len(train['x'])}")
    print(f"  Test samples:  {len(test['x'])}")

    # 2. Model
    model = DuffingHybridModel(hidden_layers=2, neurons=20).to(DEVICE)
    print(f"\n  Model on: {DEVICE}")
    print(f"  Initial m={model.m.item():.4f}, c={model.c.item():.4f}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params}")

    # 3. Train
    print(f"\nTraining (6000 epochs, Adam, cosine LR 1e-3 -> 1e-5)...")
    losses = train_model(model, train, n_epochs=6000, lr=1e-3)

    # 4. Results
    m_hat = model.m.item()
    c_hat = model.c.item()
    print(f"\n{'='*60}")
    print(f"Identified parameters:")
    print(f"  m_hat = {m_hat:.7f}  (true: {M_TRUE})")
    print(f"  c_hat = {c_hat:.10f}  (true: {C_TRUE})")
    print(f"  m error: {abs(m_hat - M_TRUE)/M_TRUE*100:.6f}%")
    print(f"  c error: {abs(c_hat - C_TRUE)/C_TRUE*100:.4f}%")
    print(f"{'='*60}")

    # 5. Test evaluation
    print("\nEvaluating on test set...")
    error, tau_pred, tau_actual = evaluate(model, test)

    # 6. All plots
    print("\nGenerating figures...")
    plot_training_loss(losses)
    print("  [OK] Training loss curve")

    plot_fig4a_error_histogram(error)
    print("  [OK] Fig. 4a -- Error histogram")

    plot_fig4b_force_comparison(test_np, tau_pred)
    print("  [OK] Fig. 4b -- Force comparison")

    plot_fig5_spring_comparison(model)
    print("  [OK] Fig. 5 -- Spring comparison")

    plot_fig7_forward_dynamics(model)
    print("  [OK] Fig. 7 -- Forward dynamics")

    plot_fig8_modified_params(model)
    print("  [OK] Fig. 8 -- Modified params dynamics")

    plot_linearization(model)
    print("  [OK] Linearization analysis")

    # 7. Save model
    ckpt_path = os.path.join(RESULTS_DIR, 'duffing_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'm_identified': m_hat,
        'c_identified': c_hat,
        'losses': losses,
    }, ckpt_path)
    print(f"\n  Model saved to {ckpt_path}")

    print(f"\nAll figures saved to {RESULTS_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
