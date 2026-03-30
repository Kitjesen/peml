# -*- coding: utf-8 -*-
"""
Experiment 2: Double Pendulum PEML -- real experimental data.

Uses the MultiArm-Pendulum dataset (Kaheman et al., 2022).
Identifies physical parameters + learns nonlinear friction from data.
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

from double_pendulum_data import prepare_dataset, PAPER_PARAMS
from double_pendulum_model import DoublePendulumHybridModel

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'double_pendulum')
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------------------------
# 1. Data
# -----------------------------------------------
def prepare_tensors():
    train_np, val_np = prepare_dataset(
        train_indices=(1, 2, 3), val_index=4,
        subsample_factor=10, smooth_window=7,
    )

    def to_tensors(d):
        out = {k: torch.tensor(v, dtype=torch.float32, device=DEVICE)
               for k, v in d.items() if isinstance(v, np.ndarray)}
        out['dt'] = d['dt']  # keep scalar
        return out

    return to_tensors(train_np), to_tensors(val_np), train_np, val_np


# -----------------------------------------------
# 2. Training
# -----------------------------------------------
def train_model(model, train, batch_size=2048):
    """
    Single-step prediction training (no finite-difference accelerations needed).

    Instead of fitting residuals M*ddq + C*dq + G = -friction, we:
      1. Predict ddq from model at time t
      2. Semi-implicit Euler: dq(t+dt) = dq(t) + ddq*dt, q(t+dt) = q(t) + dq(t+dt)*dt
      3. Compare predicted (q, dq) at t+dt with actual measurements

    Two phases:
      Phase 1: Physics only (NN frozen) -- 3000 epochs
      Phase 2: Joint physics + NN -- 5000 epochs with NN regularization
    """
    # Build consecutive pairs: (state_t, state_{t+1})
    th1 = train['theta1']
    th2 = train['theta2']
    dth1 = train['dtheta1']
    dth2 = train['dtheta2']
    dt = train['dt'] * 10  # subsampled dt (0.01s)
    n = len(th1) - 1  # pairs

    def step_loss(idx, nn_reg=0.0):
        """Compute single-step prediction loss for batch indices."""
        # Current state
        t1 = th1[idx]; t2 = th2[idx]; d1 = dth1[idx]; d2 = dth2[idx]
        # Next state (ground truth)
        t1_next = th1[idx + 1]; t2_next = th2[idx + 1]
        d1_next = dth1[idx + 1]; d2_next = dth2[idx + 1]

        # Predict acceleration
        dd1, dd2 = model.predict_accel(t1, t2, d1, d2)

        # Semi-implicit Euler integration
        d1_pred = d1 + dd1 * dt
        d2_pred = d2 + dd2 * dt
        t1_pred = t1 + d1_pred * dt
        t2_pred = t2 + d2_pred * dt

        # Loss: prediction error on (theta, dtheta)
        loss = (torch.mean((t1_pred - t1_next)**2 + (t2_pred - t2_next)**2)
                + 0.01 * torch.mean((d1_pred - d1_next)**2 + (d2_pred - d2_next)**2))

        if nn_reg > 0:
            nn_input = torch.stack([t1, t2, d1, d2], dim=-1)
            tau_nn = model.nn_friction(nn_input)
            loss = loss + nn_reg * torch.mean(tau_nn**2)

        return loss

    def run_phase(params, n_epochs, lr, label, nn_reg=0.0):
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-6)
        phase_losses = []
        for epoch in range(n_epochs):
            model.train()
            idx = torch.randint(0, n, (batch_size,), device=DEVICE)
            optimizer.zero_grad()
            loss = step_loss(idx, nn_reg=nn_reg)
            loss.backward()
            optimizer.step()
            scheduler.step()
            phase_losses.append(loss.item())
            if (epoch + 1) % 500 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  [{label}] Epoch {epoch+1:4d}/{n_epochs} | "
                      f"Loss: {loss.item():.6e} | lr={cur_lr:.2e}")
                print(f"    m1={model.m1.item():.5f} m2={model.m2.item():.5f} "
                      f"a1={model.a1.item():.5f} a2={model.a2.item():.5f} "
                      f"I1={model.I1.item():.6f} I2={model.I2.item():.6f}")
        return phase_losses

    # Physics fixed, only train NN to learn friction + residual
    print("  Training NN only (physics fixed from paper)...")
    nn_params = list(model.nn_friction.parameters())
    losses = run_phase(nn_params, n_epochs=6000, lr=1e-3, label="NN", nn_reg=0.0)

    return losses


# -----------------------------------------------
# 3. Evaluation
# -----------------------------------------------
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        res1, res2, tau1_nn, tau2_nn = model(
            data['theta1'], data['theta2'],
            data['dtheta1'], data['dtheta2'],
            data['ddtheta1'], data['ddtheta2'],
        )
    return {
        'res1': res1.cpu().numpy(),
        'res2': res2.cpu().numpy(),
        'tau1_nn': tau1_nn.cpu().numpy(),
        'tau2_nn': tau2_nn.cpu().numpy(),
    }


def forward_integrate(model, y0, t_span, t_eval):
    """Integrate the identified model forward in time."""
    model.eval()

    def rhs(t, y):
        th1, th2, dth1, dth2 = y
        with torch.no_grad():
            t_th1 = torch.tensor([th1], dtype=torch.float32, device=DEVICE)
            t_th2 = torch.tensor([th2], dtype=torch.float32, device=DEVICE)
            t_dth1 = torch.tensor([dth1], dtype=torch.float32, device=DEVICE)
            t_dth2 = torch.tensor([dth2], dtype=torch.float32, device=DEVICE)
            ddth1, ddth2 = model.predict_accel(t_th1, t_th2, t_dth1, t_dth2)
        return [dth1, dth2, ddth1.item(), ddth2.item()]

    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-8, atol=1e-10, max_step=0.01)
    return sol


# -----------------------------------------------
# 4. Visualization
# -----------------------------------------------
def plot_training_loss(losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(losses, alpha=0.15, color='C0', linewidth=0.5)
    alpha_ema = 0.01
    ema = [losses[0]]
    for l in losses[1:]:
        ema.append(alpha_ema * l + (1 - alpha_ema) * ema[-1])
    ax.semilogy(ema, color='C0', linewidth=1.5, label='Smoothed (EMA)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Residual Loss')
    ax.set_title('Double Pendulum -- Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'training_loss.png'), dpi=150)
    plt.close(fig)


def plot_residual_histogram(eval_result):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, label in [(axes[0], 'res1', 'Joint 1'), (axes[1], 'res2', 'Joint 2')]:
        r = eval_result[key]
        mu, sigma = np.mean(r), np.std(r)
        ax.hist(r, bins=100, color='steelblue', edgecolor='white', linewidth=0.3)
        ax.set_title(f'{label} residual: mu={mu:.4e}, sigma={sigma:.4e}')
        ax.set_xlabel('Residual torque [Nm]')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'residual_histogram.png'), dpi=150)
    plt.close(fig)


def plot_nn_friction(eval_result, val_np):
    """Plot the learned friction torques vs angular velocity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, dth_key, tau_key, label in [
        (axes[0], 'dtheta1', 'tau1_nn', 'Joint 1'),
        (axes[1], 'dtheta2', 'tau2_nn', 'Joint 2'),
    ]:
        dth = val_np[dth_key]
        tau = eval_result[tau_key]
        # Subsample for scatter plot
        idx = np.random.choice(len(dth), min(5000, len(dth)), replace=False)
        ax.scatter(dth[idx], tau[idx], s=1, alpha=0.3, c='C0')
        ax.set_xlabel('Angular velocity [rad/s]')
        ax.set_ylabel('NN friction torque [Nm]')
        ax.set_title(f'{label} -- Learned friction')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'nn_friction.png'), dpi=150)
    plt.close(fig)


def plot_forward_dynamics(model, val_np, duration=5.0):
    """Compare forward integration vs actual trajectory."""
    # Take first 'duration' seconds of validation data
    dt_val = val_np['dt'] * 10  # subsampled dt
    n_steps = int(duration / dt_val)
    t_actual = val_np['t'][:n_steps]
    t_actual = t_actual - t_actual[0]  # start at 0

    y0 = [val_np['theta1'][0], val_np['theta2'][0],
          val_np['dtheta1'][0], val_np['dtheta2'][0]]

    sol = forward_integrate(model, y0, [0, duration], t_actual)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t_actual, val_np['theta1'][:n_steps], 'C1', linewidth=1.2, label='Actual')
    axes[0].plot(sol.t, sol.y[0], 'C0--', linewidth=1.0, label='Hybrid Model')
    axes[0].set_ylabel('theta1 [rad]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_actual, val_np['theta2'][:n_steps], 'C1', linewidth=1.2, label='Actual')
    axes[1].plot(sol.t, sol.y[1], 'C0--', linewidth=1.0, label='Hybrid Model')
    axes[1].set_ylabel('theta2 [rad]')
    axes[1].set_xlabel('t [s]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Forward dynamics ({duration}s) -- actual vs identified HM')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'forward_dynamics.png'), dpi=150)
    plt.close(fig)


def plot_param_comparison(model):
    """Compare identified vs paper-estimated parameters."""
    def _get(name):
        prop = getattr(model, name)
        return prop.item() if isinstance(prop, torch.Tensor) else prop
    params = {n: (_get(n), PAPER_PARAMS[n])
              for n in ['m1', 'm2', 'a1', 'a2', 'L1', 'I1', 'I2', 'g']}

    names = list(params.keys())
    ours = [params[n][0] for n in names]
    paper = [params[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, ours, width, label='PEML (ours)', color='C0')
    ax.bar(x + width/2, paper, width, label='Paper estimate', color='C1')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Parameter value')
    ax.set_title('Identified parameters: PEML vs original paper')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (o, p) in enumerate(zip(ours, paper)):
        ax.text(i - width/2, o, f'{o:.5f}', ha='center', va='bottom', fontsize=7)
        ax.text(i + width/2, p, f'{p:.5f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'param_comparison.png'), dpi=150)
    plt.close(fig)


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    print("=" * 60)
    print("PEML -- Double Pendulum (real experimental data)")
    print("=" * 60)

    # 1. Data
    train, val, train_np, val_np = prepare_tensors()
    print(f"  Train: {len(train['theta1'])} samples")
    print(f"  Val:   {len(val['theta1'])} samples")

    # 2. Model
    model = DoublePendulumHybridModel(nn_hidden=2, nn_neurons=32).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model on: {DEVICE}, {total_params} parameters")

    # 3. Train
    print(f"\nTraining NN only (6000 epochs, physics fixed)...")
    losses = train_model(model, train, batch_size=2048)

    # 4. Print identified parameters
    print(f"\n{'='*60}")
    print("Identified parameters vs paper:")
    print("  All physics parameters fixed from paper estimates.")
    print("  NN learns: nonlinear friction + unmodeled effects.")
    print(f"{'='*60}")

    # 5. Evaluate
    print("\nEvaluating on validation set...")
    eval_train = evaluate(model, train)
    eval_val = evaluate(model, val)

    # 6. Plots
    print("\nGenerating figures...")

    plot_training_loss(losses)
    print("  [OK] Training loss")

    plot_residual_histogram(eval_val)
    print("  [OK] Residual histogram")

    plot_nn_friction(eval_val, val_np)
    print("  [OK] NN friction curves")

    plot_param_comparison(model)
    print("  [OK] Parameter comparison")

    plot_forward_dynamics(model, val_np, duration=5.0)
    print("  [OK] Forward dynamics (5s)")

    # 7. Save
    ckpt_path = os.path.join(RESULTS_DIR, 'double_pendulum_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'params': {name: (getattr(model, name).item()
                         if isinstance(getattr(model, name), torch.Tensor)
                         else getattr(model, name))
                   for name in ['m1', 'm2', 'a1', 'a2', 'L1', 'I1', 'I2', 'g']},
    }, ckpt_path)
    print(f"\n  Model saved to {ckpt_path}")
    print(f"  Figures saved to {RESULTS_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
