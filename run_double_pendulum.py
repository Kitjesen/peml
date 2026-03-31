# -*- coding: utf-8 -*-
"""
Double Pendulum PEML -- RK4 trajectory matching with multiple shooting.

Key insight from the original MATLAB repo: train by integrating trajectory
segments and comparing L2 norm with measured data. We use:
  - Custom RK4 integrator (differentiable, fixed dt=0.01)
  - Multiple shooting: 0.2s windows from random positions in training data
  - Physics fixed from paper, NN learns friction + unmodeled effects
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

from double_pendulum_data import PAPER_PARAMS
from models import NeuralNetwork

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'double_pendulum')
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DT_INTEGRATE = 0.01  # RK4 step size (100 Hz)
WINDOW_STEPS = 20    # 0.2s windows
WINDOW_SEC = WINDOW_STEPS * DT_INTEGRATE


# -----------------------------------------------
# 1. ODE model
# -----------------------------------------------
class DoublePendulumODE(nn.Module):
    def __init__(self, nn_hidden=2, nn_neurons=32):
        super().__init__()
        self.m1 = PAPER_PARAMS['m1']
        self.m2 = PAPER_PARAMS['m2']
        self.a1 = PAPER_PARAMS['a1']
        self.a2 = PAPER_PARAMS['a2']
        self.L1 = PAPER_PARAMS['L1']
        self.I1 = PAPER_PARAMS['I1']
        self.I2 = PAPER_PARAMS['I2']
        self.g  = PAPER_PARAMS['g']
        self.nn_friction = NeuralNetwork(4, 2, nn_hidden, nn_neurons)

    def forward(self, t, y):
        th1, th2, dth1, dth2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        m1, m2, a1, a2, L1 = self.m1, self.m2, self.a1, self.a2, self.L1
        I1, I2, g = self.I1, self.I2, self.g

        cd = torch.cos(th1 - th2)
        sd = torch.sin(th1 - th2)

        M11 = I1 + I2 + m1*a1**2 + m2*(L1**2 + a2**2) + 2*m2*L1*a2*cd
        M12 = I2 + m2*a2**2 + m2*L1*a2*cd
        M22 = I2 + m2*a2**2

        c1 = -m2*L1*a2*sd*(2*dth1*dth2 + dth2**2)
        c2 =  m2*L1*a2*sd*dth1**2
        g1 = (m1*a1 + m2*L1)*g*torch.sin(th1)
        g2 = m2*a2*g*torch.sin(th2)

        nn_in = torch.stack([th1, th2, dth1, dth2], dim=-1)
        tau_nn = self.nn_friction(nn_in)

        rhs1 = -(c1 + g1 + tau_nn[..., 0])
        rhs2 = -(c2 + g2 + tau_nn[..., 1])

        det = M11*M22 - M12*M12
        ddth1 = (M22*rhs1 - M12*rhs2) / det
        ddth2 = (-M12*rhs1 + M11*rhs2) / det

        return torch.stack([dth1, dth2, ddth1, ddth2], dim=-1)


def rk4_integrate(model, y0, dt, n_steps):
    y = y0
    traj = [y]
    t = torch.tensor(0.0, device=y0.device)
    for _ in range(n_steps):
        k1 = model(t, y)
        k2 = model(t + dt/2, y + dt/2 * k1)
        k3 = model(t + dt/2, y + dt/2 * k2)
        k4 = model(t + dt, y + dt * k3)
        y = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + dt
        traj.append(y)
    return torch.stack(traj)  # (n_steps+1, 4)


# -----------------------------------------------
# 2. Data
# -----------------------------------------------
def load_data():
    import scipy.io as sio
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'multi-pendulum-data', 'ParameterEstimation', 'DoublePendulum')
    d = sio.loadmat(os.path.join(data_dir, 'DoublePendulumDataForParameterEstimation.mat'))
    dt_raw = d['dt'].item()  # 0.001

    def extract(Y_cell):
        segs = []
        for i in range(Y_cell.shape[0]):
            seg = Y_cell[i, 0].T  # (T, 4)
            # Subsample to 100Hz for RK4 dt=0.01
            seg_100hz = seg[::10]
            segs.append(torch.tensor(seg_100hz, dtype=torch.float32, device=DEVICE))
        return segs

    train_segs = extract(d['Y_id'])
    val_segs = extract(d['Y_vad'])
    print(f"  {len(train_segs)} train + {len(val_segs)} val segments")
    print(f"  Segment length: {train_segs[0].shape[0]} steps at 100Hz = {train_segs[0].shape[0]*DT_INTEGRATE:.1f}s")
    return train_segs, val_segs


# -----------------------------------------------
# 3. Training: multiple shooting with RK4
# -----------------------------------------------
def train(model, train_segs, n_epochs=2000, lr=1e-3, windows_per_epoch=6):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    # Pre-compute all valid window starting indices
    all_windows = []
    for seg in train_segs:
        max_start = seg.shape[0] - WINDOW_STEPS - 1
        for s in range(0, max_start, WINDOW_STEPS // 2):  # 50% overlap
            all_windows.append((seg, s))
    print(f"  Total windows: {len(all_windows)} ({WINDOW_SEC}s each)")

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        # Random sample of windows
        idxs = np.random.choice(len(all_windows), windows_per_epoch, replace=False)

        for idx in idxs:
            seg, start = all_windows[idx]
            gt = seg[start:start + WINDOW_STEPS + 1]  # (21, 4)
            y0 = gt[0]

            optimizer.zero_grad()
            pred = rk4_integrate(model, y0, DT_INTEGRATE, WINDOW_STEPS)
            loss = torch.mean((pred - gt)**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / windows_per_epoch
        losses.append(avg)

        if (epoch + 1) % 200 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg:.6e} | lr={lr_now:.2e}")

    return losses


# -----------------------------------------------
# 4. Evaluation
# -----------------------------------------------
def predict_long(model, y0, duration_s):
    n_steps = int(duration_s / DT_INTEGRATE)
    model.eval()
    with torch.no_grad():
        traj = rk4_integrate(model, y0, DT_INTEGRATE, n_steps)
    return traj.cpu().numpy()


def physics_only_predict(y0_np, duration_s):
    P = PAPER_PARAMS

    def rhs(t, y):
        th1, th2, dth1, dth2 = y
        m1, m2, a1, a2, L1 = P['m1'], P['m2'], P['a1'], P['a2'], P['L1']
        I1, I2, g, k1, k2 = P['I1'], P['I2'], P['g'], P['k1'], P['k2']
        cd, sd = np.cos(th1-th2), np.sin(th1-th2)
        M11 = I1+I2+m1*a1**2+m2*(L1**2+a2**2)+2*m2*L1*a2*cd
        M12 = I2+m2*a2**2+m2*L1*a2*cd
        M22 = I2+m2*a2**2
        c1 = -m2*L1*a2*sd*(2*dth1*dth2+dth2**2)
        c2 = m2*L1*a2*sd*dth1**2
        g1 = (m1*a1+m2*L1)*g*np.sin(th1)
        g2 = m2*a2*g*np.sin(th2)
        f1 = k1*dth1+k2*(dth1-dth2)
        f2 = k2*(dth2-dth1)
        det = M11*M22-M12**2
        return [dth1, dth2, (M22*(-(c1+g1+f1))-M12*(-(c2+g2+f2)))/det,
                (-M12*(-(c1+g1+f1))+M11*(-(c2+g2+f2)))/det]

    t_eval = np.arange(0, duration_s, DT_INTEGRATE)
    sol = solve_ivp(rhs, [0, duration_s], y0_np, t_eval=t_eval, method='DOP853', rtol=1e-8, atol=1e-10)
    return sol.y.T


# -----------------------------------------------
# 5. Plots
# -----------------------------------------------
def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(losses, alpha=0.2, color='C0', linewidth=0.5)
    ema = [losses[0]]
    for l in losses[1:]:
        ema.append(0.02*l + 0.98*ema[-1])
    ax.semilogy(ema, color='C0', linewidth=1.5, label='EMA')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.set_title('Training Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, 'training_loss.png'), dpi=150); plt.close(fig)


def plot_trajectory(model, val_segs, seg_idx=0):
    seg = val_segs[seg_idx]
    duration = seg.shape[0] * DT_INTEGRATE
    seg_np = seg.cpu().numpy()
    y0 = seg[0]
    y0_np = y0.cpu().numpy()
    t_np = np.arange(seg.shape[0]) * DT_INTEGRATE

    y_hybrid = predict_long(model, y0, duration - DT_INTEGRATE)
    y_physics = physics_only_predict(y0_np, duration)

    labels = ['theta1 (rad)', 'theta2 (rad)', 'omega1 (rad/s)', 'omega2 (rad/s)']
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.plot(t_np, seg_np[:, i], 'C0', linewidth=1.2, label='Measured')
        n_ph = min(len(t_np), y_physics.shape[0])
        ax.plot(t_np[:n_ph], y_physics[:n_ph, i], 'C1--', linewidth=0.8, alpha=0.7, label='Physics Only')
        n_hy = min(len(t_np), y_hybrid.shape[0])
        ax.plot(t_np[:n_hy], y_hybrid[:n_hy, i], 'C2', linewidth=1.0, alpha=0.9, label='Hybrid')
        ax.set_ylabel(lbl); ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(ncol=3)
    axes[-1].set_xlabel('time (s)')
    fig.suptitle(f'Validation trajectory {seg_idx+1}: full-time comparison')
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, 'forward_dynamics.png'), dpi=150); plt.close(fig)


def plot_friction(model):
    model.eval()
    dth = np.linspace(-8, 8, 200)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ji, lbl in [(axes[0], 0, 'Joint 1'), (axes[1], 1, 'Joint 2')]:
        tau = []
        for v in dth:
            inp = torch.tensor([[np.pi, np.pi, v if ji==0 else 0, 0 if ji==0 else v]],
                               dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                tau.append(model.nn_friction(inp)[0, ji].item())
        ax.plot(dth, tau, 'C0', linewidth=2)
        k = PAPER_PARAMS[f'k{ji+1}']
        ax.plot(dth, k*dth, 'C1--', linewidth=1.5, label=f'Linear (k={k:.6f})')
        ax.set_xlabel('Angular velocity (rad/s)'); ax.set_ylabel('Friction torque (Nm)')
        ax.set_title(f'{lbl} -- Learned friction'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, 'nn_friction.png'), dpi=150); plt.close(fig)


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    print("=" * 60)
    print("PEML Double Pendulum -- RK4 trajectory matching")
    print("=" * 60)

    train_segs, val_segs = load_data()

    model = DoublePendulumODE(nn_hidden=2, nn_neurons=32).to(DEVICE)
    print(f"  {sum(p.numel() for p in model.parameters())} params, device={DEVICE}")

    print(f"\nTraining (RK4, {WINDOW_SEC}s windows, multiple shooting)...")
    losses = train(model, train_segs, n_epochs=2000, lr=1e-3, windows_per_epoch=6)

    print(f"\n  Final loss: {losses[-1]:.6e}")
    print("\nGenerating figures...")
    plot_loss(losses); print("  [OK] Training loss")
    plot_trajectory(model, val_segs); print("  [OK] Forward dynamics")
    plot_friction(model); print("  [OK] NN friction")

    torch.save({'model_state_dict': model.state_dict(), 'losses': losses},
               os.path.join(RESULTS_DIR, 'double_pendulum_model.pt'))
    print(f"  Saved to {RESULTS_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
