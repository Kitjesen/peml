"""
Duffing oscillator data generation (Section 3.1, 3.3).

System: m*x̄ + c*ẋ + k*x + d*x³ = τ(t)
True parameters: m=1 kg, c=0.1 Ns/m, k=1 N/m, d=0.1 N/m³

Each simulation: 100s, dt=0.5s → 200 time steps
External load: sinusoidal with random A ∈ U[-10,10] N, f ∈ U[0,5] Hz
"""

import numpy as np
from scipy.integrate import solve_ivp


# True physical parameters
M_TRUE = 1.0        # kg
C_TRUE = 0.1        # Ns/m  (paper says c=0.1, identified ĉ≈0.00999 → paper c=0.01?)
K_TRUE = 1.0        # N/m
D_TRUE = 0.1        # N/m³

# NOTE: The paper states c=0.1 Ns/m in Section 3.1 but reports identified
# ĉ=0.009989 ≈ 0.01. This is likely a typo in the paper — the true value
# should be c=0.01. We use c=0.01 to match the identification results.
C_TRUE = 0.01

T_TOTAL = 100.0     # seconds
DT = 0.5            # time step
N_STEPS = int(T_TOTAL / DT)  # 200


def duffing_rhs(t, state, tau_func, m, c, k, d):
    """Right-hand side of the Duffing ODE."""
    x, v = state
    tau = tau_func(t)
    a = (tau - c * v - k * x - d * x**3) / m
    return [v, a]


def generate_random_sinusoidal_load(rng: np.random.Generator):
    """Generate a random sinusoidal load: tau(t) = A * sin(2*pi*f*t).
    A bounds [-14, 14] to achieve x coverage ~[-7.5, 7.5] matching the paper."""
    A = rng.uniform(-14.0, 14.0)
    f = rng.uniform(0.0, 5.0)
    return lambda t: A * np.sin(2.0 * np.pi * f * t), A, f


def simulate_one(tau_func, x0=0.0, v0=0.0,
                 m=M_TRUE, c=C_TRUE, k=K_TRUE, d=D_TRUE):
    """
    Run one Duffing simulation.

    Returns:
        t_eval: [N_STEPS] time points
        x:      [N_STEPS] displacement
        x_dot:  [N_STEPS] velocity
        x_ddot: [N_STEPS] acceleration
        tau:    [N_STEPS] external force
    """
    t_eval = np.linspace(DT, T_TOTAL, N_STEPS)  # start at DT (skip t=0)

    sol = solve_ivp(
        duffing_rhs, [0, T_TOTAL], [x0, v0],
        args=(tau_func, m, c, k, d),
        t_eval=t_eval, method='RK45',
        rtol=1e-10, atol=1e-12,
    )

    x = sol.y[0]
    x_dot = sol.y[1]
    tau = np.array([tau_func(ti) for ti in t_eval])
    # Compute acceleration from the equation of motion
    x_ddot = (tau - c * x_dot - k * x - d * x**3) / m

    return t_eval, x, x_dot, x_ddot, tau


def generate_dataset(n_simulations: int, seed: int = 42):
    """
    Generate a dataset of Duffing oscillator simulations.

    Returns:
        dict with keys: t, x, x_dot, x_ddot, tau
        each of shape [n_simulations * N_STEPS]
    """
    rng = np.random.default_rng(seed)

    all_t, all_x, all_xd, all_xdd, all_tau = [], [], [], [], []

    for _ in range(n_simulations):
        tau_func, _, _ = generate_random_sinusoidal_load(rng)
        t, x, xd, xdd, tau = simulate_one(tau_func)
        all_t.append(t)
        all_x.append(x)
        all_xd.append(xd)
        all_xdd.append(xdd)
        all_tau.append(tau)

    return {
        't': np.concatenate(all_t),
        'x': np.concatenate(all_x),
        'x_dot': np.concatenate(all_xd),
        'x_ddot': np.concatenate(all_xdd),
        'tau': np.concatenate(all_tau),
    }


if __name__ == '__main__':
    # Quick sanity check
    train = generate_dataset(50, seed=42)
    test = generate_dataset(20, seed=123)
    print(f"Train: {len(train['x'])} samples")
    print(f"Test:  {len(test['x'])} samples")
    print(f"x range: [{train['x'].min():.2f}, {train['x'].max():.2f}]")
    print(f"tau range: [{train['tau'].min():.2f}, {train['tau'].max():.2f}]")
