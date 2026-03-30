"""
Hybrid Model framework for Physics-Enhanced Machine Learning (PEML).

Architecture:
    K(q, q̇, q̈) = f(q, q̇, q̈, θ) + N_φ(inputs, φ) = y

    f: known physics with trainable physical parameters θ
    N_φ: feed-forward NN approximating unknown physics g
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """Feed-forward neural network for modelling unknown physics."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_layers: int = 2, neurons_per_layer: int = 20):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, neurons_per_layer))
            layers.append(nn.Tanh())
            in_dim = neurons_per_layer
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuffingHybridModel(nn.Module):
    """
    Hybrid model for the Duffing oscillator (Section 3).

    True system:  m*x̄ + c*ẋ + k*x + d*x³ = τ
    Known physics: f(x, ẋ, x̄, θ) = m*x̄ + c*ẋ    (θ = {m, c})
    Unknown:       g(x) = k*x + d*x³               (approximated by NN)
    Output:        τ̂ = m̂*x̄ + ĉ*ẋ + N_φ(x)
    """

    def __init__(self, hidden_layers: int = 2, neurons: int = 20):
        super().__init__()
        # Trainable physical parameters (initialized with rough guesses)
        self.m = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.05))
        # NN for unknown spring: input=x, output=spring_force
        self.nn_unknown = NeuralNetwork(
            input_dim=1, output_dim=1,
            hidden_layers=hidden_layers,
            neurons_per_layer=neurons,
        )

    def forward(self, x: torch.Tensor, x_dot: torch.Tensor,
                x_ddot: torch.Tensor) -> torch.Tensor:
        """
        Predict external force τ.

        Args:
            x:      displacement [batch]
            x_dot:  velocity [batch]
            x_ddot: acceleration [batch]
        Returns:
            τ̂:     predicted external force [batch]
        """
        # Known physics: f = m*x̄ + c*ẋ
        f_known = self.m * x_ddot + self.c * x_dot
        # Unknown physics: g ≈ N_φ(x)
        g_nn = self.nn_unknown(x.unsqueeze(-1)).squeeze(-1)
        return f_known + g_nn

    def get_nn_spring_force(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the NN-learned spring force for visualization."""
        with torch.no_grad():
            return self.nn_unknown(x.unsqueeze(-1)).squeeze(-1)

    def forward_dynamics_accel(self, x: torch.Tensor, x_dot: torch.Tensor,
                               tau: torch.Tensor) -> torch.Tensor:
        """
        Rearranged model for forward integration (Eq. 12):
            x̄ = (1/m̂) * (τ - ĉ*ẋ - N_φ(x))
        """
        g_nn = self.nn_unknown(x.unsqueeze(-1)).squeeze(-1)
        return (tau - self.c * x_dot - g_nn) / self.m
