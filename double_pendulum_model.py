# -*- coding: utf-8 -*-
"""
Double pendulum hybrid model for PEML.

Known physics (Lagrangian mechanics):
    M(q) * ddq + C(q, dq) * dq + G(q) = tau_friction

    where M is the mass matrix, C the Coriolis/centrifugal matrix,
    G the gravity vector. Structure is known, parameters are trainable.

Unknown physics:
    Joint friction/damping -- the paper uses linear k1*dtheta1, k2*dtheta2
    but real friction is nonlinear (Coulomb, Stribeck, viscous).
    We use a NN to learn this.

For the mounted (cart-fixed) double pendulum with no external torque:
    M(q) * ddq + C(q, dq) * dq + G(q) + f_friction(q, dq) = 0

Rearranged for identification (predict accelerations):
    ddq = M(q)^{-1} * (-C(q, dq)*dq - G(q) - f_friction(q, dq))

Or equivalently, predict the residual torque:
    tau_residual = M(q)*ddq + C(q,dq)*dq + G(q) = -f_friction(q, dq)
"""

import torch
import torch.nn as nn
import numpy as np
from models import NeuralNetwork


class DoublePendulumHybridModel(nn.Module):
    """
    Hybrid model for mounted double pendulum.

    Trainable physical parameters: m1, m2, a1, a2, L1, I1, I2, g
    NN learns: nonlinear friction f(theta1, theta2, dtheta1, dtheta2)

    The model predicts angular accelerations [ddtheta1, ddtheta2].
    """

    def __init__(self, nn_hidden=2, nn_neurons=32):
        super().__init__()

        # All physical parameters FIXED from paper measurements.
        # Only NN is trainable -- learns friction + unmodeled effects.
        self.register_buffer('_m1', torch.tensor(0.093844))
        self.register_buffer('_m2', torch.tensor(0.137596))
        self.register_buffer('_a1', torch.tensor(0.108565))
        self.register_buffer('_a2', torch.tensor(0.116779))
        self.register_buffer('_L1', torch.tensor(0.172719))
        self.register_buffer('_I1', torch.tensor(0.000438))
        self.register_buffer('_I2', torch.tensor(0.001269))
        self.register_buffer('_g',  torch.tensor(9.808580))

        # NN for unknown friction: input=(theta1, theta2, dtheta1, dtheta2) -> (tau1, tau2)
        self.nn_friction = NeuralNetwork(
            input_dim=4, output_dim=2,
            hidden_layers=nn_hidden,
            neurons_per_layer=nn_neurons,
        )

    @property
    def m1(self): return self._m1
    @property
    def m2(self): return self._m2
    @property
    def a1(self): return self._a1
    @property
    def a2(self): return self._a2
    @property
    def L1(self): return self._L1
    @property
    def I1(self): return self._I1
    @property
    def I2(self): return self._I2
    @property
    def g(self): return self._g

    def mass_matrix(self, theta1, theta2):
        """2x2 mass matrix M(q), returned as (M11, M12, M21, M22) per sample."""
        m1, m2 = self.m1, self.m2
        a1, a2, L1 = self.a1, self.a2, self.L1
        I1, I2 = self.I1, self.I2

        cos_diff = torch.cos(theta1 - theta2)

        M11 = I1 + I2 + m1 * a1**2 + m2 * (L1**2 + a2**2) + 2 * m2 * L1 * a2 * cos_diff
        M12 = I2 + m2 * a2**2 + m2 * L1 * a2 * cos_diff
        M21 = M12
        M22 = I2 + m2 * a2**2

        return M11, M12, M21, M22

    def coriolis_gravity(self, theta1, theta2, dtheta1, dtheta2):
        """Compute C(q,dq)*dq + G(q), returned as (h1, h2) per sample."""
        m2 = self.m2
        a1, a2, L1 = self.a1, self.a2, self.L1
        g = self.g
        m1 = self.m1

        sin_diff = torch.sin(theta1 - theta2)
        cos_diff = torch.cos(theta1 - theta2)

        # Coriolis/centrifugal terms
        h1 = (-m2 * L1 * a2 * sin_diff * (2 * dtheta1 * dtheta2 + dtheta2**2)
               + (m1 * a1 + m2 * L1) * g * torch.sin(theta1)
               + m2 * a2 * g * torch.sin(theta2) * cos_diff)
        # Correction: use proper Lagrangian derivation
        h1 = (-m2 * L1 * a2 * dtheta2**2 * sin_diff
               - m2 * L1 * a2 * 2 * dtheta1 * dtheta2 * sin_diff
               + (m1 * a1 + m2 * L1) * g * torch.sin(theta1))

        # Actually, let's use the standard form from the MATLAB ODE.
        # For the residual approach, we compute: tau = M*ddq + h(q, dq)
        # where h includes Coriolis + gravity. Rather than deriving h separately,
        # we'll use the forward model directly.

        # Simpler: just return gravity terms and let Coriolis be in the full forward
        return h1, None  # placeholder, we use full forward instead

    def forward(self, theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2):
        """
        Predict residual: tau_residual = M*ddq + C*dq + G(q)
        The NN should learn: tau_residual = -f_friction(q, dq)

        This is the "inverse dynamics" formulation for identification.
        """
        M11, M12, M21, M22 = self.mass_matrix(theta1, theta2)

        m1, m2 = self.m1, self.m2
        a1, a2, L1 = self.a1, self.a2, self.L1
        g = self.g

        sin_diff = torch.sin(theta1 - theta2)

        # Coriolis + centrifugal
        c1 = -m2 * L1 * a2 * sin_diff * (2 * dtheta1 * dtheta2 + dtheta2**2)
        c2 =  m2 * L1 * a2 * sin_diff * dtheta1**2

        # Gravity
        g1 = (m1 * a1 + m2 * L1) * g * torch.sin(theta1)
        g2 = m2 * a2 * g * torch.sin(theta2)

        # Known physics: M*ddq + C*dq + G = tau_known
        tau1_known = M11 * ddtheta1 + M12 * ddtheta2 + c1 + g1
        tau2_known = M21 * ddtheta1 + M22 * ddtheta2 + c2 + g2

        # NN friction prediction
        nn_input = torch.stack([theta1, theta2, dtheta1, dtheta2], dim=-1)
        tau_nn = self.nn_friction(nn_input)  # [batch, 2]
        tau1_nn = tau_nn[:, 0]
        tau2_nn = tau_nn[:, 1]

        # Total: known_torque + nn_friction = 0 (free swing, no external torque)
        # So residual = tau_known + tau_nn should be ~0
        residual1 = tau1_known + tau1_nn
        residual2 = tau2_known + tau2_nn

        return residual1, residual2, tau1_nn, tau2_nn

    def predict_accel(self, theta1, theta2, dtheta1, dtheta2):
        """
        Predict accelerations for forward dynamics integration.
        ddq = M^{-1} * (-C*dq - G - f_nn)
        """
        M11, M12, M21, M22 = self.mass_matrix(theta1, theta2)

        m1, m2 = self.m1, self.m2
        a1, a2, L1 = self.a1, self.a2, self.L1
        g = self.g

        sin_diff = torch.sin(theta1 - theta2)

        c1 = -m2 * L1 * a2 * sin_diff * (2 * dtheta1 * dtheta2 + dtheta2**2)
        c2 =  m2 * L1 * a2 * sin_diff * dtheta1**2

        g1 = (m1 * a1 + m2 * L1) * g * torch.sin(theta1)
        g2 = m2 * a2 * g * torch.sin(theta2)

        # NN friction
        nn_input = torch.stack([theta1, theta2, dtheta1, dtheta2], dim=-1)
        tau_nn = self.nn_friction(nn_input)
        tau1_nn = tau_nn[:, 0]
        tau2_nn = tau_nn[:, 1]

        # RHS = -(C*dq + G + f_nn)
        rhs1 = -(c1 + g1 + tau1_nn)
        rhs2 = -(c2 + g2 + tau2_nn)

        # Solve M * ddq = rhs via 2x2 inverse
        det = M11 * M22 - M12 * M21
        ddtheta1 = (M22 * rhs1 - M12 * rhs2) / det
        ddtheta2 = (-M21 * rhs1 + M11 * rhs2) / det

        return ddtheta1, ddtheta2
