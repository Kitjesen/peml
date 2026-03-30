# -*- coding: utf-8 -*-
"""
Double pendulum data loading and preprocessing.

Data source: https://github.com/dynamicslab/MultiArm-Pendulum
Kaheman et al. (2022) "The Experimental Multi-Arm Pendulum on a Cart"

The mounted double pendulum (cart fixed) has state [theta1, theta2, dtheta1, dtheta2].
We compute angular accelerations via finite differences for the PEML framework.
"""

import os
import numpy as np
import scipy.io as sio

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'multi-pendulum-data')

# Estimated parameters from the original paper (10 values)
# [m1, m2, a1, a2, L1, I1, I2, k1, k2, g]
PAPER_PARAMS = {
    'm1': 0.093844,   # kg - mass of arm 1
    'm2': 0.137596,   # kg - mass of arm 2
    'a1': 0.108565,   # m  - distance from pivot to COM of arm 1
    'a2': 0.116779,   # m  - distance from pivot to COM of arm 2
    'L1': 0.172719,   # m  - length of arm 1
    'I1': 0.000438,   # kg*m^2 - inertia of arm 1
    'I2': 0.001269,   # kg*m^2 - inertia of arm 2
    'k1': 0.000237,   # N*m*s  - friction coeff joint 1
    'k2': 0.000010,   # N*m*s  - friction coeff joint 2
    'g':  9.808580,   # m/s^2  - gravity
}


def load_freeswing(index=1):
    """Load a free-swing double pendulum dataset.

    Args:
        index: 1-4, which free-swing experiment

    Returns:
        dict with keys: t, theta1, theta2, dtheta1, dtheta2, dt
    """
    path = os.path.join(DATA_DIR, 'Datas', 'DoublePendulum',
                        f'DoubleDataFreeSwing_{index}_Dt_0_001.mat')
    d = sio.loadmat(path)
    return {
        't': d['Time'].flatten(),
        'theta1': d['Theta1'].flatten(),
        'theta2': d['Theta2'].flatten(),
        'dtheta1': d['dTheta1'].flatten(),
        'dtheta2': d['dTheta2'].flatten(),
        'dt': d['dt'].item(),
    }


def compute_accelerations(data, smooth_window=5):
    """Compute angular accelerations via smoothed finite differences.

    Args:
        data: dict from load_freeswing()
        smooth_window: window size for Savitzky-Golay-like smoothing

    Returns:
        dict with added keys: ddtheta1, ddtheta2, and trimmed arrays
    """
    dt = data['dt']

    # Central finite difference for acceleration
    ddtheta1 = np.gradient(data['dtheta1'], dt)
    ddtheta2 = np.gradient(data['dtheta2'], dt)

    # Simple moving average smoothing to reduce noise in accelerations
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        ddtheta1 = np.convolve(ddtheta1, kernel, mode='same')
        ddtheta2 = np.convolve(ddtheta2, kernel, mode='same')

    # Trim edges where finite differences are unreliable
    margin = max(smooth_window, 5)
    sl = slice(margin, -margin)

    return {
        't': data['t'][sl],
        'theta1': data['theta1'][sl],
        'theta2': data['theta2'][sl],
        'dtheta1': data['dtheta1'][sl],
        'dtheta2': data['dtheta2'][sl],
        'ddtheta1': ddtheta1[sl],
        'ddtheta2': ddtheta2[sl],
        'dt': dt,
    }


def subsample(data, factor=10):
    """Subsample data to reduce size. Original is 1kHz, factor=10 gives 100Hz."""
    sl = slice(None, None, factor)
    return {k: (v[sl] if isinstance(v, np.ndarray) else v)
            for k, v in data.items()}


def prepare_dataset(train_indices=(1, 2, 3), val_index=4,
                    subsample_factor=10, smooth_window=5):
    """Prepare training and validation datasets.

    Args:
        train_indices: which free-swing experiments for training
        val_index: which for validation
        subsample_factor: downsample from 1kHz
        smooth_window: smoothing for acceleration computation

    Returns:
        train_data, val_data: dicts with numpy arrays
    """
    def process(index):
        raw = load_freeswing(index)
        acc = compute_accelerations(raw, smooth_window=smooth_window)
        return subsample(acc, factor=subsample_factor)

    # Concatenate training data from multiple experiments
    train_parts = [process(i) for i in train_indices]
    train_data = {}
    for key in train_parts[0]:
        if isinstance(train_parts[0][key], np.ndarray):
            train_data[key] = np.concatenate([p[key] for p in train_parts])
        else:
            train_data[key] = train_parts[0][key]

    val_data = process(val_index)

    return train_data, val_data


if __name__ == '__main__':
    train, val = prepare_dataset()
    print(f"Train: {len(train['t'])} samples from experiments 1,2,3")
    print(f"Val:   {len(val['t'])} samples from experiment 4")
    print(f"dt: {train['dt']}s (original), subsampled to {train['dt']*10}s")
    print(f"theta1 range: [{train['theta1'].min():.2f}, {train['theta1'].max():.2f}] rad")
    print(f"dtheta1 range: [{train['dtheta1'].min():.2f}, {train['dtheta1'].max():.2f}] rad/s")
    print(f"ddtheta1 range: [{train['ddtheta1'].min():.2f}, {train['ddtheta1'].max():.2f}] rad/s^2")
