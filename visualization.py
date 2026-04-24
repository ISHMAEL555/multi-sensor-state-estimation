"""Simple optional plotting helpers."""

import importlib
import importlib.util

import numpy as np


def plot_trajectories(true_p: np.ndarray, ekf_p: np.ndarray, ukf_p: np.ndarray):
    if importlib.util.find_spec("matplotlib") is None:
        print("Plot skipped: matplotlib is not installed.")
        return

    plt = importlib.import_module("matplotlib.pyplot")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(true_p[:, 0], true_p[:, 1], true_p[:, 2], label="True", linewidth=2)
    ax.plot(ekf_p[:, 0], ekf_p[:, 1], ekf_p[:, 2], label="EKF")
    ax.plot(ukf_p[:, 0], ukf_p[:, 1], ukf_p[:, 2], label="UKF")
    ax.set_title("Trajectory comparison")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    plt.tight_layout()
    plt.show()
