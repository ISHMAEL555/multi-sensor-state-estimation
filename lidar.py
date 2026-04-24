# sensors/lidar.py
import numpy as np
from utils.quaternion import quat_to_rot

def simulate_lidar(true, k, R_lidar, degradation_start):
    p = true['p'][k]
    q = true['q'][k]
    if k > degradation_start:
        R_lidar = R_lidar * 3.0  # degraded
    return p + np.random.multivariate_normal(np.zeros(3), R_lidar), q.copy()