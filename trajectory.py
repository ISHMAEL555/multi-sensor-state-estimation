import numpy as np
from utils.quaternion import quat_mul, normalize, small_angle_quat

def generate_synthetic_trajectory(n_steps, dt):
    t = np.arange(n_steps) * dt
    p = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    q = np.zeros((n_steps, 4))
    q[0] = [1, 0, 0, 0]

    # Urban-like trajectory: straight + turns + elevation change
    for k in range(1, n_steps):
        # Acceleration profile (forward + some lateral for turns)
        a = np.array([0.5 + 0.1*np.sin(t[k]), 0.2*np.sin(0.5*t[k]), 0.05*np.sin(0.3*t[k])])
        
        # Angular velocity (yaw turns)
        w = np.array([0, 0, 0.1 * np.sin(0.4 * t[k])])

        R = quat_to_rot(q[k-1]) if k > 1 else np.eye(3)
        
        p[k] = p[k-1] + v[k-1] * dt
        v[k] = v[k-1] + R @ a * dt
        
        dq = small_angle_quat(w * dt)
        q[k] = normalize(quat_mul(q[k-1], dq))

    return {'t': t, 'p': p, 'v': v, 'q': q}