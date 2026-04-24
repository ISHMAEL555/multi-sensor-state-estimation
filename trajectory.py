import numpy as np

from quaternion import quat_mul, normalize, quat_to_rot, small_angle_quat


def generate_synthetic_trajectory(n_steps: int, dt: float):
    """Urban-like trajectory with smooth turns and mild elevation changes."""
    t = np.arange(n_steps) * dt
    p = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    q = np.zeros((n_steps, 4))
    a_world = np.zeros((n_steps, 3))
    w_body = np.zeros((n_steps, 3))

    q[0] = [1.0, 0.0, 0.0, 0.0]

    for k in range(1, n_steps):
        a_world[k] = np.array(
            [
                0.5 + 0.15 * np.sin(0.35 * t[k]),
                0.25 * np.sin(0.20 * t[k]),
                0.06 * np.sin(0.12 * t[k]),
            ]
        )

        w_body[k] = np.array(
            [0.01 * np.sin(0.13 * t[k]), 0.005 * np.sin(0.22 * t[k]), 0.12 * np.sin(0.40 * t[k])]
        )

        p[k] = p[k - 1] + v[k - 1] * dt + 0.5 * a_world[k] * dt * dt
        v[k] = v[k - 1] + a_world[k] * dt

        dq = small_angle_quat(w_body[k] * dt)
        q[k] = normalize(quat_mul(q[k - 1], dq))

    return {"t": t, "p": p, "v": v, "q": q, "a_world": a_world, "w_body": w_body}
