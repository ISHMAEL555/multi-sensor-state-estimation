import numpy as np

from quaternion import quat_to_rot


def simulate_imu(true_state, k, sigma_a, sigma_w, sigma_ba, sigma_bw, ba_prev, bw_prev, dt):
    """Simulate body-frame accelerometer and gyro readings with drifting biases."""
    q = true_state["q"][k]
    a_world = true_state["a_world"][k]
    w_true = true_state["w_body"][k]

    g_world = np.array([0.0, 0.0, -9.81])
    R_wb = quat_to_rot(q)

    # accelerometer measures specific force in body frame: R^T(a - g)
    a_true_body = R_wb.T @ (a_world - g_world)

    ba = ba_prev + np.random.normal(0.0, sigma_ba * np.sqrt(dt), 3)
    bw = bw_prev + np.random.normal(0.0, sigma_bw * np.sqrt(dt), 3)

    a_meas = a_true_body + ba + np.random.normal(0.0, sigma_a, 3)
    w_meas = w_true + bw + np.random.normal(0.0, sigma_w, 3)
    return a_meas, w_meas, ba, bw
