import numpy as np

def simulate_imu(true_state, k, sigma_a, sigma_w, sigma_ba, sigma_bw, ba_prev, bw_prev):
    p, v, q = true_state['p'][k], true_state['v'][k], true_state['q'][k]
    
    # True acceleration and angular rate in body frame
    R = quat_to_rot(q)
    a_true_body = R.T @ np.array([0.5, 0.0, -9.81])  # rough gravity compensation simulation
    
    w_true = np.array([0.0, 0.0, 0.05])  # base yaw rate + noise from trajectory

    # Add white noise + bias (random walk)
    ba = ba_prev + np.random.normal(0, sigma_ba * np.sqrt(0.01), 3)
    bw = bw_prev + np.random.normal(0, sigma_bw * np.sqrt(0.01), 3)
    
    a_meas = a_true_body + ba + np.random.normal(0, sigma_a, 3)
    w_meas = w_true + bw + np.random.normal(0, sigma_w, 3)
    
    return a_meas, w_meas, ba, bw