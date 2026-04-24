import numpy as np

import config
from camera import simulate_camera
from degradation import window_end
from ekf import ErrorStateEKF
from gps import simulate_gps
from imu import simulate_imu
from lidar import simulate_lidar
from metrics import component_rmse, rmse
from trajectory import generate_synthetic_trajectory
from ukf import PositionVelocityUKF
from visualization import plot_trajectories


def run(plot=False):
    np.random.seed(config.seed)

    true = generate_synthetic_trajectory(config.n_steps, config.dt)

    gps_dropout_end = window_end(config.gps_dropout_start, config.gps_dropout_duration)
    cam_blackout_end = window_end(config.camera_blackout_start, config.camera_blackout_duration)

    ekf = ErrorStateEKF(
        p0=true["p"][0],
        v0=true["v"][0],
        q0=true["q"][0],
        P0=config.P0,
        q_accel=config.q_accel,
    )
    ukf = PositionVelocityUKF(
        p0=true["p"][0],
        v0=true["v"][0],
        q0=true["q"][0],
        P0=config.P0,
        q_accel=config.q_accel,
    )

    ba = np.zeros(3)
    bw = np.zeros(3)

    ekf_p_hist = np.zeros((config.n_steps, 3))
    ukf_p_hist = np.zeros((config.n_steps, 3))

    for k in range(config.n_steps):
        a_meas, w_meas, ba, bw = simulate_imu(
            true,
            k,
            config.sigma_a,
            config.sigma_w,
            config.sigma_ba,
            config.sigma_bw,
            ba,
            bw,
            config.dt,
        )

        ekf.predict(a_meas, w_meas, config.dt)
        ukf.predict(a_meas, w_meas, config.dt)

        z_gps = simulate_gps(true, k, config.R_gps, config.gps_dropout_start, gps_dropout_end)
        z_lidar, _, lidar_cov = simulate_lidar(true, k, config.R_lidar, config.lidar_degradation_start)
        z_cam, _ = simulate_camera(true, k, config.R_camera, config.camera_blackout_start, cam_blackout_end)

        if z_gps is not None:
            ekf.update_position(z_gps, config.R_gps)
            ukf.update_position(z_gps, config.R_gps)

        ekf.update_position(z_lidar, lidar_cov)
        ukf.update_position(z_lidar, lidar_cov)

        if z_cam is not None:
            ekf.update_position(z_cam, config.R_camera)
            ukf.update_position(z_cam, config.R_camera)

        ekf_p_hist[k] = ekf.state[0]
        ukf_p_hist[k] = ukf.state[0]

    true_p = true["p"]
    print("----- Results -----")
    print(f"EKF RMSE (3D): {rmse(ekf_p_hist, true_p):.3f} m")
    print(f"UKF RMSE (3D): {rmse(ukf_p_hist, true_p):.3f} m")

    ekf_axis = component_rmse(ekf_p_hist, true_p)
    ukf_axis = component_rmse(ukf_p_hist, true_p)
    print(f"EKF axis RMSE [x y z]: {ekf_axis}")
    print(f"UKF axis RMSE [x y z]: {ukf_axis}")

    if plot:
        plot_trajectories(true_p, ekf_p_hist, ukf_p_hist)


if __name__ == "__main__":
    run(plot=False)
