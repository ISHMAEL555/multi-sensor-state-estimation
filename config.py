import numpy as np

# Simulation parameters
dt = 0.01
n_steps = 10000
seed = 7

# IMU noise and bias random walk
sigma_a = 0.05
sigma_w = 0.01
sigma_ba = 0.001
sigma_bw = 0.0005

# Position measurement noise covariances
R_gps = np.eye(3) * 2.0**2
R_lidar = np.eye(3) * 0.3**2
R_camera = np.eye(3) * 0.4**2

# Sensor degradation windows (index domain)
gps_dropout_start = 4000
gps_dropout_duration = 6000

lidar_degradation_start = 2000

camera_blackout_start = 6000
camera_blackout_duration = 2000

# Filters
P0 = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
q_accel = 0.2  # process acceleration noise (m/s^2)
