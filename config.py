import numpy as np

# Simulation parameters
dt = 0.01                    # 100 Hz IMU
n_steps = 10000              # 100 seconds total
t = np.arange(n_steps) * dt

# Noise parameters
sigma_a = 0.05               # IMU accel white noise (m/s²)
sigma_w = 0.01               # IMU gyro white noise (rad/s)
sigma_ba = 0.001             # Accel bias random walk
sigma_bw = 0.0005            # Gyro bias random walk

# Measurement noises
R_gps = np.eye(3) * 2.0**2
R_lidar = np.eye(3) * 0.3**2
R_camera = np.eye(3) * 0.4**2

# Degradation parameters
gps_dropout_start = 4000     # ~40s
gps_dropout_duration = 6000  # 60 seconds dropout

lidar_degradation_start = 2000
camera_blackout_start = 6000
camera_blackout_duration = 2000