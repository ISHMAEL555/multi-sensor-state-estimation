# Multi-Sensor State Estimation

Lightweight simulation of 3D motion with IMU/GPS/LiDAR/Camera measurements and two estimators:

- Error-state-inspired EKF over position/velocity with quaternion attitude propagation.
- UKF over position/velocity with quaternion attitude propagation.

## Run

```bash
python main.py
```

Optional plotting:

```bash
python -c "import main; main.run(plot=True)"
```

## Files

- `trajectory.py`: truth trajectory generation.
- `imu.py`, `gps.py`, `lidar.py`, `camera.py`: measurement simulation.
- `ekf.py`, `ukf.py`: filtering.
- `metrics.py`: RMSE utilities.
- `config.py`: all tunables.

## Notes

- Plotting helper lives in `visualization.py`.
- Plotting requires `matplotlib`; estimation itself only requires `numpy`.
