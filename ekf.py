import numpy as np

from quaternion import quat_mul, quat_to_rot, small_angle_quat, normalize


class ErrorStateEKF:
    """Simple inertial propagation + linear position updates over [p, v]."""

    def __init__(self, p0, v0, q0, P0, q_accel):
        self.p = p0.astype(float).copy()
        self.v = v0.astype(float).copy()
        self.q = normalize(q0.astype(float).copy())

        self.P = P0.astype(float).copy()  # 6x6 covariance over [p, v]
        self.q_accel = float(q_accel)

    def predict(self, a_meas_body, w_meas_body, dt):
        dq = small_angle_quat(w_meas_body * dt)
        self.q = normalize(quat_mul(self.q, dq))

        g_world = np.array([0.0, 0.0, -9.81])
        a_world = quat_to_rot(self.q) @ a_meas_body + g_world

        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt

        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt

        G = np.zeros((6, 3))
        G[0:3, :] = 0.5 * np.eye(3) * dt * dt
        G[3:6, :] = np.eye(3) * dt

        Q = (self.q_accel**2) * np.eye(3)
        self.P = F @ self.P @ F.T + G @ Q @ G.T

    def update_position(self, z, R):
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)

        x = np.hstack([self.p, self.v])
        y = z - self.p

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        x_new = x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        self.p = x_new[:3]
        self.v = x_new[3:]

    @property
    def state(self):
        return self.p.copy(), self.v.copy(), self.q.copy()
