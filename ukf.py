import numpy as np

from quaternion import quat_mul, quat_to_rot, small_angle_quat, normalize


class PositionVelocityUKF:
    """Unscented Kalman Filter over x=[p,v] with quaternion attitude propagation."""

    def __init__(self, p0, v0, q0, P0, q_accel, alpha=0.5, beta=2.0, kappa=0.0):
        self.x = np.hstack([p0, v0]).astype(float)
        self.q = normalize(q0.astype(float).copy())
        self.P = P0.astype(float).copy()
        self.q_accel = float(q_accel)

        self.n = self.x.size
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lmbda = alpha * alpha * (self.n + kappa) - self.n

        self.Wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lmbda)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lmbda / (self.n + self.lmbda)
        self.Wc[0] = self.Wm[0] + (1.0 - alpha * alpha + beta)

    def _sigma_points(self):
        jitter = 1e-9 * np.eye(self.n)
        S = np.linalg.cholesky((self.n + self.lmbda) * (self.P + jitter))
        X = np.zeros((2 * self.n + 1, self.n))
        X[0] = self.x
        for i in range(self.n):
            X[i + 1] = self.x + S[:, i]
            X[self.n + i + 1] = self.x - S[:, i]
        return X

    def predict(self, a_meas_body, w_meas_body, dt):
        dq = small_angle_quat(w_meas_body * dt)
        self.q = normalize(quat_mul(self.q, dq))

        g_world = np.array([0.0, 0.0, -9.81])
        a_world = quat_to_rot(self.q) @ a_meas_body + g_world

        X = self._sigma_points()
        for i in range(X.shape[0]):
            p = X[i, :3]
            v = X[i, 3:]
            p = p + v * dt + 0.5 * a_world * dt * dt
            v = v + a_world * dt
            X[i, :3] = p
            X[i, 3:] = v

        x_pred = np.sum(self.Wm[:, None] * X, axis=0)
        P_pred = np.zeros((self.n, self.n))
        for i in range(X.shape[0]):
            dx = (X[i] - x_pred)[:, None]
            P_pred += self.Wc[i] * (dx @ dx.T)

        G = np.zeros((6, 3))
        G[0:3, :] = 0.5 * np.eye(3) * dt * dt
        G[3:6, :] = np.eye(3) * dt
        Q = (self.q_accel**2) * np.eye(3)

        self.x = x_pred
        self.P = P_pred + G @ Q @ G.T

    def update_position(self, z, R):
        X = self._sigma_points()
        Z = X[:, :3]

        z_pred = np.sum(self.Wm[:, None] * Z, axis=0)
        S = np.zeros((3, 3))
        Cxz = np.zeros((self.n, 3))

        for i in range(Z.shape[0]):
            dz = (Z[i] - z_pred)[:, None]
            dx = (X[i] - self.x)[:, None]
            S += self.Wc[i] * (dz @ dz.T)
            Cxz += self.Wc[i] * (dx @ dz.T)

        S += R
        K = Cxz @ np.linalg.inv(S)

        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ S @ K.T

    @property
    def state(self):
        return self.x[:3].copy(), self.x[3:].copy(), self.q.copy()
