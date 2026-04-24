import numpy as np


def simulate_lidar(true, k, R_lidar, degradation_start):
    p = true["p"][k]
    q = true["q"][k]
    cov = R_lidar * (3.0 if k > degradation_start else 1.0)
    return p + np.random.multivariate_normal(np.zeros(3), cov), q.copy(), cov
