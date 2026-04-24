import numpy as np


def simulate_camera(true, k, R_camera, blackout_start, blackout_end):
    if blackout_start <= k <= blackout_end:
        return None, None
    p = true["p"][k]
    q = true["q"][k]
    return p + np.random.multivariate_normal(np.zeros(3), R_camera), q.copy()
