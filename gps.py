# sensors/gps.py
import numpy as np

def simulate_gps(true, k, R_gps, dropout_start, dropout_end):
    if dropout_start <= k <= dropout_end:
        return None  # dropout
    return true['p'][k] + np.random.multivariate_normal(np.zeros(3), R_gps)