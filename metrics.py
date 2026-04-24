import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def component_rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((a - b) ** 2, axis=0))
