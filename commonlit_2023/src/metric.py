import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    squared_error = (y_pred - y_true) ** 2
    mean_squared_error = squared_error.mean()
    rmse = np.sqrt(mean_squared_error)
    return rmse


def mcrmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    squared_error = (y_pred - y_true) ** 2
    mean_squared_error = squared_error.mean(axis=0)
    mcrmse = np.sqrt(mean_squared_error).mean()
    return mcrmse
