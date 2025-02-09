import numpy as np

from metric import mcrmse, rmse


def tests_rmse() -> None:
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    score = rmse(y_true, y_pred)
    assert score == 0


def tests_mc_rmse() -> None:
    y_true = np.array([[1, 2, 3], [1, 2, 3]])
    y_pred = np.array([[1, 2, 3], [1, 2, 3]])

    score = mcrmse(y_true, y_pred)
    assert score == 0
