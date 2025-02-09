import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import roc_curve
from .common import load_pickle, dump_pickle

plt.style.use("seaborn-darkgrid")


def plot_importance(filepath, y, data, xerr=None, figsize=(10, 15)):
    # Plot Importance DataFrame.
    plt.figure(figsize=figsize)
    plt.title("Feature importance")
    plt.barh(y, width=data, xerr=xerr, label="importance")
    plt.tight_layout()
    plt.legend(loc="lower right")
    plt.savefig(filepath)
    plt.close("all")


def plot_roc_curve(y_true, y_score, filepath, figsize=(7, 6)):
    """Plot the roc curve.
    Parameters
    ----------
    y_true : numpy.ndarray
        The target vector.
    y_score : numpy.ndarray
        The score vector.
    figsize : tuple
        Figure dimension ``(width, height)`` in inches.
    Returns
    -------
    None
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(filepath)
    plt.close("all")
