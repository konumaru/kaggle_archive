import sys

sys.path.append("..")

import numpy as np
import pandas as pd

from utils.trainer.tabnet import TabNetClassifierTrainer


def main():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X, Y = make_classification(
        random_state=12,
        n_samples=5000,
        n_features=100,
        n_redundant=3,
        n_informative=20,
        n_clusters_per_class=1,
        n_classes=2,
    )
    X = pd.DataFrame(X)

    X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, Y, test_size=0.2, stratify=Y
    )

    params = {"n_d": 16, "n_a": 16, "n_steps": 3, "seed": 42, "device_name": "cuda"}
    train_params = {
        "eval_metric": ["auc"],
        "max_epochs": 20,
        "patience": 5,
        "batch_size": 1024,
        "num_workers": 4,
        "drop_last": True,
    }

    print(X_train.head())
    trainer = TabNetClassifierTrainer()
    trainer.fit(
        params,
        train_params,
        X_train,
        pd.DataFrame(y_train),
        X_valid,
        pd.DataFrame(y_valid),
    )
    pred = trainer.predict(X_test)

    auc = roc_auc_score(y_test, pred)
    print("AUC is", auc)


if __name__ == "__main__":
    main()
