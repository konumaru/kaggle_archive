import os
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm


def main():
    test = pd.read_parquet("./output/test_fe.parquet")
    # features = [col for col in test.columns if col not in ["customer_ID", "target"]]

    with open("./output/train_features.pkl", "rb") as f:
        features = pickle.load(f)
    features = [col for col in features if col not in ["customer_ID", "target"]]

    seed = 42
    num_seed = 5
    num_fold = 5
    predictions = np.zeros(len(test))
    for i in range(num_seed):
        sub_seed = seed + i
        print(f"\n\n=== Prediction with seed={sub_seed} ===")
        for fold in tqdm(range(num_fold)):
            bst = xgb.Booster()
            bst.load_model(f"./output/xgb_seed={sub_seed}_fold={fold}.json")

            batch_size = 100_000

            for i in range(0, test.shape[0], batch_size):
                start = i
                end = min(i + batch_size, test.shape[0])
                dtest = xgb.DMatrix(data=test.iloc[start:end][features])
                predictions[start:end] += bst.predict(dtest) / (num_fold * num_seed)

    submission = pd.DataFrame(
        {
            "customer_ID": test["customer_ID"],
            "prediction": predictions,
        }
    )
    submission.to_csv("./output/submission.csv", index=False)


if __name__ == "__main__":
    main()

    exp_name = os.path.basename(os.getcwd())
    subprocess.run(
        f'kaggle competitions submit -c amex-default-prediction -f ./output/submission.csv -m "{exp_name}"',
        shell=True,
    )
