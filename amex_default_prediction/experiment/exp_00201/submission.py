import os
import pickle
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def main():
    test = pd.read_parquet("./output/test_fe.parquet")

    with open("./output/train_features.pkl", "rb") as f:
        features = pickle.load(f)
    features = [col for col in features if col not in ["customer_ID", "target"]]

    seed = 42
    num_seed = 5
    num_fold = 5
    predictions = np.zeros(len(test))
    for i in tqdm(range(num_seed)):
        sub_seed = seed + i
        print(f"\n\n=== Prediction with seed={sub_seed} ===")
        for fold in tqdm(range(num_fold), leave=False):
            with open(f"./output/lgb_seed={seed}_fold={fold}.pickle", "rb") as f:
                model = pickle.load(f)

            batch_size = 100_000

            for i in tqdm(range(0, test.shape[0], batch_size), leave=False):
                start = i
                end = min(i + batch_size, test.shape[0])
                predictions[start:end] += model.predict(
                    test.iloc[start:end][features]
                ) / (num_fold * num_seed)

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
