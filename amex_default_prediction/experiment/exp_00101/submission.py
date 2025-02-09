import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm


def main():
    test = pd.read_parquet("./output/test_fe.parquet")
    features = [col for col in test.columns if col not in ["customer_ID", "target"]]

    num_fold = 5
    predictions = np.zeros(len(test))
    for fold in tqdm(range(num_fold)):
        bst = xgb.Booster()
        bst.load_model(f"./output/xgb_{fold}.json")

        dtest = xgb.DMatrix(data=test[features])
        predictions += bst.predict(dtest) / num_fold

    submission = pd.DataFrame(
        {
            "customer_ID": test["customer_ID"],
            "prediction": predictions,
        }
    )
    submission.to_csv("./output/submission.csv", index=False)


if __name__ == "__main__":
    main()
