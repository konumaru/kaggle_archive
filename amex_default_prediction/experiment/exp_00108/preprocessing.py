import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    train = pd.read_parquet("../../data/train.parquet")
    test = pd.read_parquet("../../data/test.parquet")

    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]

    for cat_col in cat_features:
        encoder = LabelEncoder()
        encoder.fit(pd.concat([train[cat_col], test[cat_col]]))

        with open(f"./output/{cat_col}.labelencoder.pkl", "wb") as f:
            pickle.dump(encoder, f)


if __name__ == "__main__":
    main()
