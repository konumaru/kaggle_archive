import pickle

import pandas as pd
import xgboost as xgb


def main():
    seed = 42
    num_seed = 5
    num_fold = 5

    f_scores = []
    # for i in range(num_seed):
    # sub_seed = seed + i
    sub_seed = 46
    for fold in range(num_fold):
        bst = xgb.Booster()
        bst.load_model(f"./output/xgb_seed={sub_seed}_fold={fold}.json")

        f_score = bst.get_score()
        f_scores.append(f_score)

    df = pd.DataFrame(f_scores).T
    df["avg"] = df.mean(axis=1)

    df.sort_values(by="avg", ascending=False, inplace=True)
    print(df.head())

    top_features = df.index[:500].tolist()

    with open("./output/top_features.pkl", "wb") as f:
        pickle.dump(top_features, f)


if __name__ == "__main__":
    main()
