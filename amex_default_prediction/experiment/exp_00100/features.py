import gc
import pickle
from re import I
from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_difference(data, num_features):
    df1 = []
    customer_ids = []

    for customer_id, df in tqdm(data.groupby(["customer_ID"])):
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        df1.append(diff_df1)
        customer_ids.append(customer_id)

    df1 = np.concatenate(df1, axis=0)
    df1 = pd.DataFrame(df1, columns=[col + "_diff" for col in df[num_features].columns])
    df1["customer_ID"] = customer_ids
    return df1


def number_aggregate(data, num_features):
    num_agg = data.groupby("customer_ID")[num_features].agg(
        ["mean", "std", "min", "max", "last"]
    )
    num_agg.columns = ["_".join(x) for x in num_agg.columns]
    num_agg.reset_index(inplace=True)

    for col in tqdm(num_features):
        try:
            num_agg[f"{col}_last_mean_diff"] = (
                num_agg[f"{col}_last"] - num_agg[f"{col}_mean"]
            )
        except:
            pass

        # NOTE: Add round last float features to 2 decimal place.
        num_agg[col + "_last_round2"] = num_agg[col + "_last"].round(2)

    num_cols = num_agg.select_dtypes(include="number").columns
    for col in num_cols:
        num_agg[col] = num_agg[col].astype(np.float32)
    return num_agg


def category_aggregate(data, cat_features):
    cat_agg = data.groupby("customer_ID")[cat_features].agg(
        ["count", "last", "nunique"]
    )
    cat_agg.columns = ["_".join(x) for x in cat_agg.columns]
    cat_agg.reset_index(inplace=True)

    cols = list(cat_agg.dtypes[cat_agg.dtypes == "int64"].index)
    for col in tqdm(cols):
        cat_agg[col] = cat_agg[col].astype(np.int32)
    return cat_agg


def label_encode(data, cat_features):
    for cat_col in cat_features:
        with open(f"./output/{cat_col}.labelencoder.pkl", "rb") as f:
            encoder = pickle.load(f)
            data[f"{cat_col}_last"] = encoder.transform(data[f"{cat_col}_last"])
    return data


def main():
    train = pd.read_parquet("../../data/train.parquet")

    features = train.drop(["customer_ID", "S_2"], axis=1).columns.to_list()
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
    num_features = [col for col in features if col not in cat_features]

    print("Starting train feature engineer...")
    train_labels = pd.read_csv("../../data/train_labels.csv")
    train_diff = get_difference(train, num_features)
    train_num_agg = number_aggregate(train, num_features)
    train_cat_agg = category_aggregate(train, cat_features)
    train = (
        train_num_agg.merge(train_cat_agg, how="inner", on="customer_ID")
        .merge(train_diff, how="inner", on="customer_ID")
        .merge(train_labels, how="inner", on="customer_ID")
    )
    train = label_encode(train, cat_features)
    train.to_parquet("./output/train_fe.parquet")
    print(train.shape)
    del train, train_num_agg, train_cat_agg, train_diff
    gc.collect()

    print("Starting test feature engineer...")
    test = pd.read_parquet("../../data/test.parquet")

    test_diff = get_difference(test, num_features)
    test_num_agg = number_aggregate(test, num_features)
    test_cat_agg = category_aggregate(test, cat_features)
    test = test_num_agg.merge(test_cat_agg, how="inner", on="customer_ID").merge(
        test_diff, how="inner", on="customer_ID"
    )
    test = label_encode(test, cat_features)
    test.to_parquet("./output/test_fe.parquet")
    del test, test_num_agg, test_cat_agg, test_diff
    gc.collect()


if __name__ == "__main__":
    main()
