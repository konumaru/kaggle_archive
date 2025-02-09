import gc
import pickle

import numpy as np
import pandas as pd
from cuml.preprocessing import TargetEncoder
from tqdm.auto import tqdm

"""IDEA
- get_differenceのdiffを複数作る
- first / last の diff div 特徴量を追加する
"""


def get_difference(data, num_features):
    # TODO: fillna(-1)されているので、平均をとったときやや意図しない結果になっている可能性可能性がある
    # -1.replace(np.nan)して、diff, rolling したあとにfillna(-1)を再度行うほうがよさそう
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


def get_rolling_mean(data, num_features):
    # TODO: Add rolling feature max, min, nunique
    df1 = []
    df2 = []
    df3 = []
    customer_ids = []

    for customer_id, df in tqdm(data.groupby(["customer_ID"])):
        diff_df1 = (
            df.loc[:-7, num_features].mean().values.astype(np.float32).reshape(1, -1)
        )
        df1.append(diff_df1)

        diff_df2 = (
            df.loc[:-14, num_features].mean().values.astype(np.float32).reshape(1, -1)
        )
        df2.append(diff_df2)

        diff_df3 = (
            df.loc[:-28, num_features].mean().values.astype(np.float32).reshape(1, -1)
        )
        df3.append(diff_df3)
        customer_ids.append(customer_id)

    df1 = np.concatenate(df1, axis=0)
    df1 = pd.DataFrame(
        df1,
        columns=[col + "_rolling7" for col in df[num_features].columns],
    )
    df2 = np.concatenate(df2, axis=0)
    df2 = pd.DataFrame(
        df2,
        columns=[col + "_rolling14" for col in df[num_features].columns],
    )
    df3 = np.concatenate(df3, axis=0)
    df3 = pd.DataFrame(
        df3,
        columns=[col + "_rolling28" for col in df[num_features].columns],
    )
    df = pd.concat([df1, df2, df3], axis=1)
    df["customer_ID"] = customer_ids
    return df


def number_aggregate(data, num_features):
    # TODO: add sum function.
    num_agg = data.groupby("customer_ID")[num_features].agg(
        ["mean", "std", "min", "max", "last", "first"]
    )
    num_agg.columns = ["_".join(x) for x in num_agg.columns]
    num_agg.reset_index(inplace=True)

    extend_num_agg = {}
    for col in tqdm(num_features):
        extend_num_agg[f"{col}_last_mean_diff"] = (
            num_agg[f"{col}_last"] - num_agg[f"{col}_mean"]
        ).to_numpy()

        # NOTE: Add round last float features to 2 decimal place.
        # - High corr _last and _last_round2 so that remove from features.
        # extend_num_agg[f"{col}_last_round2"] = (
        #     num_agg[f"{col}_last"].round(2).to_numpy()
        # )

        # NOTE: sub div features
        calc_cols = [
            ("last", "first"),
            ("last", "max"),
            ("last", "mean"),
            # NOTE: Not improved.
            # ("first", "min"),
            # ("last", "min"),
            # ("max", "min"),
        ]
        for col_first, col_second in calc_cols:
            extend_num_agg[f"{col}_sub_last_first"] = (
                num_agg[f"{col}_{col_first}"] - num_agg[f"{col}_{col_second}"]
            )
            extend_num_agg[f"{col}_div_last_first"] = (
                num_agg[f"{col}_{col_first}"] / num_agg[f"{col}_{col_second}"]
            ).replace([np.inf, -np.inf], np.nan)

    num_agg = pd.concat([num_agg, pd.DataFrame(extend_num_agg)], axis=1)

    num_cols = num_agg.select_dtypes(include="number").columns
    for col in num_cols:
        num_agg[col] = num_agg[col].astype(np.float32)
    return num_agg


def category_aggregate(data, cat_features):
    cat_agg = data.groupby("customer_ID")[cat_features].agg(
        ["count", "last", "first", "nunique"]
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


def target_encode(data: pd.DataFrame, is_test: bool = False):
    encoder_filepath = "./output/target_encoder.pkl"
    te_data = pd.DataFrame()

    def rounded_category(
        series: pd.Series, decimals: int, min_val: float = None, max_val: float = None
    ):
        category = series.round(decimals).clip(min_val, max_val)
        return category

    if is_test:
        with open(encoder_filepath, "rb") as f:
            encoders = pickle.load(f)

        for f_name, encoder in encoders.items():
            if f_name == "P_2":
                category = data["P_2_last"].round(1).clip(-0.2, None).astype(str).copy()
                te_data[f"{f_name}_te"] = encoder.transform(category)
            elif f_name == "B_5":
                category = rounded_category(data[f"{f_name}_last"], 0, None, 5.0)
                te_data[f"{f_name}_te"] = encoder.transform(category)
            elif f_name == "S_3":
                category = rounded_category(data[f"{f_name}_last"], 1, -0.1, 1.2)
                te_data[f"{f_name}_te"] = encoder.transform(category)
            elif f_name == "D_39":
                category = rounded_category(data[f"{f_name}_last"], 0, None, 32.0)
                te_data[f"{f_name}_te"] = encoder.transform(category)
            elif f_name == "D_43":
                category = rounded_category(data[f"{f_name}_last"], 2, None, 1.00)
                te_data[f"{f_name}_te"] = encoder.transform(category)
            elif f_name == "R_1":
                category = rounded_category(data[f"{f_name}_last"], 1, None, 2.0)
                te_data[f"{f_name}_te"] = encoder.transform(category)
    else:
        encoders = {}
        feature_names = ["P_2", "B_5", "S_3", "D_39", "D_43", "R_1"]
        for f_name in feature_names:
            if f_name == "P_2":
                category = data["P_2_last"].round(1).clip(-0.2, None).astype(str).copy()
                encoder = TargetEncoder(n_folds=5, smooth=20, output_type="numpy")
                te_data[f"{f_name}_te"] = encoder.fit_transform(category, data.target)
                encoders[f_name] = encoder
            elif f_name == "B_5":
                category = rounded_category(data[f"{f_name}_last"], 0, None, 5.0)
                encoder = TargetEncoder(n_folds=5, smooth=10, output_type="numpy")
                te_data[f"{f_name}_te"] = encoder.fit_transform(category, data.target)
                encoders[f_name] = encoder
            elif f_name == "S_3":
                category = rounded_category(data[f"{f_name}_last"], 1, -0.1, 1.2)
                encoder = TargetEncoder(n_folds=5, smooth=10, output_type="numpy")
                te_data[f"{f_name}_te"] = encoder.fit_transform(category, data.target)
                encoders[f_name] = encoder
            elif f_name == "D_39":
                category = rounded_category(data[f"{f_name}_last"], 0, None, 32.0)
                encoder = TargetEncoder(n_folds=5, smooth=10, output_type="numpy")
                te_data[f"{f_name}_te"] = encoder.fit_transform(category, data.target)
                encoders[f_name] = encoder
            elif f_name == "D_43":
                category = rounded_category(data[f"{f_name}_last"], 2, None, 1.00)
                encoder = TargetEncoder(n_folds=5, smooth=10, output_type="numpy")
                te_data[f"{f_name}_te"] = encoder.fit_transform(category, data.target)
                encoders[f_name] = encoder
            elif f_name == "R_1":
                category = rounded_category(data[f"{f_name}_last"], 1, None, 2.0)
                encoder = TargetEncoder(n_folds=5, smooth=10, output_type="numpy")
                te_data[f"{f_name}_te"] = encoder.fit_transform(category, data.target)
                encoders[f_name] = encoder

        with open(encoder_filepath, "wb") as f:
            pickle.dump(encoders, f)

    data = pd.concat([data, te_data], axis=1)
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
    train_rolling_mean = get_rolling_mean(train, num_features)
    train_num_agg = number_aggregate(train, num_features)
    train_cat_agg = category_aggregate(train, cat_features)
    train = (
        train_num_agg.merge(train_cat_agg, how="inner", on="customer_ID")
        .merge(train_rolling_mean, how="inner", on="customer_ID")
        .merge(train_diff, how="inner", on="customer_ID")
        .merge(train_labels, how="inner", on="customer_ID")
    )
    train = label_encode(train, cat_features)
    # train = target_encode(train)
    print(train.head())

    train.to_parquet("./output/train_fe.parquet")
    del train, train_num_agg, train_cat_agg, train_diff
    gc.collect()

    print("Starting test feature engineer...")
    test = pd.read_parquet("../../data/test.parquet")

    test_diff = get_difference(test, num_features)
    test_rolling_mean = get_rolling_mean(test, num_features)
    test_num_agg = number_aggregate(test, num_features)
    test_cat_agg = category_aggregate(test, cat_features)
    test = (
        test_num_agg.merge(test_cat_agg, how="inner", on="customer_ID")
        .merge(test_rolling_mean, how="inner", on="customer_ID")
        .merge(test_diff, how="inner", on="customer_ID")
    )
    test = label_encode(test, cat_features)
    # test = target_encode(test, is_test=True)
    print(test.head())

    test.to_parquet("./output/test_fe.parquet")
    del test, test_num_agg, test_cat_agg, test_diff
    gc.collect()


if __name__ == "__main__":
    main()
