import collections
import glob
import os
import pathlib
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils import feature


def Xy_splitter(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target_days = 7
    latest_date = data["t_dat"].max()

    train_end_date = latest_date - timedelta(days=target_days)
    X = data[data["t_dat"] <= train_end_date]
    y = data[data["t_dat"] > train_end_date]

    # NOTE: Exclude data without non buyer.
    target_customer_ids = y["customer_id"].unique()
    X = X[X["customer_id"].isin(target_customer_ids)]
    return X, y


def create_target(data: pd.DataFrame) -> pd.Series:
    # NOTE: map@12 において、購買したarticle_idのリストはすべて必要
    def normalize_articles_couter(articles: List):
        articles_cnt = collections.Counter(articles)
        articles_cnt_max = max(articles_cnt.values())
        articles_cnt = {
            k: v if v == 0 else np.log(v + 10) / np.log(articles_cnt_max + 10)
            for k, v in articles_cnt.items()
        }
        # NOTE: one-hot values is below
        # articles_cnt = {k: 1 for k, _ in articles_cnt.items()}
        return articles_cnt

    out = data.groupby("customer_id")["article_id"].agg(list)
    out = out.apply(lambda x: normalize_articles_couter(x))
    out = out.reset_index()
    return out


@feature(use_cache=True)
def transaction_seq(data: pd.DataFrame, y: pd.DataFrame, save_dir: str) -> np.ndarray:
    out = (
        data.groupby(["t_dat", "customer_id", "article_id", "sales_channel_id"])
        .size()
        .reset_index()
        .rename(columns={0: "article_id_freq"})
    )
    out["article_id_freq"] = out["article_id_freq"].clip(1, 9)
    out = out.groupby("customer_id")[
        ["article_id", "sales_channel_id", "article_id_freq", "t_dat"]
    ].agg(lambda x: list(x))

    def serial_number(t_dat_seq):
        u_t_dat = sorted(list(set(t_dat_seq)))
        t_dat_map = {d: min(i + 1, 9) for i, d in enumerate(u_t_dat)}
        return [t_dat_map[d] for d in t_dat_seq]

    out["active_token_id"] = out["t_dat"].apply(lambda x: serial_number(x))
    out.drop("t_dat", axis=1, inplace=True)

    out = y.drop("article_id", axis=1).merge(out, on="customer_id", how="left")
    out = out.drop("customer_id", axis=1).to_numpy()
    return out


@feature(use_cache=True)
def weeks_before_seq(data, y, t_dat_max, save_dir):
    weeks_before = (t_dat_max - data["t_dat"]).dt.days // 7
    weeks_before += 1
    data["weeks_before"] = weeks_before / weeks_before.max()
    gdf = data.groupby("customer_id")["weeks_before"].agg(lambda x: list(x))

    gdf = y.drop("article_id", axis=1).merge(gdf, on="customer_id", how="left")
    weeks_before = gdf["weeks_before"].to_numpy()
    return weeks_before


@feature(use_cache=False)
def customer_feat(X, y, save_dir):
    customers = pd.read_csv("../data/raw/customers.csv")
    customers["age"] = customers["age"] / 100
    customers["age"].fillna(-1, inplace=True)
    customers["fashion_news_frequency"] = (
        customers["fashion_news_frequency"].isin(["Regularly", "Monthly"]).astype(int)
    )
    customers["club_member_status"] = (
        customers["club_member_status"].isin(["ACTIVE"]).astype(int)
    )

    latest_dat = X["t_dat"].max()
    last_active_dat = X.groupby("customer_id")["t_dat"].max()
    elasped_days = (
        (latest_dat - last_active_dat)
        .dt.days.to_frame()
        .rename(columns={"t_dat": "elasped_day"})
    )
    customers = customers.merge(elasped_days, how="left", on="customer_id")
    customers["elasped_day"].fillna(-1, inplace=True)
    customers["elasped_day"] = customers["elasped_day"].clip(0, 60) / 60

    customers = customers[
        [
            "customer_id",
            "club_member_status",
            "fashion_news_frequency",
            "age",
            "elasped_day",
        ]
    ]
    customers = y[["customer_id"]].merge(customers, on="customer_id", how="left")
    return customers.drop("customer_id", axis=1).to_numpy().astype("float32")


def create_feature(X: pd.DataFrame, y: pd.DataFrame, save_dir: str, t_dat_max):
    os.makedirs(save_dir, exist_ok=True)

    _ = transaction_seq(X, y, save_dir=save_dir)
    _ = weeks_before_seq(X, y, t_dat_max, save_dir=save_dir)
    _ = customer_feat(X, y, save_dir=save_dir)


@feature(use_cache=True)
def article_id_map(save_dir: str):
    # TODO: 数回しか登場しないarticleを切る, target をcountにした後でいいかも
    data = pd.read_csv("../data/raw/transactions_train.csv", dtype={"article_id": str})
    active_articles = data.groupby("article_id")["t_dat"].max().reset_index()
    active_articles = active_articles[
        active_articles["t_dat"] >= "2019-09-01"
    ].reset_index()

    article_ids = np.sort(active_articles["article_id"].to_numpy())
    articleIds_index = {a_id: i for i, a_id in enumerate(article_ids, 1)}
    return articleIds_index


def main():
    articleIds_index = article_id_map(save_dir="../data/working/")

    files = sorted(glob.glob("../data/split/*"))
    for file in files:
        print("Create feature from", file)
        filepath = pathlib.Path(file)
        f_name = filepath.name.split(".")[0]

        data = pd.read_parquet(filepath)
        data = data[data["article_id"].isin(articleIds_index.keys())]
        data["article_id"] = data["article_id"].map(articleIds_index)
        t_dat_max = data["t_dat"].max()

        if "submission" in file:
            submit = pd.read_csv("../data/raw/sample_submission.csv")
            # Create features
            submit.rename(columns={"prediction": "article_id"}, inplace=True)
            save_dir = f"../data/feature/{f_name}/"
            create_feature(data, submit, save_dir, t_dat_max)
        else:
            X, y = Xy_splitter(data)
            y = create_target(y)
            y.to_pickle(f"../data/feature/{f_name}_y.pkl")

            # Create features
            save_dir = f"../data/feature/{f_name}/"
            create_feature(X, y, save_dir, t_dat_max)


if __name__ == "__main__":
    main()
