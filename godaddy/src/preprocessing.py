import pathlib
from typing import Dict, Tuple

import pandas as pd

from utils.io import save_pickle


def pivot_census(census: pd.DataFrame) -> pd.DataFrame:

    dfs = []
    for year in range(2017, 2022):
        cols = census.columns[census.columns.str.contains(str(year))].tolist()
        cols += ["cfips"]

        df = census[cols]
        df.columns = [c.replace(f"_{year}", "") for c in cols]
        df = df.assign(year=year)
        dfs.append(df)

    pivoted = pd.concat(dfs, axis=0).reset_index(drop=True)
    return pivoted


def concate_census_ex() -> pd.DataFrame:
    census_ex_dfs: Dict[int, pd.DataFrame] = {}

    for year in range(2017, 2024):
        use_cols = ["cfips", "year"]
        feat_cols = [
            "S0101_C01_020E",
            "S0101_C01_021E",
            "S0101_C01_022E",
            "S0101_C01_023E",
            "S0101_C01_024E",
            "S0101_C01_025E",
            "S0101_C01_026E",
            "S0101_C01_027E",
            "S0101_C01_028E",
            "S0101_C01_029E",
            "S0101_C01_030E",
            "S0101_C01_031E",
        ]
        if year < 2022:
            df = pd.read_csv(
                f"./data/external/census-data-for-godaddy/ACSST5Y{year}.S0101-Data.csv"
            )
            df = df.iloc[1:]
            df = df.assign(
                cfips=df["GEO_ID"].apply(lambda x: int(x.split("US")[-1])), year=year
            )

            df = df[use_cols + feat_cols]
        else:
            df = census_ex_dfs[2021].copy()
            df["year"] = year
            df[feat_cols] = pd.NA

        census_ex_dfs[year] = df

    census_ex = pd.concat(list(census_ex_dfs.values()), axis=0).reset_index(drop=True)
    return census_ex


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = pathlib.Path("./data/raw")

    train = pd.read_csv(input_dir / "train.csv")
    revealed_test = pd.read_csv(input_dir / "revealed_test.csv")
    test = pd.read_csv(input_dir / "test.csv")
    census = pd.read_csv(input_dir / "census_starter.csv")

    test = test.merge(
        revealed_test.drop(["first_day_of_month", "cfips"], axis=1),
        how="left",
        on=["row_id"],
    )

    data = (
        pd.concat([train, test])
        .sort_values(by=["cfips", "first_day_of_month"])
        .reset_index(drop=True)
    )
    data["first_day_of_month"] = pd.to_datetime(
        data["first_day_of_month"], format="%Y-%m-%d"
    )
    data = data.assign(year=data["first_day_of_month"].dt.year)

    data = data.assign(
        microbusiness_density=data.groupby("cfips")["microbusiness_density"].ffill(),
        active=data.groupby("cfips")["active"].ffill(),
    )

    census_pivoted = pivot_census(census)

    data = data.merge(census_pivoted, how="left", on=["cfips", "year"])
    data = data.assign(
        **{
            c: data.groupby(["cfips", "year"])[c].ffill()
            for c in [
                "pct_bb",
                "pct_college",
                "pct_foreign_born",
                "pct_it_workers",
                "median_hh_inc",
            ]
        }
    )

    return data, census_pivoted


def main():
    data, census_pivoted = load_data()
    census_ex = concate_census_ex()

    census = pd.merge(census_ex, census_pivoted, how="left", on=["year", "cfips"])

    print(data.head())
    print(census.head())

    data.to_csv("./data/preprocessing/data.csv", index=False)
    census_pivoted.to_csv("./data/preprocessing/census_pivoted.csv", index=False)
    census_ex.to_csv("./data/preprocessing/census_ex.csv", index=False)
    census.to_csv("./data/preprocessing/census.csv", index=False)

    cfips_Mbd = (
        data[data["first_day_of_month"] < "2023-01-01"]
        .groupby("cfips")["microbusiness_density"]
        .agg("last")
        .to_dict()
    )
    save_pickle("./data/preprocessing/last_mbd_dict.pkl", cfips_Mbd)


if __name__ == "__main__":
    main()
