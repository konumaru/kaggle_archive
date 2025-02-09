import gc
import pickle

import numpy as np
import pandas as pd

# from cuml.decomposition import IncrementalPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def transform(pca, data, batch_size=50000):
    outputs = []
    for start_idx in tqdm(range(0, len(data) + batch_size, batch_size)):
        end_idx = min(start_idx + batch_size, len(data))
        batch_data = data[start_idx:end_idx, :]
        if len(batch_data) > 0:
            output = pca.transform(batch_data)
            outputs.append(output)
    return np.concatenate(outputs, axis=0)


def main():
    train = pd.read_parquet("./output/train_fe.parquet")

    all_cols = train.columns.tolist()
    num_cols = train.dtypes[train.dtypes == "float32"].index
    other_cols = [c for c in all_cols if c not in num_cols]

    # Fillna.
    avg_dict = train[num_cols].mean().to_dict()
    for col_name, avg_value in tqdm(avg_dict.items()):
        train[col_name].fillna(avg_value, inplace=True)
    train.fillna(0.0, inplace=True)

    train_pca = train[num_cols].to_numpy()
    train_other = train[other_cols].copy()

    del train
    gc.collect()

    n_components = 1800
    encoder = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", IncrementalPCA(n_components=n_components, batch_size=2048)),
        ]
    )
    encoder.fit(train_pca)

    with open(f"./output/pca{n_components}_encoder.pkl", "wb") as file:
        pickle.dump(encoder, file)

    # with open(f"./output/pca{n_components}_encoder.pkl", "rb") as file:
    #     pca = pickle.load(file)

    pca_col_name = [f"pca_{i}" for i in range(n_components)]

    train_pca = transform(encoder, train_pca)
    train_pca = pd.DataFrame(train_pca, columns=pca_col_name, dtype="float32")
    assert train_other.shape[0] == train_pca.shape[0]
    train = pd.concat([train_other, train_pca], axis=1)
    print(train.head())
    train.to_parquet(f"./output/train_fe_pca{n_components}.parquet", index=False)

    print(train.head())

    del train, train_pca
    gc.collect()

    test = pd.read_parquet("./output/test_fe.parquet")
    for col_name, avg_value in tqdm(avg_dict.items()):
        test[col_name].fillna(avg_value, inplace=True)
    test.fillna(0.0, inplace=True)

    test_pca = test[num_cols].to_numpy()
    test_pca = transform(encoder, test_pca)
    test_pca = pd.DataFrame(test_pca, columns=pca_col_name, dtype="float32")
    other_cols.remove("target")
    assert test.shape[0] == test_pca.shape[0]
    test = pd.concat([test[other_cols], test_pca], axis=1)
    test.to_parquet(f"./output/test_fe_pca{n_components}.parquet", index=False)

    print(test.head())


if __name__ == "__main__":
    main()
