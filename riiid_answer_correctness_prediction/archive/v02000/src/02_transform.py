import os
import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

from utils.common import timer


def preprocess(data: pd.DataFrame):
    data["prior_question_had_explanation"] = (
        data["prior_question_had_explanation"].fillna(False).astype("int8")
    )

    mean_prior_question_elapsed_time = 25424  # Calclated by all train data.
    data["prior_question_elapsed_time"].fillna(
        mean_prior_question_elapsed_time, inplace=True
    )

    return data


def dump_processed_dataset(src_filename: str, dst_filename: str):
    # Load data.
    data = pd.read_csv(
        src_filename,
        dtype={
            "row_id": "int64",
            "timestamp": "int64",
            "user_id": "int32",
            "content_id": "int16",
            "content_type_id": "int8",
            "task_container_id": "int16",
            "user_answer": "int8",
            "answered_correctly": "int8",
            "prior_question_elapsed_time": "float32",
            "prior_question_had_explanation": "boolean",
        },
    )

    data = preprocess(data)
    data = data.reset_index(drop=True)
    print(data.head())
    # Dump data.
    data.to_pickle(dst_filename)


def add_dummies(data: pd.DataFrame, column: str):
    ohe = pd.get_dummies(data[column]).add_prefix(f"{column}_")
    data = data.drop(column, axis=1)
    data = data.join(ohe)
    return data


def reduction_questions():
    def strList_to_intList(x, split_str=" "):
        if x is np.nan:
            return list()
        else:
            x = x.split(split_str)
            x = [int(_x) for _x in x]
            return x

    questions = pd.read_csv("../data/raw/questions.csv")
    questions["tags"] = questions["tags"].map(strList_to_intList)

    all_tags = []
    for _tags in questions["tags"].tolist():
        if _tags is not np.nan:
            all_tags.extend(_tags)
    all_tags = sorted(list(set(all_tags)))

    result = {}
    for q_id, tags in questions[["question_id", "tags"]].values:
        result[q_id] = [int(t in tags) for t in all_tags]

    result = pd.DataFrame.from_dict(
        result, orient="index", columns=[f"tag{t}" for t in all_tags]
    )
    result["part"] = questions["part"]
    result = add_dummies(result, "part")

    num_nmf_dim = 3
    nmf = NMF(n_components=num_nmf_dim, init="random", random_state=42)
    nmf_data = nmf.fit_transform(result.to_numpy())

    num_lda_dim = 5
    lda = LatentDirichletAllocation(n_components=num_lda_dim, random_state=42)
    lda_data = lda.fit_transform(result.to_numpy())

    print(pd.DataFrame(lda_data[:, 1]).head())

    data = pd.DataFrame(
        np.concatenate([nmf_data, lda_data], axis=1),
        columns=[f"nmf-{i}" for i in range(num_nmf_dim)]
        + [f"lda-{i}" for i in range(num_lda_dim)],
    )
    data["question_id"] = questions["question_id"]
    return data


def main():
    src_dir = "../data/01_preprocessing/"
    dst_dir = "../data/02_transform/"

    dump_processed_dataset(
        os.path.join(src_dir, "train_sample_trainset.csv"),
        os.path.join(dst_dir, "train.pkl"),
    )
    dump_processed_dataset(
        os.path.join(src_dir, "train_sample_evalset.csv"),
        os.path.join(dst_dir, "eval.pkl"),
    )

    content = pd.read_csv(os.path.join(src_dir, "content_features.csv"))
    content = add_dummies(content, "part")
    content["num_tags"] = content["tags"].apply(lambda x: len(str(x).split(" ")))
    content.drop(["correct_answer", "tags"], axis=1, inplace=True)

    nmf = reduction_questions()
    content = content.merge(
        nmf, how="left", left_on="content_id", right_on="question_id"
    )

    print(content.columns)
    print(content.head())
    content.to_pickle(os.path.join(dst_dir, "content.pkl"))


if __name__ == "__main__":
    with timer("Load and preprocessing"):
        main()
