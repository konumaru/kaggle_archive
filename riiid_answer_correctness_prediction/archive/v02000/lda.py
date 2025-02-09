import os
import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

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

    questions = pd.read_csv("../../data/raw/questions.csv")
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
    data = reduction_questions()
    data.to_csv('reduction_questions.csv', index=False)
    print(data.head())

if __name__=='__main__':
    main()
