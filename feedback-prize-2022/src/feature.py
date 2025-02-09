import collections
from typing import Dict

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords


def count_pos_tag(text):
    pos_tag = nltk.pos_tag(text)
    _, pos = zip(*pos_tag)
    pos_counts = collections.Counter(pos)
    return pos_counts


def get_text_features(text: str) -> Dict[str, float]:
    stop_words = stopwords.words("english")  # type: ignore

    feature = {}
    feature["char_count"] = len(text)
    feature["word_count"] = len(text.split())
    feature["sentence_count"] = text.count(".") + text.count("\n")
    feature["word_per_sentence"] = (
        feature["word_count"] / feature["sentence_count"]
        if feature["sentence_count"] > 0
        else np.nan
    )
    feature["unique_word_count"] = len(set(text.split()))
    feature["max_word_len"] = max([len(w) for w in text.split()])
    feature["word_density"] = feature["char_count"] / feature["word_count"]
    feature["upper_case_word_count"] = len([w for w in text.split() if w.isupper()])
    feature["stopword_count"] = len(
        [w for w in text.split() if w.lower() in stop_words]
    )

    feature.update(count_pos_tag(text))

    return feature


def main():
    train = pd.read_csv("../data/raw/train.csv")

    features = train.loc[:5].apply(
        lambda x: get_text_features(str(x["full_text"])),  # type: ignore
        axis=1,
        result_type="expand",
    )
    print(features)
    print(features.columns.tolist())


if __name__ == "__main__":
    main()
