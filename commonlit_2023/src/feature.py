import pathlib
import re
import statistics
import string
from typing import List

import hydra
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

from utils import timer
from utils.feature import BaseFeature, feature
from utils.io import load_pickle

FEATURE_DIR = "./data/feature"

stop_words = set(stopwords.words("english"))  # type: ignore
stop_words.add(",")
stop_words.add(".")


@feature(FEATURE_DIR)
def fold(data: pd.DataFrame) -> np.ndarray:
    return data["fold"].to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def content(data: pd.DataFrame) -> np.ndarray:
    return data["content"].to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def wording(data: pd.DataFrame) -> np.ndarray:
    return data["wording"].to_numpy().reshape(-1, 1)


def create_target_and_fold(data: pd.DataFrame) -> None:
    _data = data.copy()
    funcs = [fold, content, wording]
    for func in funcs:
        func(_data)


# -- For Feature Engineering --


def quotes_counter(row: pd.Series):
    summary = row["text"]
    text = row["prompt_text"]

    quotes_from_summary = re.findall(r'"([^"]*)"', summary)
    if len(quotes_from_summary) > 0:
        return [quote in text for quote in quotes_from_summary].count(True)
    else:
        return 0


def word_overlap_counter(row: pd.Series) -> int:
    STOP_WORDS = set(stopwords.words("english"))  # type: ignore

    def check_is_stop_word(word):
        return word in STOP_WORDS

    prompt_words = row["prompt_text"]
    summary_words = row["text"]
    if STOP_WORDS:
        prompt_words = list(filter(check_is_stop_word, prompt_words))
        summary_words = list(filter(check_is_stop_word, summary_words))
    return len(set(prompt_words).intersection(set(summary_words)))


def ngram_co_occurrence_counter(row: pd.Series, n: int = 2) -> int:
    def ngrams(token, n):
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    original_tokens = row["prompt_text"].split(" ")
    summary_tokens = row["text"].split(" ")

    original_ngrams = set(ngrams(original_tokens, n))
    summary_ngrams = set(ngrams(summary_tokens, n))

    common_ngrams = original_ngrams.intersection(summary_ngrams)
    return len(common_ngrams)


def pos_tag_counter(row: pd.Series, pos: str) -> int:
    words = row["text"].split(" ")
    words = [word for word in words if len(word) > 0]
    tags = nltk.pos_tag(words)
    return len([tag for word, tag in tags if tag == pos])


def round_to_5(n):
    return round(n / 5) * 5


def analyze_text(text):
    words = text.split()
    word_lengths = [len(word) for word in words]
    # max_length = max(word_lengths)
    avg_length = sum(word_lengths) / len(word_lengths)
    median_length = statistics.median(word_lengths)
    return pd.Series([avg_length, median_length])


def clean_text(text: str) -> str:
    word_tokens = word_tokenize(text)
    filtered_sentence = [
        w.lower() for w in word_tokens if not w.lower() in stop_words
    ]
    return " ".join(filtered_sentence)


class CommonLitFeature(BaseFeature):
    def __init__(
        self,
        data: pd.DataFrame,
        sentence_encoder: SentenceTransformer,
        use_cache: bool = True,
        is_test: bool = False,
        feature_dir: str | None = None,
        preprocess_dir: str | None = None,
    ) -> None:
        super().__init__(data, use_cache, is_test, feature_dir)

        if preprocess_dir:
            self.preprocess_dir = pathlib.Path(preprocess_dir)

        self.sentence_encoder = sentence_encoder
        self.sentence_encoder.max_seq_length = 256

        self.stop_words = set(stopwords.words("english"))  # type: ignore

    @BaseFeature.cache()
    def text_length(self) -> np.ndarray:
        results = self.data["text"].str.len().to_numpy().reshape(-1, 1)
        return results

    @BaseFeature.cache()
    def word_count(self) -> np.ndarray:
        results = self.data["text"].str.split().str.len()
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def sentence_count(self) -> np.ndarray:
        results = self.data["text"].str.split(".").str.len()
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def quoted_sentence_count(self) -> np.ndarray:
        results = self.data["text"].apply(
            lambda x: len(re.findall(r'"(.*?)"', str(x)))
        )
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def consecutive_dots_count(self) -> np.ndarray:
        results = self.data["text"].apply(
            lambda x: len(re.findall(r"\.{3,4}", str(x)))
        )
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def quotes_count(self) -> np.ndarray:
        results = self.data.apply(quotes_counter, axis=1)
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def word_overlap_count(self) -> np.ndarray:
        results = self.data.apply(word_overlap_counter, axis=1)
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def ngram_co_occurrence_count(self) -> np.ndarray:
        bi_gram = self.data.apply(
            ngram_co_occurrence_counter, axis=1, args=(2,)
        )
        tr_gram = self.data.apply(
            ngram_co_occurrence_counter, axis=1, args=(3,)
        )

        results = pd.concat([bi_gram, tr_gram], axis=1)
        return results.to_numpy()

    @BaseFeature.cache()
    def spell_miss_count(self) -> np.ndarray:
        def counter(row: pd.Series) -> int:
            words = row["text"].split(" ")
            return len(SpellChecker().unknown(words))

        results = self.data.apply(counter, axis=1)
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def num_punctuations(self) -> np.ndarray:
        results = self.data["text"].apply(
            lambda x: len([c for c in str(x) if c in list(string.punctuation)])
        )
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def num_stopwords(self) -> np.ndarray:
        results = self.data["text"].apply(
            lambda x: len(
                [w for w in str(x).lower().split() if w in self.stop_words]
            )
        )
        return results.to_numpy().reshape(-1, 1)

    @BaseFeature.cache()
    def target_encoded_word_count(self) -> np.ndarray:
        _data = self.data.copy()
        word_cnt = self.word_count().ravel()
        _data["clipped_word_count"] = (
            pd.Series(word_cnt).clip(25, 200).apply(round_to_5)
        )

        if self.is_test:
            encoding_map = pd.read_csv(
                self.preprocess_dir / "target_encoded_word_count.csv"
            )
            _data = _data.merge(
                encoding_map, on="clipped_word_count", how="left"
            )
            results = _data[["content", "wording"]].to_numpy()
            return results
        else:
            results = (
                _data.groupby(["fold", "clipped_word_count"])[
                    ["content", "wording"]
                ]
                .transform("mean")
                .to_numpy()
            )
            return results

    @BaseFeature.cache(False)
    def prompt_text_similarity(self) -> np.ndarray:
        results = np.zeros(self.data.shape[0])
        for prompt_id in self.data["prompt_id"].unique():
            is_this_prompt = self.data["prompt_id"] == prompt_id

            p_text = (
                self.data.query("prompt_id == @prompt_id")["prompt_text"]
                .unique()
                .tolist()[0]
            )
            text = self.data.query("prompt_id == @prompt_id")["text"].tolist()

            p_text_vec = self.sentence_encoder.encode(
                p_text, normalize_embeddings=True
            )
            text_vec = self.sentence_encoder.encode(
                text, normalize_embeddings=True
            )

            results[is_this_prompt] = cosine_similarity(
                [p_text_vec], text_vec  # type: ignore
            ).ravel()
        return results.reshape(-1, 1)

    @BaseFeature.cache(False)
    def text_wv(self) -> np.ndarray:
        def get_words_avg_vec(wv, words: List[str]) -> List[float]:
            vec = []
            for w in words:
                try:
                    vec.append(wv[w])
                except KeyError:
                    pass
            return list(np.mean(vec, axis=0))

        filepaths = (
            self.preprocess_dir / "fasttext-wiki-news-subwords-300.npy",
            self.preprocess_dir / "glove-wiki-gigaword-300.npy",
        )

        tmp = []
        for filepath in filepaths:
            wv = KeyedVectors.load(str(filepath), mmap="r")

            text_words = self.data["text"].apply(clean_text).str.split(" ")
            text_wv = np.array([get_words_avg_vec(wv, w) for w in text_words])

            results = np.zeros(self.data.shape[0])
            for prompt_id in self.data["prompt_id"].unique():
                is_this_prompt = self.data["prompt_id"] == prompt_id
                p_text = (
                    self.data.query("prompt_id == @prompt_id")["prompt_text"]
                    .unique()
                    .tolist()[0]
                )
                p_text_words = clean_text(p_text).split(" ")
                p_text_wv = np.array(
                    get_words_avg_vec(wv, p_text_words)
                ).reshape(1, -1)

                results[is_this_prompt] = cosine_similarity(
                    p_text_wv, text_wv[is_this_prompt]
                ).ravel()

            tmp.append(results.reshape(-1, 1))

        return np.concatenate(tmp, axis=1)


def create_target_encoding_map(cfg: DictConfig, data: pd.DataFrame) -> None:
    feature_dir = pathlib.Path(cfg.path.feature)
    output_dir = pathlib.Path(cfg.path.preprocessed)

    word_cnt = pd.Series(load_pickle(feature_dir / "word_count.pkl").ravel())
    data["clipped_word_count"] = word_cnt.clip(25, 200).apply(round_to_5)
    encoding_map = data.groupby(["clipped_word_count"])[
        ["content", "wording"]
    ].mean()
    encoding_map.to_csv(output_dir / "target_encoded_word_count.csv")


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path(cfg.path.preprocessed)

    train = pd.read_csv(input_dir / "train.csv")
    print(train.shape)

    create_target_and_fold(train)

    features = CommonLitFeature(
        train,
        sentence_encoder=SentenceTransformer(
            "all-MiniLM-L6-v2", device="cuda:0"
        ),
        feature_dir=cfg.path.feature,
        preprocess_dir=cfg.path.preprocessed,
    )
    results = features.create_features()
    print(pd.DataFrame(results).info())

    create_target_encoding_map(cfg, train)


if __name__ == "__main__":
    with timer("Create feature"):
        main()
