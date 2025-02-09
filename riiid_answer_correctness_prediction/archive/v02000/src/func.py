import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import deque
from collections import defaultdict
from joblib import Parallel, delayed

from utils.common import dump_pickle


class UserState:
    """memo
    - past_XXXX で現時点より過去の特徴量を表す
    - sum は基本的に特徴量とせず、avgを使う
    """

    def __init__(self, is_test=False):
        self.state = defaultdict(self._init_dict)
        self.is_test = is_test

    def _init_dict(self):
        return {
            "row_id": int,
            "user_id": int,
            "timestamp": int,
            "content_id": int,
            "content_type_id": int,
            "prior_question_elapsed_time": int,
            "prior_question_had_explanation": int,
            "answered_correctly": int,
            # Prior features.
            "past_answer_correctly": list(),
            "past_user_answer": list(),
            "past_content_type_id": list(),
            "past_timestamp": list(),
            "past_prior_question_elapsed_time": list(),
            "past_prior_question_had_explanation": list(),
        }

    def update_state(self, row):
        user_id = row["user_id"]

        if row["content_type_id"] == 0:
            if self.is_test:
                self.state[user_id]["past_answer_correctly"] = row[
                    "prior_group_answers_correct"
                ]
                self.state[user_id]["past_user_answer"] = row["prior_group_responses"]
            else:
                self.state[user_id]["past_answer_correctly"].append(
                    row["answered_correctly"]
                )
                self.state[user_id]["past_user_answer"].append(row["user_answer"])

            self.state[user_id]["past_timestamp"].append(row["timestamp"])
            self.state[user_id]["past_prior_question_elapsed_time"].append(
                row["prior_question_elapsed_time"]
            )
            self.state[user_id]["past_prior_question_had_explanation"].append(
                row["prior_question_had_explanation"]
            )
            self.state[user_id]["past_timestamp"].append(row["timestamp"])
        else:
            # TODO:
            # 過去のlectureのtype_ofのcount, ratio
            # lecture_partのcount, ration
            pass
        self.state[user_id]["past_content_type_id"].append(row["content_type_id"])

    def get_feature(self, row):
        user_state = self.state[row["user_id"]]

        # =========================
        # Current features.
        # =========================
        feature = {
            "row_id": row["row_id"],
            "user_id": row["user_id"],
            "timestamp": row["timestamp"],
            "content_id": row["content_id"],
            "content_type_id": row["content_type_id"],
            "task_container_id": row["task_container_id"],
            "prior_question_elapsed_time": row["prior_question_elapsed_time"],
            "prior_question_had_explanation": row["prior_question_had_explanation"],
            "answered_correctly": row["answered_correctly"],
        }
        # =========================
        # Spot features
        # =========================
        if (len(user_state["past_answer_correctly"]) > 0) and (
            row["task_container_id"] > 0
        ):
            feature.update(
                {
                    "div_answered_count_task_container_id": (
                        len(user_state["past_answer_correctly"])
                        / row["task_container_id"]
                    )
                }
            )
        else:
            feature.update({"div_answered_count_task_container_id": np.nan})

        # =========================
        # Past features.
        # =========================
        cols = [
            "prior_question_elapsed_time",
            "prior_question_had_explanation",
            "content_type_id",
            "answer_correctly",
        ]
        for col in cols:
            if len(user_state[f"past_{col}"]) > 0:
                feature.update(
                    {
                        f"past_avg_{col}": np.mean(user_state[f"past_{col}"]),
                        f"past_std_{col}": np.std(user_state[f"past_{col}"]),
                    }
                )
            else:
                feature.update(
                    {
                        f"past_avg_{col}": np.nan,
                        f"past_std_{col}": np.nan,
                    }
                )
        # =========================
        # Past features only recently.
        # =========================
        cols = [
            "answer_correctly",
            "content_type_id",
            "prior_question_elapsed_time",
            "prior_question_had_explanation",
        ]
        for col in cols:
            feature.update(
                {
                    f"past_avg_{col}_recently{i}": np.mean(
                        user_state[f"past_{col}"][-i:]
                    )
                    if len(user_state[f"past_{col}"]) > i
                    else np.nan
                    for i in [5, 10, 50, 100]
                }
            )
            feature.update(
                {
                    f"past_std_{col}_recently{i}": np.std(
                        user_state[f"past_{col}"][-i:]
                    )
                    if len(user_state[f"past_{col}"]) > i
                    else np.nan
                    for i in [5, 10, 50, 100]
                }
            )
        # =========================
        # Timestamp features.
        # =========================
        diff_timestamps = np.diff(user_state["past_timestamp"], n=1)
        feature.update(
            {
                f"diff_{i}_pre_timestamp": (
                    diff_timestamps[-(i + 1)] if len(diff_timestamps) > i else np.nan
                )
                for i in range(5)
            }
        )
        # =========================
        # prior_question_elapsed_time features.
        # =========================
        diff_timestamps = np.diff(user_state["past_prior_question_elapsed_time"], n=1)
        feature.update(
            {
                f"diff_{i}_pre_past_prior_question_elapsed_time": (
                    diff_timestamps[-(i + 1)] if len(diff_timestamps) > i else np.nan
                )
                for i in range(5)
            }
        )
        # =========================
        # past_user_answer features.
        # =========================
        # past_user_answer = user_state["past_user_answer"]
        # answer_count = len(past_user_answer)
        # num, count = np.unique(past_user_answer, return_counts=True)
        # for _num, _count in zip(num, count):
        #     ratio = _count / answer_count if answer_count > 0 else np.nan
        #     feature.update({f"past_user_answer_{_num}_ratio": ratio})

        return feature

    def load_state(self, filepath):
        with open(filepath, "rb") as file:
            self.state = pickle.load(file)

    def dump_state(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self.state, file, protocol=pickle.HIGHEST_PROTOCOL)


def get_feature_and_update_state(us, idx, row):
    feature = us.get_feature(row)
    us.update_state(row)
    return feature
