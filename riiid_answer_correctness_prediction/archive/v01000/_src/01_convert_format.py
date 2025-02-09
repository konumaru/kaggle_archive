import os
import dask_cudf
from utils.common import timer


def main():
    load_dir = "../data/raw"
    dump_dir = "../data/preprocess"

    with timer("Convert train to Parquert"):
        train = dask_cudf.read_csv(
            os.path.join(load_dir, "train.csv"),
            low_memory=False,
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
        train.to_parquet(os.path.join(dump_dir, "train.parquet"))
        print(train.head())

    with timer("Convert questions to Parquert"):
        questions = dask_cudf.read_csv(
            os.path.join(load_dir, "questions.csv"), low_memory=False
        )
        questions.to_parquet(os.path.join(dump_dir, "questions.parquet"))
        print(questions.head())

    with timer("Convert lectures to Parquert"):
        lectures = dask_cudf.read_csv(
            os.path.join(load_dir, "lectures.csv"), low_memory=False
        )
        lectures.to_parquet(os.path.join(dump_dir, "lectures.parquet"))
        print(lectures.head())


if __name__ == "__main__":
    main()
