import os
import glob

import cudf
import dask_cudf

import numpy as np
import pandas as pd

from utils.common import timer
from utils.common import cache_result
from utils.common import load_text
from utils.client import BQClient, GCSClient

from dotenv import load_dotenv

load_dotenv()


def export_table(filepath, table_name):
    project_id = os.environ["PROJECT_ID"]
    dataset_id = os.environ["DATASET_ID"]

    query = load_text(filepath)
    client = BQClient(project_id)
    client.run_query(query, f"{project_id}.{dataset_id}.{table_name}")


def extract_table(table_name, filename):
    # Clear already exit blobs.
    bucket_name = os.environ["BUCKET_NAME"]
    blob_prefix = f"{table_name}/data_"
    client = GCSClient(os.environ["PROJECT_ID"])
    client.delete_blobs(bucket_name, blob_prefix)
    # BQ to GCS.
    client = BQClient(os.environ["PROJECT_ID"])
    client.extract_table(
        destination_uri=f"gs://{bucket_name}/{filename}",
        dataset_id=os.environ["DATASET_ID"],
        table_name=table_name,
    )


def download_file(table_name):
    bucket_name = os.environ["BUCKET_NAME"]
    source_blob_name = f"{table_name}/data.csv"
    destination_file_name = f"../data/{table_name}/data.csv"

    if not os.path.exists(f"../data/{table_name}"):
        os.makedirs(f"../data/{table_name}")

    client = GCSClient(os.environ["PROJECT_ID"])
    client.download_file(
        bucket_name,
        source_blob_name=source_blob_name,
        destination_file_name=destination_file_name,
    )


def download_files(table_name):
    bucket_name = os.environ["BUCKET_NAME"]
    blob_prefix = f"{table_name}/data_"
    destination_dir = f"../data/{table_name}/"

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    client = GCSClient(os.environ["PROJECT_ID"])
    client.download_files(
        bucket_name,
        blob_prefix=blob_prefix,
        destination_dir=destination_dir,
    )


def clean_dump_dirs():
    dirs = [
        "train_dataset",
        "eval_dataset",
    ]
    for d in dirs:
        files = glob.glob(f"../data/{d}/*.csv", recursive=True)

        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))


def export_df2bq(df, table_name):
    project_id = os.environ["PROJECT_ID"]
    dataset_id = os.environ["DATASET_ID"]
    table_id = f"{project_id}.{dataset_id}.{table_name}"

    client = BQClient(project_id)
    client.export_df_to_bq(df, table_id)
    print(f"Export dataframe to {table_id}")


def add_dummies(data: pd.DataFrame, column: str):
    ohe = pd.get_dummies(data[column]).add_prefix(f"{column}_")
    data = data.drop(column, axis=1)
    data = data.join(ohe)
    return data


def preprocess_questions():
    questions = pd.read_csv("../data/raw/questions.csv")
    questions.rename(columns={"question_id": "content_id"}, inplace=True)

    questions["encoded_tags"] = pd.factorize(questions["tags"])[0]
    questions["num_tags"] = questions["tags"].apply(lambda x: len(str(x).split(" ")))
    # One-hot encoding
    questions = add_dummies(questions, "part")

    questions.drop(["bundle_id", "correct_answer", "tags"], axis=1, inplace=True)
    return questions


def preprocess_lectures():
    lectures = pd.read_csv("../data/raw/lectures.csv")
    lectures.rename(columns={"lecture_id": "content_id"}, inplace=True)
    # One-hot encoding
    lectures["type_of"] = lectures["type_of"].str.replace(" ", "_")
    lectures = add_dummies(lectures, "type_of")
    return lectures


def main():
    clean_dump_dirs()
    # Preprocess raw data.
    questions = preprocess_questions()
    lectures = preprocess_lectures()
    df = pd.concat([questions, lectures], axis=0)

    # Export tables.
    export_df2bq(df, "content_metadata")
    # # # Feature engineering
    export_table("query/train_aggedby_user_id.sql", "train_aggedby_user_id")
    export_table("query/train_groupby_content_id.sql", "train_groupby_content_id")
    export_table("query/train_join_userId_contentId.sql", "train_join_userId_contentId")
    # Export dataset tables
    export_table("query/train_dataset.sql", "train_dataset")
    export_table("query/eval_dataset.sql", "eval_dataset")
    export_table("query/latest_train_features.sql", "latest_train_features")

    # Extract tables.
    extract_table("train_groupby_content_id", "train_groupby_content_id.csv")
    extract_table("train_dataset", "train_dataset/data_*.csv")
    extract_table("eval_dataset", "eval_dataset/data.csv")
    extract_table("latest_train_features", "latest_train_features.csv")

    # Download files from bucket.
    download_files("train_dataset")
    download_file("eval_dataset")


if __name__ == "__main__":
    main()
