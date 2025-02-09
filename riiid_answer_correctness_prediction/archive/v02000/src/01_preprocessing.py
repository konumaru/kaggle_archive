import os
import glob

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
    destination_file_name = f"../data/01_preprocessing/{table_name}.csv"

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


def main():
    export_table("./query/train_add_random.sql", "train_add_random")

    export_table("./query/train_sample_trainset.sql", "train_sample_trainset")
    extract_table("train_sample_trainset", "train_sample_trainset/data.csv")
    download_file("train_sample_trainset")

    export_table("./query/train_sample_evalset.sql", "train_sample_evalset")
    extract_table("train_sample_evalset", "train_sample_evalset/data.csv")
    download_file("train_sample_evalset")

    export_table("./query/content_features.sql", "content_features")
    extract_table("content_features", "content_features/data.csv")
    download_file("content_features")

    export_table("./query/task_container_feature.sql", "task_container_feature")
    extract_table("task_container_feature", "task_container_feature/data.csv")
    download_file("task_container_feature")


if __name__ == "__main__":
    main()
