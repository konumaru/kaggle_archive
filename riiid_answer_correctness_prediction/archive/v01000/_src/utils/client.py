import os

import pandas as pd

from google.cloud import bigquery
from google.cloud import storage


class BQClient:
    def __init__(self, project_id: str):
        """
        Parameters
        ----------
        project_id : str
            ProjectID of Bigquery.`
        """
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)

    def run_query(self, query: str, table_id: str):
        """Run query and export resutl to table_id.

        Parameters
        ----------
        query : str
            query to run.
        table_id : str
            destination table id like `project_id.dataset_id.table_name`

        Example
        -------
        query = 'select * from project_id.dataset_id.table_name'
        client = BQClient(project_id)
        client.run_query(query, table_id=f"{project_id}.{dataset_id}.{table_name}")
        """
        job_config = bigquery.QueryJobConfig(
            destination=table_id,
            write_disposition="WRITE_TRUNCATE",
            use_legacy_sql=False,
        )
        query_job = self.client.query(query, job_config=job_config)
        query_job.result()

    def extract_table(
        self,
        destination_uri: str,
        dataset_id: str,
        table_name: str,
        location: str = "US",
    ):
        """Extract table to GoogleCloudStrage.

        Parameters
        ----------
        destination_uri : str
            Downloaded destination filepath.
        dataset_id : str
            DatsetID of Bigquery
        table_name : str
            Target table name.
        location : str, optional
            Location of project of taget table , by default "US"

        Example
        -------
        destination_uri = f"gs://{bucket_name}/data_*.csv"
        client.extract_table(destination_uri, "dataset_id", "table_name")
        """
        dataset_ref = bigquery.DatasetReference(self.project_id, dataset_id)
        table_ref = dataset_ref.table(table_name)

        extract_job = self.client.extract_table(
            table_ref, destination_uri, location=location
        )
        extract_job.result()
        print(
            f"Extracted {self.project_id}:{dataset_id}.{table_name} to {destination_uri}"
        )

    def export_df_to_bq(self, dataframe: pd.DataFrame, table_id: str):
        """Export pd.DataFrame to bigquery table.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Source data of pandas dataframe.
        table_id : str
            destination table id like `project_id.dataset_id.table_name`

        Example
        -------
        df = pd.DataFrame({"a": np.arange(10), "b": np.zeros(10)})
        client = BQClient(project_id)
        client.export_df_to_bq(query, table_id=f"{project_id}.{dataset_id}.{table_name}")
        """
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = self.client.load_table_from_dataframe(
            dataframe, table_id, job_config=job_config
        )
        job.result()


class GCSClient:
    def __init__(self, project_id: str):
        """
        Parameters
        ----------
        project_id : str
            ProjectID of Bigquery.
        """
        self.project_id = project_id
        self.client = storage.Client(project_id)

    def download_file(
        self, bucket_name: str, source_blob_name: str, destination_file_name: str
    ):
        """Download file from strage bucket.

        Parameters
        ----------
        bucket_name : str
            Bucket name of google cloud strage.
        source_blob_name : str
            Source blob name, like hogehoge/fugafuga.csv
        destination_file_name : str
            Saved filename or filepath.

        Example
        -------
        client = GCSClient(project_id)
        client.download_file(
            bucket_name,
            "hogehoge/fugafuga.csv",
            destination_file_name="fugafuga.csv",
        )
        """

        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

    def download_files(self, bucket_name: str, blob_prefix: str, destination_dir: str):
        """Download files from strage bucket.

        Parameters
        ----------
        bucket_name : str
            Bucket name of google cloud strage.
        blob_prefix : str
            Source blob prefix, like hogehoge/fugafuga_
        destination_dir : str
            Saved dir path.

        Example
        -------
        client = GCSClient(project_id)
        client.download_files(
            bucket_name,
            "hogehoge/fugafuga/",
            destination_dir="fugafuga/",
        )
        """
        bucket = self.client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=blob_prefix)
        for blob in blobs:
            filename = blob.name.replace("/", "_")
            destination_filepath = os.path.join(destination_dir, filename)
            blob.download_to_filename(destination_filepath)
            print(f"Blob {filename} downloaded to {destination_filepath}.")

    def delete_blob(self, bucket_name: str, blob_name: str):
        """Delete blob from strage bucket.

        Parameters
        ----------
        bucket_name : str
            Bucket name of google cloud strage.
        blob_name : str
            Delete blob name, like hogehoge/fugafuga.csv.

        Example
        -------
        client = GCSClient(project_id)
        client.delete_blob(
            bucket_name,
            "hogehoge/fugafuga.csv"
        )
        """
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        print(f"Blob {blob_name} deleted.")

    def delete_blobs(self, bucket_name: str, blob_prefix: str):
        """Delete blobs from strage bucket.

        Parameters
        ----------
        bucket_name : str
            Bucket name of google cloud strage.
        blob_prefix : str
            Delete blob prefix, like hogehoge/fugafuga_.

        Example
        -------
        client = GCSClient(project_id)
        client.delete_blobs(
            bucket_name,
            "train_agged_features_sample/train_agged_",
        )
        """
        bucket = self.client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=blob_prefix)
        for blob in blobs:
            filename = blob.name.replace("/", "_")
            blob.delete()
            print(f"Blob {filename} deleted.")
