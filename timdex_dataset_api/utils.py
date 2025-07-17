# Python
import logging
import os
import pathlib
from typing import Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.client import BaseClient

logger = logging.getLogger(__name__)


class S3Client:
    """
    Convenience wrapper around boto3 that accepts full S3 URIs instead of
    separate bucket / key arguments.

    • If the environment variable MINIO_S3_ENDPOINT_URL is set, the client
      automatically talks to that endpoint (handy for MinIO or other
      S3-compatible stores).
    • Otherwise it uses the default AWS endpoint & credential resolution chain.
    """

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        endpoint_env_var: str = "MINIO_S3_ENDPOINT_URL",
        default_region: str = "us-east-1",
    ) -> None:
        self._endpoint_env_var = endpoint_env_var
        self._default_region = default_region
        self._client: BaseClient = self._create_client()

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #

    def download_file(self, s3_uri: str, local_path: str | pathlib.Path) -> None:
        """
        Download *s3_uri* → *local_path*.

        Example
        -------
        s3.download_file("s3://my-bucket/data/file.csv", "/tmp/file.csv")
        """
        bucket, key = self._split_s3_uri(s3_uri)

        local_path = pathlib.Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        self._client.download_file(bucket, key, str(local_path))
        logger.info(f"Downloaded {s3_uri} → {local_path}")

    def upload_file(self, local_path: str | pathlib.Path, s3_uri: str) -> None:
        """
        Upload *local_path* → *s3_uri*.

        Example
        -------
        s3.upload_file("report.csv", "s3://my-bucket/reports/2025-q2.csv")
        """
        bucket, key = self._split_s3_uri(s3_uri)

        local_path = pathlib.Path(local_path)
        self._client.upload_file(str(local_path), bucket, key)
        logger.info(f"Uploaded {local_path} → {s3_uri}")

    def delete_file(self, s3_uri: str) -> None:
        """
        Delete the object referenced by *s3_uri*.

        Example
        -------
        s3.delete_file("s3://my-bucket/old/unused.txt")
        """
        bucket, key = self._split_s3_uri(s3_uri)

        self._client.delete_object(Bucket=bucket, Key=key)
        logger.info(f"Deleted {s3_uri}")

    def delete_folder(self, s3_uri: str, batch_size: int = 1_000) -> int:
        """
        Delete **all** objects whose keys start with the given prefix.

        Parameters
        ----------
        s3_uri : str
            Prefix expressed as a URI, e.g. ``s3://my-bucket/logs/2025/``.
        batch_size : int, optional
            How many keys to send in each `DeleteObjects` call (max 1 000).

        Returns
        -------
        int
            The total number of objects deleted.

        Notes
        -----
        S3 has no real folders, only key prefixes.  This helper paginates over
        the keys and issues batched `DeleteObjects` requests until everything
        under the prefix is gone.
        """
        bucket, prefix = self._split_s3_uri(s3_uri)

        paginator = self._client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        deleted_total = 0
        objects_to_delete: list[dict] = []

        for page in pages:
            for obj in page.get("Contents", []):
                objects_to_delete.append({"Key": obj["Key"]})

                # Send a batch delete when we reach the chosen batch size
                if len(objects_to_delete) == batch_size:
                    self._flush_batch(bucket, objects_to_delete)
                    deleted_total += len(objects_to_delete)
                    objects_to_delete.clear()

        # Flush any remaining keys (last partial batch)
        if objects_to_delete:
            self._flush_batch(bucket, objects_to_delete)
            deleted_total += len(objects_to_delete)

        print(f"Deleted {deleted_total} object(s) under prefix {s3_uri}")
        return deleted_total

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _flush_batch(self, bucket: str, objects: list[dict]) -> None:
        """
        Helper: issue a `DeleteObjects` call for a batch of keys.
        """
        self._client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": objects, "Quiet": True},
        )

    def _create_client(self) -> BaseClient:
        """
        Instantiate a boto3 S3 client, optionally pointing at a custom
        (MinIO) endpoint.
        """
        endpoint_url: Optional[str] = os.getenv(self._endpoint_env_var)
        if endpoint_url:
            return boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=os.getenv("MINIO_USERNAME"),
                aws_secret_access_key=os.getenv("MINIO_PASSWORD"),
                region_name=os.getenv("MINIO_REGION", self._default_region),
            )
        return boto3.client("s3")

    @staticmethod
    def _split_s3_uri(s3_uri: str) -> Tuple[str, str]:
        """
        Validate and split an S3 URI into (bucket, key).

        Raises
        ------
        ValueError
            If *s3_uri* is not a valid S3 URI.
        """
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
            raise ValueError(f"Invalid S3 URI: {s3_uri!r}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")  # strip leading slash from /key
        return bucket, key
