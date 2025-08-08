"""timdex_dataset_api/utils.py"""

import logging
import os
import pathlib
from urllib.parse import urlparse

import boto3
from mypy_boto3_s3.service_resource import S3ServiceResource

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(
        self,
    ) -> None:
        self.resource = self._create_resource()

    def _create_resource(self) -> S3ServiceResource:
        """Instantiate a boto3 S3 resource.

        If env var MINIO_S3_ENDPOINT_URL is set, assume using local set of MinIO env vars.
        """
        endpoint_url = os.getenv("MINIO_S3_ENDPOINT_URL")
        if endpoint_url:
            logger.debug("MinIO env vars detected, using for S3Client.")
            return boto3.resource(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=os.getenv("MINIO_USERNAME"),
                aws_secret_access_key=os.getenv("MINIO_PASSWORD"),
                region_name=os.getenv("MINIO_REGION", "us-east-1"),
            )
        return boto3.resource("s3")

    def object_exists(self, s3_uri: str) -> bool:
        bucket, key = self._split_s3_uri(s3_uri)
        try:
            self.resource.Object(bucket, key).load()
            return True  # noqa: TRY300
        except self.resource.meta.client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def download_file(self, s3_uri: str, local_path: str | pathlib.Path) -> None:
        bucket, key = self._split_s3_uri(s3_uri)
        local_path = pathlib.Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.resource.Bucket(bucket).download_file(key, str(local_path))
        logger.info(f"Downloaded {s3_uri} to {local_path}")

    def upload_file(self, local_path: str | pathlib.Path, s3_uri: str) -> None:
        bucket, key = self._split_s3_uri(s3_uri)
        local_path = pathlib.Path(local_path)
        self.resource.Bucket(bucket).upload_file(str(local_path), key)
        logger.info(f"Uploaded {local_path} to {s3_uri}")

    def delete_file(self, s3_uri: str) -> None:
        bucket, key = self._split_s3_uri(s3_uri)
        self.resource.Object(bucket, key).delete()
        logger.info(f"Deleted {s3_uri}")

    def delete_folder(self, s3_uri: str) -> list[str]:
        """Delete all objects whose keys start with the given prefix."""
        bucket, prefix = self._split_s3_uri(s3_uri)
        bucket_obj = self.resource.Bucket(bucket)
        receipt = bucket_obj.objects.filter(Prefix=prefix).delete()

        deleted_keys = []
        for request in receipt:
            deleted_keys.extend([item["Key"] for item in request["Deleted"]])
        logger.info(f"Deleted objects with prefix '{s3_uri}': {deleted_keys}")
        return deleted_keys

    @staticmethod
    def _split_s3_uri(s3_uri: str) -> tuple[str, str]:
        """Validate and split an S3 URI into (bucket, key)."""
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
            raise ValueError(f"Invalid S3 URI: {s3_uri!r}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")  # strip leading slash from /key
        return bucket, key
