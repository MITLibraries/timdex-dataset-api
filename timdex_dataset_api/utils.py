"""timdex_dataset_api/utils.py"""

import logging
import os
import pathlib
import time
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Unpack
from urllib.parse import urlparse

import boto3
from duckdb.duckdb import DuckDBPyConnection  # type: ignore[import-untyped]
from duckdb_engine import ConnectionWrapper
from sqlalchemy import (
    MetaData,
    Table,
    and_,
    create_engine,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.service_resource import S3ServiceResource

    from timdex_dataset_api.dataset import DatasetFilters

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(
        self,
    ) -> None:
        self.resource = self._create_resource()

    def _create_resource(self) -> "S3ServiceResource":
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

    def list_objects(self, s3_prefix: str) -> list[str]:
        bucket, _ = self._split_s3_uri(s3_prefix)
        objects = [obj.key for obj in self.resource.Bucket(bucket).objects.all()]
        logger.debug(f"Found {len(objects)} objects in {s3_prefix}: {objects}")
        return objects

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


def sa_reflect_duckdb_conn(
    conn: DuckDBPyConnection, schema: str | None = None
) -> MetaData:
    """Use reflection to return SQLAlchemy metadata about a DuckDB connection.

    Args:
        - conn: DuckDB connection
        - schema: if provided, schema to reflect from; default of None results in the
        DuckDB 'main' schema
    """
    start_time = time.perf_counter()
    db_metadata = MetaData()

    engine = create_engine(
        "duckdb://",
        creator=lambda: ConnectionWrapper(conn),
    )

    db_metadata.reflect(
        bind=engine,
        schema=schema,
        views=True,
    )
    logger.debug(
        f"SQLAlchemy reflection elapsed: {round(time.perf_counter() - start_time, 3)}s"
    )

    return db_metadata


def coerce_sa_predicate(field: str, value: Any) -> Any:  # noqa: ANN401
    """Convert a DatasetFilter value into a more convenient or universal type."""
    if field == "run_date":
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)

    if field == "run_timestamp":
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            iso = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

    if field == "run_record_offset":
        return int(value)

    return value


def build_filter_expr_sa(
    meta_table: Table,
    **filters: Unpack["DatasetFilters"],
) -> Any:  # noqa: ANN401
    """Build a SQLAlchemy WHERE clause predicate based on key/value DatasetFilters.

    At this time, only an 'AND' style WHERE clause is supported when DatasetFilters are
    passed.  Note that most TIMDEXDataset.read methods also support a 'where' argument
    that will accept raw SQL if this limitation is problematic.
    """
    predicates = []

    for key, value in filters.items():
        col = getattr(meta_table.c, key, None)

        if col is None:
            raise ValueError(
                f"Could not find column '{key}' on  table '{meta_table.name}'."
            )

        if value is None:
            predicates.append(col.is_(None))

        elif isinstance(value, list):
            coerced = [coerce_sa_predicate(key, v) for v in value]
            predicates.append(col.in_(coerced))

        else:
            predicates.append(col == coerce_sa_predicate(key, value))

    if predicates:
        return and_(*predicates)
    return None
