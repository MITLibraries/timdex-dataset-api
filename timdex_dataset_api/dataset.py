"""timdex_dataset_api/dataset.py"""

import time

import boto3
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import fs

from timdex_dataset_api.config import configure_logger
from timdex_dataset_api.exceptions import DatasetNotLoadedError

logger = configure_logger(__name__)

TIMDEX_DATASET_SCHEMA = pa.schema(
    (
        pa.field("timdex_record_id", pa.string()),
        pa.field("source_record", pa.binary()),
        pa.field("transformed_record", pa.binary()),
        pa.field("source", pa.string()),
        pa.field("run_date", pa.date32()),
        pa.field("run_type", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("action", pa.string()),
    )
)


class TIMDEXDataset:

    def __init__(self, location: str | list[str]):
        self.location = location
        self.dataset: ds.Dataset = None  # type: ignore[assignment]
        self.schema = TIMDEX_DATASET_SCHEMA

    @classmethod
    def load(cls, location: str) -> "TIMDEXDataset":
        """Return an instantiated TIMDEXDataset object given a dataset location.

        Argument 'location' may be a local filesystem path or an S3 URI to a parquet
        dataset.
        """
        timdex_dataset = cls(location=location)
        timdex_dataset.dataset = timdex_dataset.load_dataset()
        return timdex_dataset

    @staticmethod
    def get_s3_filesystem() -> fs.FileSystem:
        """Instantiate a pyarrow S3 Filesystem for dataset loading."""
        session = boto3.session.Session()
        credentials = session.get_credentials()
        if not credentials:
            raise RuntimeError("Could not locate AWS credentials")
        return fs.S3FileSystem(
            secret_key=credentials.secret_key,
            access_key=credentials.access_key,
            region=session.region_name,
            session_token=credentials.token,
        )

    @staticmethod
    def parse_location(
        location: str | list[str],
    ) -> tuple[fs.FileSystem, str | list[str]]:
        """Parse and return the filesystem and normalized source location(s).

        Handles both single location strings and lists of Parquet file paths.
        """
        source: str | list[str]
        if isinstance(location, str):
            if location.startswith("s3://"):
                filesystem = TIMDEXDataset.get_s3_filesystem()
                source = location.removeprefix("s3://")
            else:
                filesystem = fs.LocalFileSystem()
                source = location
        elif isinstance(location, list):
            if all(loc.startswith("s3://") for loc in location):
                filesystem = TIMDEXDataset.get_s3_filesystem()
                source = [loc.removeprefix("s3://") for loc in location]
            elif all(not loc.startswith("s3://") for loc in location):
                filesystem = fs.LocalFileSystem()
                source = location
            else:
                raise ValueError("Mixed S3 and local paths are not supported.")
        else:
            raise TypeError("Location type must be str or list[str].")

        return filesystem, source

    def load_dataset(self) -> ds.Dataset:
        """Lazy load a pyarrow.Dataset for an already instantiated TIMDEXDataset object.

        The dataset is loaded via the expected schema as defined by module constant
        TIMDEX_DATASET_SCHEMA.  If the target dataset differs in any way, errors may be
        raised when reading or writing data.
        """
        start_time = time.perf_counter()
        filesystem, source = self.parse_location(self.location)
        dataset = ds.dataset(
            source,
            schema=self.schema,
            format="parquet",
            partitioning="hive",
            filesystem=filesystem,
        )
        logger.info(
            f"Dataset successfully loaded: '{self.location}', "
            f"{round(time.perf_counter()-start_time, 2)}s"
        )
        return dataset

    @property
    def row_count(self) -> int:
        """Get row count from loaded dataset."""
        if not self.dataset:
            raise DatasetNotLoadedError
        return self.dataset.count_rows()
