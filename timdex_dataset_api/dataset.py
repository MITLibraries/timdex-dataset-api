"""timdex_dataset_api/dataset.py"""

import itertools
import time
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

import boto3
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import fs

from timdex_dataset_api.config import configure_logger
from timdex_dataset_api.exceptions import DatasetNotLoadedError

if TYPE_CHECKING:
    from timdex_dataset_api.record import DatasetRecord  # pragma: nocover

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
        pa.field("year", pa.string()),
        pa.field("month", pa.string()),
        pa.field("day", pa.string()),
    )
)

TIMDEX_DATASET_PARTITION_COLUMNS = [
    "year",
    "month",
    "day",
]

DEFAULT_BATCH_SIZE = 1_000
MAX_ROWS_PER_GROUP = DEFAULT_BATCH_SIZE
MAX_ROWS_PER_FILE = 100_000


class TIMDEXDataset:

    def __init__(self, location: str | list[str]):
        self.location = location
        self.filesystem, self.source = self.parse_location(self.location)
        self.dataset: ds.Dataset = None  # type: ignore[assignment]
        self.schema = TIMDEX_DATASET_SCHEMA
        self.partition_columns = TIMDEX_DATASET_PARTITION_COLUMNS
        self._written_files: list[ds.WrittenFile] = None  # type: ignore[assignment]

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

    @classmethod
    def parse_location(
        cls,
        location: str | list[str],
    ) -> tuple[fs.FileSystem, str | list[str]]:
        """Parse and return the filesystem and normalized source location(s).

        Handles both single location strings and lists of Parquet file paths.
        """
        match location:
            case str():
                return cls._parse_single_location(location)
            case list():
                return cls._parse_multiple_locations(location)
            case _:
                raise TypeError("Location type must be str or list[str].")

    @classmethod
    def _parse_single_location(
        cls, location: str
    ) -> tuple[fs.FileSystem, str | list[str]]:
        """Get filesystem and normalized location for single location."""
        if location.startswith("s3://"):
            filesystem = TIMDEXDataset.get_s3_filesystem()
            source = location.removeprefix("s3://")
        else:
            filesystem = fs.LocalFileSystem()
            source = location
        return filesystem, source

    @classmethod
    def _parse_multiple_locations(
        cls, location: list[str]
    ) -> tuple[fs.FileSystem, str | list[str]]:
        """Get filesystem and normalized location for multiple locations."""
        if all(loc.startswith("s3://") for loc in location):
            filesystem = TIMDEXDataset.get_s3_filesystem()
            source = [loc.removeprefix("s3://") for loc in location]
        elif all(not loc.startswith("s3://") for loc in location):
            filesystem = fs.LocalFileSystem()
            source = location
        else:
            raise ValueError("Mixed S3 and local paths are not supported.")
        return filesystem, source

    def load_dataset(self) -> ds.Dataset:
        """Lazy load a pyarrow.Dataset for an already instantiated TIMDEXDataset object.

        The dataset is loaded via the expected schema as defined by module constant
        TIMDEX_DATASET_SCHEMA.  If the target dataset differs in any way, errors may be
        raised when reading or writing data.
        """
        start_time = time.perf_counter()
        dataset = ds.dataset(
            self.source,
            schema=self.schema,
            format="parquet",
            partitioning="hive",
            filesystem=self.filesystem,
        )
        logger.info(
            f"Dataset successfully loaded: '{self.location}', "
            f"{round(time.perf_counter()-start_time, 2)}s"
        )
        return dataset

    def reload(self) -> None:
        """Reload dataset.

        After a write has been performed, a reload of the dataset is required to reflect
        any newly added records.
        """
        self.dataset = self.load_dataset()

    @property
    def row_count(self) -> int:
        """Get row count from loaded dataset."""
        if not self.dataset:
            raise DatasetNotLoadedError
        return self.dataset.count_rows()

    def write(
        self,
        records_iter: Iterator["DatasetRecord"],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_threads: bool = True,
    ) -> list[ds.WrittenFile]:
        """Write records to the TIMDEX parquet dataset.

        This method expects an iterator of DatasetRecord instances.

        This method encapsulates all dataset writing mechanics and performance
        optimizations (e.g. batching) so that the calling context can focus on yielding
        data.

        For write, the configuration existing_data_behavior="delete_matching" is used.
        This means that during write, if any pre-existing files are found for the exact
        combinations of partitions for that batch, those pre-existing files will be
        deleted.  This effectively makes a write idempotent to the TIMDEX dataset.

        A max_open_files=500 configuration is set to avoid AWS S3 503 error "SLOW_DOWN"
        if too many PutObject calls are made in parallel.  Testing suggests this does not
        substantially slow down the overall write.

        Args:
            - records_iter: Iterator of DatasetRecord instances
            - batch_size: size for batches to yield and write, directly affecting row
                group size in final parquet files
            - use_threads: boolean if threads should be used for writing
        """
        start_time = time.perf_counter()
        self._written_files = []

        if isinstance(self.source, list):
            raise TypeError(
                "Dataset location must be the root of a single dataset for writing"
            )

        record_batches_iter = self.get_dataset_record_batches(
            records_iter,
            batch_size=batch_size,
        )

        ds.write_dataset(
            record_batches_iter,
            base_dir=self.source,
            basename_template="%s-{i}.parquet" % (str(uuid.uuid4())),  # noqa: UP031
            existing_data_behavior="delete_matching",
            filesystem=self.filesystem,
            file_visitor=lambda written_file: self._written_files.append(written_file),  # type: ignore[arg-type]
            format="parquet",
            max_open_files=500,
            max_rows_per_file=MAX_ROWS_PER_FILE,
            max_rows_per_group=MAX_ROWS_PER_GROUP,
            partitioning=self.partition_columns,
            partitioning_flavor="hive",
            schema=self.schema,
            use_threads=use_threads,
        )

        self.log_write_statistics(start_time)
        return self._written_files  # type: ignore[return-value]

    def get_dataset_record_batches(
        self,
        records_iter: Iterator["DatasetRecord"],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Iterator[pa.RecordBatch]:
        """Yield pyarrow.RecordBatches for writing.

        This method expects an iterator of DatasetRecord instances.

        Each DatasetRecord is validated and serialized to a dictionary before added to a
        pyarrow.RecordBatch for writing.

        Args:
            - records_iter: Iterator of DatasetRecord instances
            - batch_size: size for batches to yield and write, directly affecting row
                group size in final parquet files
        """
        for i, record_batch in enumerate(itertools.batched(records_iter, batch_size)):
            batch_start_time = time.perf_counter()
            batch = pa.RecordBatch.from_pylist(
                [record.to_dict() for record in record_batch]
            )
            logger.debug(
                f"Batch {i + 1} yielded for writing, "
                f"elapsed: {round(time.perf_counter()-batch_start_time, 6)}s"
            )
            yield batch

    def log_write_statistics(self, start_time: float) -> None:
        """Parse written files from write and log statistics."""
        total_time = round(time.perf_counter() - start_time, 2)
        total_files = len(self._written_files)
        total_rows = sum(
            [
                wf.metadata.num_rows  # type: ignore[attr-defined]
                for wf in self._written_files
            ]
        )
        total_size = sum(
            [wf.size for wf in self._written_files]  # type: ignore[attr-defined]
        )
        logger.info(
            f"Dataset write complete - elapsed: "
            f"{total_time}s, "
            f"total files: {total_files}, "
            f"total rows: {total_rows}, "
            f"total size: {total_size}"
        )
