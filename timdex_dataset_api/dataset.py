"""timdex_dataset_api/dataset.py"""

import itertools
import operator
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime
from functools import reduce
from typing import TYPE_CHECKING, TypedDict, Unpack

import boto3
import pyarrow as pa
import pyarrow.compute as pc
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


class DatasetFilters(TypedDict, total=False):
    timdex_record_id: str | None
    source: str | None
    run_date: str | date | None
    run_type: str | None
    run_id: str | None
    action: str | None
    year: str | None
    month: str | None
    day: str | None


DEFAULT_BATCH_SIZE = 1_000
MAX_ROWS_PER_GROUP = DEFAULT_BATCH_SIZE
MAX_ROWS_PER_FILE = 100_000


def strict_date_parse(date_string: str) -> date:
    return datetime.strptime(date_string, "%Y-%m-%d").astimezone(UTC).date()


class TIMDEXDataset:

    def __init__(self, location: str | list[str]):
        """Initialize TIMDEXDataset object.

        Args:
            location (str | list[str]): Local filesystem path or an S3 URI to
                a parquet dataset. For partitioned datasets, set to the base directory.
        """
        self.location = location
        self.filesystem, self.source = self.parse_location(self.location)
        self.dataset: ds.Dataset = None  # type: ignore[assignment]
        self.schema = TIMDEX_DATASET_SCHEMA
        self.partition_columns = TIMDEX_DATASET_PARTITION_COLUMNS
        self._written_files: list[ds.WrittenFile] = None  # type: ignore[assignment]

    @property
    def row_count(self) -> int:
        """Get row count from loaded dataset."""
        if not self.dataset:
            raise DatasetNotLoadedError
        return self.dataset.count_rows()

    def load(
        self,
        **filters: Unpack[DatasetFilters],
    ) -> None:
        """Lazy load a pyarrow.dataset.Dataset and set to self.dataset.

        Loading is comprised of two main steps:

        - load: Lazily load full dataset. PyArrow will "discover" full dataset.
            Note: This step may take a couple of seconds but leans on PyArrow's
            parquet reading processes.
        - filter: Lazily filter rows in the PyArrow dataset by conditions on
            TIMDEX_DATASET_FILTER_COLUMNS.

        The dataset is loaded via the expected schema as defined by module constant
        TIMDEX_DATASET_SCHEMA.  If the target dataset differs in any way, errors may be
        raised when reading or writing data.

        Args:
            - filters: kwargs typed via DatasetFilters TypedDict
                - Filters passed directly in method call, e.g. source="alma",
                 run_date="2024-12-20", etc., but are typed according to DatasetFilters.
        """
        start_time = time.perf_counter()

        # load dataset
        self.dataset = ds.dataset(
            self.source,
            schema=self.schema,
            format="parquet",
            partitioning="hive",
            filesystem=self.filesystem,
        )

        # filter dataset
        self.dataset = self._get_filtered_dataset(**filters)

        logger.info(
            f"Dataset successfully loaded: '{self.location}', "
            f"{round(time.perf_counter()-start_time, 2)}s"
        )

    def _get_filtered_dataset(
        self,
        **filters: Unpack[DatasetFilters],
    ) -> ds.Dataset:
        """Lazy filter self.dataset and return a new pyarrow Dataset object.

        This method will construct a single pyarrow.compute.Expression
        that is combined from individual equality comparison predicates
        using the provided filters.

        Args:
            - filters: kwargs typed via DatasetFilters TypedDict
                - Filters passed directly in method call, e.g. source="alma",
                 run_date="2024-12-20", etc., but are typed according to DatasetFilters.

        Raises:
            DatasetNotLoadedError: Raised if `self.dataset` is None.
                TIMDEXDataset.load must be called before any filter method calls.
            ValueError: Raised if provided 'run_date' is an invalid type or
                cannot be parsed.

        Returns:
            ds.Dataset: Original pyarrow.dataset.Dataset (if no filters applied)
                or new pyarrow.dataset.Dataset with applied filters.
        """
        if not self.dataset:
            raise DatasetNotLoadedError

        # if run_date provided, derive year, month, and day partition filters and set
        if filters.get("run_date"):
            filters.update(self._parse_date_filters(filters["run_date"]))

        # create filter expressions for element-wise equality comparisons
        expressions = []
        for field, value in filters.items():
            if value:
                expressions.append(pc.equal(pc.field(field), value))

        # if filter expressions not found, return original dataset
        if not expressions:
            return self.dataset

        # combine filter expressions as a single predicate
        combined_expressions = reduce(operator.and_, expressions)
        logger.debug(
            "Filtering dataset based on the following column-value pairs: "
            f"{combined_expressions}"
        )

        return self.dataset.filter(combined_expressions)

    def _parse_date_filters(self, run_date: str | date | None) -> DatasetFilters:
        """Parse date filters from 'run_date'.

        Args:
            run_date (str | date | None): If str, the value must match the
                date format "%Y-%m-%d"; if date, ymd values are extracted
                as str.

        Raises:
            TypeError: Raised when 'run_date' is an invalid type.
            ValueError: Raised when either a datetime.date object cannot be parsed
                from a provided 'run_date' str.

        Returns:
            DatasetFilters[dict]: values for run_date, year, month, and day
        """
        if isinstance(run_date, str):
            run_date_obj = strict_date_parse(run_date)
        elif isinstance(run_date, date):
            run_date_obj = run_date
        else:
            raise TypeError(
                "Provided 'run_date' value must be a string matching format "
                "'%Y-%m-%d' or a datetime.date."
            )

        return {
            "run_date": run_date_obj,
            "year": run_date_obj.strftime("%Y"),
            "month": run_date_obj.strftime("%m"),
            "day": run_date_obj.strftime("%d"),
        }

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

        This method uses the configuration existing_data_behavior="overwrite_or_ignore",
        which will ignore any existing data and will overwrite files with the same name
        as the parquet file. Since a UUID is generated for each write via the
        basename_template, this effectively makes a write idempotent to the
        TIMDEX dataset.

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
            existing_data_behavior="overwrite_or_ignore",
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
