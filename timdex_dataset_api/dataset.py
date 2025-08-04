"""timdex_dataset_api/dataset.py"""

import itertools
import json
import operator
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import reduce
from typing import TYPE_CHECKING, TypedDict, Unpack

import boto3
import pandas as pd
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
        pa.field("action", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("run_record_offset", pa.int32()),
        pa.field("year", pa.string()),
        pa.field("month", pa.string()),
        pa.field("day", pa.string()),
        pa.field("run_timestamp", pa.timestamp("us", tz="UTC")),
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
    action: str | None
    run_id: str | None
    run_record_offset: int | None
    year: str | None
    month: str | None
    day: str | None
    run_timestamp: str | datetime | None


@dataclass
class TIMDEXDatasetConfig:
    """Configurations for dataset operations.

    - read_batch_size: row size of batches read, affecting memory consumption
    - write_batch_size: row size of batches written, directly affecting row group size in
        final parquet files
    - max_rows_per_group: max number of rows per row group in a parquet file
    - max_rows_per_file: max number of rows in a single parquet file
    - batch_read_ahead: number of batches to optimistically read ahead when batch reading
        from a dataset; pyarrow default is 16
    - fragment_read_ahead: number of fragments to optimistically read ahead when batch
        reaching from a dataset; pyarrow default is 4
    """

    read_batch_size: int = field(
        default_factory=lambda: int(os.getenv("TDA_READ_BATCH_SIZE", "1_000"))
    )
    write_batch_size: int = field(
        default_factory=lambda: int(os.getenv("TDA_WRITE_BATCH_SIZE", "1_000"))
    )
    max_rows_per_group: int = field(
        default_factory=lambda: int(os.getenv("TDA_MAX_ROWS_PER_GROUP", "1_000"))
    )
    max_rows_per_file: int = field(
        default_factory=lambda: int(os.getenv("TDA_MAX_ROWS_PER_FILE", "100_000"))
    )
    batch_read_ahead: int = field(
        default_factory=lambda: int(os.getenv("TDA_BATCH_READ_AHEAD", "0"))
    )
    fragment_read_ahead: int = field(
        default_factory=lambda: int(os.getenv("TDA_FRAGMENT_READ_AHEAD", "0"))
    )


class TIMDEXDataset:

    def __init__(
        self,
        location: str | list[str],
        config: TIMDEXDatasetConfig | None = None,
    ):
        """Initialize TIMDEXDataset object.

        Args:
            location (str | list[str]): Local filesystem path or an S3 URI to
                a parquet dataset. For partitioned datasets, set to the base directory.
        """
        self.config = config or TIMDEXDatasetConfig()
        self.location = location

        # pyarrow dataset
        self.filesystem, self.paths = self.parse_location(self.location)
        self.dataset: ds.Dataset = None  # type: ignore[assignment]
        self.schema = TIMDEX_DATASET_SCHEMA
        self.partition_columns = TIMDEX_DATASET_PARTITION_COLUMNS

        # writing
        self._written_files: list[ds.WrittenFile] = None  # type: ignore[assignment]

    @property
    def data_records_root(self) -> str:
        return f"{self.location.removesuffix('/')}/data/records"  # type: ignore[union-attr]

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

        # reset paths from original location before load
        _, self.paths = self.parse_location(self.location)

        # perform initial load of full dataset
        self.dataset = self._load_pyarrow_dataset()

        # filter dataset
        self.dataset = self._get_filtered_dataset(**filters)

        logger.info(
            f"Dataset successfully loaded: '{self.location}', "
            f"{round(time.perf_counter()-start_time, 2)}s"
        )

    def _load_pyarrow_dataset(self) -> ds.Dataset:
        """Load the pyarrow dataset per local filesystem and paths attributes."""
        return ds.dataset(
            self.paths,
            schema=self.schema,
            format="parquet",
            partitioning="hive",
            filesystem=self.filesystem,
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
        for field, value in filters.items():  # noqa: F402
            if isinstance(value, list):
                expressions.append(ds.field(field).isin(value))
            else:
                expressions.append(ds.field(field) == value)

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
            run_date_obj = datetime.strptime(run_date, "%Y-%m-%d").astimezone(UTC).date()
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
        """Instantiate a pyarrow S3 Filesystem for dataset loading.

        If the env var 'MINIO_S3_ENDPOINT_URL' is present, assume a local MinIO S3
        instance and configure accordingly, otherwise assume normal AWS S3.
        """
        session = boto3.session.Session()
        credentials = session.get_credentials()
        if not credentials:
            raise RuntimeError("Could not locate AWS credentials")

        if os.getenv("MINIO_S3_ENDPOINT_URL"):
            return fs.S3FileSystem(
                access_key=os.environ["MINIO_USERNAME"],
                secret_key=os.environ["MINIO_PASSWORD"],
                endpoint_override=os.environ["MINIO_S3_ENDPOINT_URL"],
            )

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
            - use_threads: boolean if threads should be used for writing
        """
        start_time = time.perf_counter()
        self._written_files = []

        dataset_filesystem, dataset_path = self.parse_location(self.data_records_root)
        if isinstance(dataset_path, list):
            raise TypeError(
                "Dataset location must be the root of a single dataset for writing"
            )

        record_batches_iter = self.create_record_batches(records_iter)

        ds.write_dataset(
            record_batches_iter,
            base_dir=dataset_path,
            basename_template="%s-{i}.parquet" % (str(uuid.uuid4())),  # noqa: UP031
            existing_data_behavior="overwrite_or_ignore",
            filesystem=dataset_filesystem,
            file_visitor=lambda written_file: self._written_files.append(written_file),  # type: ignore[arg-type]
            format="parquet",
            max_open_files=500,
            max_rows_per_file=self.config.max_rows_per_file,
            max_rows_per_group=self.config.max_rows_per_group,
            partitioning=self.partition_columns,
            partitioning_flavor="hive",
            schema=self.schema,
            use_threads=use_threads,
        )

        self.log_write_statistics(start_time)
        return self._written_files  # type: ignore[return-value]

    def create_record_batches(
        self, records_iter: Iterator["DatasetRecord"]
    ) -> Iterator[pa.RecordBatch]:
        """Yield pyarrow.RecordBatches for writing.

        This method expects an iterator of DatasetRecord instances.

        Each DatasetRecord is serialized to a dictionary, any column data shared by all
        rows is added to the record, and then added to a pyarrow.RecordBatch for writing.

        Args:
            - records_iter: Iterator of DatasetRecord instances
        """
        for i, record_batch in enumerate(
            itertools.batched(records_iter, self.config.write_batch_size)
        ):
            record_dicts = [record.to_dict() for record in record_batch]
            batch = pa.RecordBatch.from_pylist(record_dicts)
            logger.debug(f"Yielding batch {i + 1} for dataset writing.")
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

    def read_batches_iter(
        self,
        columns: list[str] | None = None,
        **filters: Unpack[DatasetFilters],
    ) -> Iterator[pa.RecordBatch]:
        """Yield pyarrow.RecordBatches from the dataset.

        While batch_size will limit the max rows per batch, filtering may result in some
        batches having less than this limit.

        Args:
            - columns: list[str], list of columns to return from the dataset
            - filters: pairs of column:value to filter the dataset
        """
        if not self.dataset:
            raise DatasetNotLoadedError(
                "Dataset is not loaded. Please call the `load` method first."
            )
        dataset = self._get_filtered_dataset(**filters)

        batches = dataset.to_batches(
            columns=columns,
            batch_size=self.config.read_batch_size,
            batch_readahead=self.config.batch_read_ahead,
            fragment_readahead=self.config.fragment_read_ahead,
        )

        for batch in batches:
            if len(batch) > 0:
                yield batch

    def read_dataframes_iter(
        self,
        columns: list[str] | None = None,
        **filters: Unpack[DatasetFilters],
    ) -> Iterator[pd.DataFrame]:
        """Yield record batches as Pandas DataFrames from the dataset.

        Args: see self.read_batches_iter()
        """
        for record_batch in self.read_batches_iter(
            columns=columns,
            **filters,
        ):
            yield record_batch.to_pandas()

    def read_dataframe(
        self,
        columns: list[str] | None = None,
        **filters: Unpack[DatasetFilters],
    ) -> pd.DataFrame | None:
        """Yield record batches as Pandas DataFrames and concatenate to single dataframe.

        WARNING: this will pull all records from currently filtered dataset into memory.

        If no batches are found based on filtered dataset, None is returned.

        Args: see self.read_batches_iter()
        """
        df_batches = [
            record_batch.to_pandas()
            for record_batch in self.read_batches_iter(
                columns=columns,
                **filters,
            )
        ]
        if not df_batches:
            return None
        return pd.concat(df_batches)

    def read_dicts_iter(
        self,
        columns: list[str] | None = None,
        **filters: Unpack[DatasetFilters],
    ) -> Iterator[dict]:
        """Yield individual record rows as dictionaries from the dataset.

        Args: see self.read_batches_iter()
        """
        for record_batch in self.read_batches_iter(
            columns=columns,
            **filters,
        ):
            yield from record_batch.to_pylist()

    def read_transformed_records_iter(
        self,
        **filters: Unpack[DatasetFilters],
    ) -> Iterator[dict]:
        """Yield individual transformed records as dictionaries from the dataset.

        If 'transformed_record' is None (common scenarios are action="skip"|"error"), the
        yield statement will not be executed for the row.  Note that for action="delete" a
        transformed record still may be yielded if present.

        Args: see self.read_batches_iter()
        """
        for record_dict in self.read_dicts_iter(
            columns=["timdex_record_id", "transformed_record"],
            **filters,
        ):
            if transformed_record := record_dict["transformed_record"]:
                yield json.loads(transformed_record)
