"""timdex_dataset_api/dataset.py"""

import itertools
import json
import operator
import os
from fileinput import filename
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import reduce
from typing import TYPE_CHECKING, TypedDict, Unpack

import boto3
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import fs

from timdex_dataset_api.config import configure_logger
from timdex_dataset_api.exceptions import DatasetNotLoadedError
from timdex_dataset_api.metadata import TIMDEXDatasetMetadata

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
    duckdb_join_batch_size: int = 100_000  # TODO: consider env var?


class TIMDEXDataset:

    def __init__(
        self,
        location: str,
        config: TIMDEXDatasetConfig | None = None,
    ):
        """Initialize TIMDEXDataset object.

        Args:
            location (str): Local filesystem path or an S3 URI to
                ETL dataset root.
        """
        self.config = config or TIMDEXDatasetConfig()
        self.location = location

        # pyarrow dataset
        self.filesystem, self.paths = self.parse_location(self.records_location)
        self.dataset: ds.Dataset = None  # type: ignore[assignment]
        self.schema = TIMDEX_DATASET_SCHEMA
        self.partition_columns = TIMDEX_DATASET_PARTITION_COLUMNS

        self.metadata = TIMDEXDatasetMetadata(self.location)

        # reading
        self._current_records: bool = False

    @classmethod
    def parse_location(cls, location: str) -> tuple[fs.FileSystem, str]:
        """Parse and return the filesystem and normalized location."""
        location_parts = urlparse(location)
        if location_parts.scheme == "s3":
            filesystem = TIMDEXDataset._get_s3_filesystem()
            path = str(
                Path(location_parts.netloc) / Path(location_parts.path.removeprefix("/"))
            )
        else:
            filesystem = fs.LocalFileSystem()
            path = location
        return filesystem, path

    @property
    def records_location(self):
        return f"{self.location}/data/records"

    @staticmethod
    def _get_s3_filesystem() -> fs.FileSystem:
        """Instantiate a pyarrow S3 Filesystem for dataset loading.

        If the env var 'MINIO_S3_ENDPOINT_URL' is present, assume a local MinIO S3
        instance and configure accordingly, otherwise assume normal AWS S3.
        """
        if os.getenv("MINIO_S3_ENDPOINT_URL"):
            return fs.S3FileSystem(
                access_key=os.environ["MINIO_USERNAME"],
                secret_key=os.environ["MINIO_PASSWORD"],
                endpoint_override=os.environ["MINIO_S3_ENDPOINT_URL"],
            )

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

    def _load_pyarrow_dataset(self) -> ds.Dataset:
        """Load the pyarrow dataset per local filesystem and paths attributes."""
        return ds.dataset(
            self.paths,
            schema=self.schema,
            format="parquet",
            partitioning="hive",
            filesystem=self.filesystem,
        )

    def load(
        self,
        *,
        current_records: bool = False,
        include_parquet_files: list[str] | None = None,
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
            - current_records: bool
                - if True, all records yielded from this instance will be the current
                version of the record in the dataset.
        """
        start_time = time.perf_counter()

        self._current_records = current_records

        # TODO: revisit, but probably good logic to have
        # NOTE/BUG: what if we want current + explicit list?
        # limit to current parquet files
        if current_records:
            self.paths = self.metadata.get_current_parquet_files(**filters)
            # use explicit list of parquet files
        # NOTE: this opens the door for a metadata query to provide an explicit list...
        elif include_parquet_files:
            self.paths = include_parquet_files
        # reset paths from original location before load
        else:
            _, self.paths = self.parse_location(self.records_location)

        # load pyarrow dataset of records
        self.dataset = self._load_pyarrow_dataset()

        # filter dataset
        self.dataset = self._get_filtered_dataset(**filters)

        logger.info(
            f"Dataset successfully loaded: '{self.location}', "
            f"{round(time.perf_counter() - start_time, 2)}s"
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

    def write(
        self,
        records_iter: Iterator["DatasetRecord"],
        *,
        use_threads: bool = True,
        write_metadata: bool = True,
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
        written_files: list[ds.WrittenFile] = []

        filesystem, dataset_records_path = self.parse_location(self.records_location)

        record_batches_iter = self.create_record_batches(records_iter)

        ds.write_dataset(
            record_batches_iter,
            base_dir=dataset_records_path,
            basename_template="%s-{i}.parquet" % (str(uuid.uuid4())),  # noqa: UP031
            existing_data_behavior="overwrite_or_ignore",
            filesystem=filesystem,
            file_visitor=lambda written_file: written_files.append(written_file),  # type: ignore[arg-type]
            format="parquet",
            max_open_files=500,
            max_rows_per_file=self.config.max_rows_per_file,
            max_rows_per_group=self.config.max_rows_per_group,
            partitioning=self.partition_columns,
            partitioning_flavor="hive",
            schema=self.schema,
            use_threads=use_threads,
        )

        self.log_write_statistics(start_time, written_files)

        # TODO: temporary shim for development work, but maybe decent option?
        if write_metadata:
            # TODO: newly added, revisit implementation
            self.write_metadata_append_deltas(written_files)

            # TODO: also feels clunky
            self.metadata.refresh()

        return written_files

    # TODO: newly added, revisit implementation
    # QUESTION: should this be a responsibility of the metadata class?
    # NOTE: there should be a method dedicated to a *single* ETL parquet file
    #   this could be handy for writing deltas after an ETL run if missed
    def write_metadata_append_deltas(self, written_files: list):
        for written_file in written_files:
            filepath = written_file.path
            self.write_append_delta_for_etl_parquet_file(filepath)

    def write_append_delta_for_etl_parquet_file(self, filepath: str):
        logger.info(f"Writing append delta for new parquet file: {filepath}")
        filesystem, _ = self.parse_location(self.records_location)
        metadata_columns = [
            "timdex_record_id",
            "source",
            "run_date",
            "run_type",
            "action",
            "run_id",
            "run_record_offset",
            "run_timestamp",
        ]
        table = pq.read_table(
            filepath,
            filesystem=filesystem,
            columns=metadata_columns,
        )

        new_array = pa.array([filepath] * table.num_rows)
        table = table.append_column("filename", new_array)

        output_path = (
            f"{self.metadata.append_deltas_path.removeprefix('s3://')}/"
            f"append_delta-{filepath.split('/')[-1]}"
        )

        # TODO: works, but clunky
        components = urlparse(self.metadata.append_deltas_path)
        if components.scheme != "s3":
            if not os.path.exists(self.metadata.append_deltas_path):
                os.mkdir(self.metadata.append_deltas_path)

        pq.write_table(table, output_path, filesystem=filesystem)

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

    def log_write_statistics(
        self,
        start_time: float,
        written_files: list[ds.WrittenFile],
    ) -> None:
        """Parse written files from write and log statistics."""
        total_time = round(time.perf_counter() - start_time, 2)
        total_files = len(written_files)
        total_rows = sum(
            [wf.metadata.num_rows for wf in written_files]  # type: ignore[attr-defined]
        )
        total_size = sum([wf.size for wf in written_files])  # type: ignore[attr-defined]
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

        If the flag self._current_records is set, this method leans on
        self._yield_current_record_deduped_batches() to apply deduplication of records to
        ensure only current versions of the record are ever yielded.

        Args:
            - columns: list[str], list of columns to return from the dataset
            - filters: pairs of column:value to filter the dataset
        """
        if not self.dataset:
            raise DatasetNotLoadedError(
                "Dataset is not loaded. Please call the `load` method first."
            )
        dataset = self._get_filtered_dataset(**filters)

        # if current records, add required columns for deduplication
        if self._current_records:
            if not columns:
                columns = TIMDEX_DATASET_SCHEMA.names
            columns.extend(["timdex_record_id", "run_id"])
            columns = list(set(columns))

        batches = dataset.to_batches(
            columns=columns,
            batch_size=self.config.read_batch_size,
            batch_readahead=self.config.batch_read_ahead,
            fragment_readahead=self.config.fragment_read_ahead,
        )

        if self._current_records:
            yield from self._yield_current_record_batches(batches, **filters)
        else:
            for batch in batches:
                if len(batch) > 0:
                    yield batch

    def _yield_current_record_batches(
        self,
        batches: Iterator[pa.RecordBatch],
        **filters: Unpack[DatasetFilters],
    ) -> Iterator[pa.RecordBatch]:
        """Method to yield only the most recent version of each record.

        When multiple versions of a record (same timdex_record_id) exist in the dataset,
        this method ensures only the most recent version is returned.  If filtering is
        applied that removes this most recent version of a record, that timdex_record_id
        will not be yielded at all.

        This method uses TIMDEXDatasetMetadata to provide a mapping of timdex_record_id to
        run_id for the current ETL run for that record.  While yielding records, only when
        the timdex_record_id + run_id match the mapping is a record yielded.

        Args:
            - batches: batches of records to actually yield from
            - filters: pairs of column:value to filter the dataset metadata required
        """
        # get map of timdex_record_id to run_id for current version of that record
        record_to_run_map = self.metadata.get_current_record_to_run_map(**filters)

        # loop through batches, yielding only current records
        for batch in batches:

            if batch.num_rows == 0:
                continue

            to_yield_indices = []

            record_ids = batch.column("timdex_record_id").to_pylist()
            run_ids = batch.column("run_id").to_pylist()

            for i, (record_id, run_id) in enumerate(
                zip(
                    record_ids,
                    run_ids,
                    strict=True,
                )
            ):
                if record_to_run_map.get(record_id) == run_id:
                    to_yield_indices.append(i)

            if to_yield_indices:
                yield batch.take(pa.array(to_yield_indices))  # type: ignore[arg-type]

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

    # DEBUG ---------------------------------------------------
    # DEBUG: Experimental pure DuckDB for query + retrieval
    # DEBUG ---------------------------------------------------
    def read_sql_batches_iter(self, sql: str):
        conn = self.metadata.conn

        # set connection configs
        # TODO: set by self.config
        conn.execute(
            f"""
            set enable_external_file_cache = false;
            set memory_limit = '4GB';
            set threads = 4;
            set preserve_insertion_order=false;
            """
        )

        # prepare query
        sql = sql.replace(";", "")

        # add ordering
        # TODO: would need to remove other ordering if present...
        sql += " order by filename, run_record_offset"

        # get metadata results
        meta_df = conn.query(sql).to_df()

        # raise error if required columns not present
        required_columns = {"timdex_record_id", "run_id", "run_record_offset", "filename"}
        if set(meta_df.columns) <= required_columns:
            raise ValueError(f"Missing one or more required columns: {required_columns}")

        # loop through metadata chunks and perform data query
        total_time = time.perf_counter()
        total_yield_count = 0
        for chunk in range(0, len(meta_df), self.config.duckdb_join_batch_size):
            batch_time = time.perf_counter()
            batch_yield_count = 0
            chunk_df = meta_df[chunk : chunk + self.config.duckdb_join_batch_size]

            # register
            conn.register("metadata_chunk", chunk_df)

            # build data query
            file_list = chunk_df["filename"].unique()
            parquet_list_sql = (
                "["
                + ",".join(f"'s3://{f.removeprefix("s3://")}'" for f in file_list)
                + "]"
            )
            rro_list_sql = ",".join(
                str(rro) for rro in chunk_df["run_record_offset"].unique()
            )
            join_cols_sql = ",".join(["timdex_record_id", "run_id", "run_record_offset"])
            data_query = f"""
            select
                ds.*
            from read_parquet(
                {parquet_list_sql},
                hive_partitioning=true,
                filename=true
            ) as ds
            inner join metadata_chunk mc using ({join_cols_sql})
            where ds.run_record_offset in ({rro_list_sql})
            """

            # stream batch results
            for batch in conn.execute(data_query).fetch_record_batch(
                rows_per_batch=self.config.read_batch_size
            ):
                total_yield_count += len(batch)
                batch_yield_count += len(batch)
                yield batch
            conn.unregister("meta_chunk")

            batch_rps = round(batch_yield_count / (time.perf_counter() - batch_time), 3)
            logger.debug(
                f"DuckDB read - batch yielded: {batch_yield_count} "
                f"@ {batch_rps} records/second, total yielded: {total_yield_count}"
            )

    def read_sql_dataframe(self, sql: str) -> pd.DataFrame | None:
        df_batches = [
            record_batch.to_pandas() for record_batch in self.read_sql_batches_iter(sql)
        ]
        if not df_batches:
            return None
        return pd.concat(df_batches)
