"""timdex_dataset_api/dataset.py"""

import itertools
import json
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack
from urllib.parse import urlparse

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from duckdb import DuckDBPyConnection
from pyarrow import fs

from timdex_dataset_api.config import configure_logger
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
    timdex_record_id: str | list[str] | None
    source: str | list[str] | None
    run_date: str | date | list[str | date] | None
    run_type: str | list[str] | None
    action: str | list[str] | None
    run_id: str | list[str] | None
    run_record_offset: int | list[int] | None
    run_timestamp: str | datetime | list[str | datetime] | None


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
    duckdb_join_batch_size: int = field(
        default_factory=lambda: int(os.getenv("TDA_DUCKDB_JOIN_BATCH_SIZE", "100_000"))
    )


class TIMDEXDataset:

    def __init__(
        self,
        location: str,
        *,
        config: TIMDEXDatasetConfig | None = None,
        preload_current_records: bool = False,
    ):
        """Initialize TIMDEXDataset object.

        Args:
            location: Local filesystem path or an S3 URI to a parquet dataset.
            config: Optional TIMDEXDatasetConfig instance.
            preload_current_records: if True, create in-memory temp table for
                current_records (faster for repeated queries); if False, create view only
                (default, lower memory)
        """
        self.config = config or TIMDEXDatasetConfig()
        self.location = location
        self.preload_current_records = preload_current_records

        self.create_data_structure()

        # pyarrow dataset
        self.schema = TIMDEX_DATASET_SCHEMA
        self.partition_columns = TIMDEX_DATASET_PARTITION_COLUMNS
        self.dataset = self.load_pyarrow_dataset()

        # dataset metadata
        self.metadata = TIMDEXDatasetMetadata(
            location,
            preload_current_records=preload_current_records,
        )

        # DuckDB context
        self.conn = self.setup_duckdb_context()

    @property
    def location_scheme(self) -> Literal["file", "s3"]:
        scheme = urlparse(self.location).scheme
        if scheme == "":
            return "file"
        if scheme == "s3":
            return "s3"
        raise ValueError(f"Location with scheme type '{scheme}' not supported.")

    @property
    def data_records_root(self) -> str:
        return f"{self.location.removesuffix('/')}/data/records"  # type: ignore[union-attr]

    def refresh(self) -> None:
        """Fully reload TIMDEXDataset instance."""
        self.__init__(  # type: ignore[misc]
            self.location,
            config=self.config,
            preload_current_records=self.preload_current_records,
        )

    def create_data_structure(self) -> None:
        """Ensure ETL records data structure exists in TIMDEX dataset."""
        if self.location_scheme == "file":
            Path(self.data_records_root).mkdir(
                parents=True,
                exist_ok=True,
            )

    def load_pyarrow_dataset(self, parquet_files: list[str] | None = None) -> ds.Dataset:
        """Lazy load a pyarrow.dataset.Dataset.

        The dataset is loaded via the expected schema as defined by module constant
        TIMDEX_DATASET_SCHEMA.  If the target dataset differs in any way, errors may be
        raised when reading or writing data.

        Args:
            parquet_files: explicit list of parquet files to construct pyarrow dataset
        """
        start_time = time.perf_counter()

        # get pyarrow filesystem and dataset path basesd on self.location
        filesystem, path = self.parse_location(self.data_records_root)

        # set source for pyarrow dataset
        source: str | list[str] = parquet_files or path

        dataset = ds.dataset(
            source,
            schema=self.schema,
            format="parquet",
            partitioning="hive",
            filesystem=filesystem,
        )

        logger.info(
            f"Dataset successfully loaded: '{self.data_records_root}', "
            f"{round(time.perf_counter()-start_time, 2)}s"
        )

        return dataset

    def parse_location(
        self,
        location: str,
    ) -> tuple[fs.FileSystem, str]:
        """Parse and return a pyarrow filesystem and normalized parquet path(s)."""
        if self.location_scheme == "s3":
            filesystem = TIMDEXDataset.get_s3_filesystem()
            source = location.removeprefix("s3://")
        else:
            filesystem = fs.LocalFileSystem()
            source = location
        return filesystem, source

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
            return fs.S3FileSystem(  # pragma: nocover
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

    def setup_duckdb_context(self) -> DuckDBPyConnection:
        """Create a DuckDB connection that metadata and data query and retrieval.

        This method extends TIMDEXDatasetMetadata's pre-existing DuckDB connection, adding
        a 'data' schema and any other configurations needed.
        """
        start_time = time.perf_counter()

        conn = self.metadata.conn

        # create data schema
        conn.execute("""create schema data;""")

        logger.debug(
            f"DuckDB data context created, {round(time.perf_counter()-start_time,2)}s"
        )
        return conn

    def write(
        self,
        records_iter: Iterator["DatasetRecord"],
        *,
        use_threads: bool = True,
        write_append_deltas: bool = True,
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
            - write_append_deltas: boolean if append deltas should be written for records
                written during write
        """
        start_time = time.perf_counter()
        written_files: list[ds.WrittenFile] = []

        filesystem, path = self.parse_location(self.data_records_root)

        # write ETL parquet records
        record_batches_iter = self.create_record_batches(records_iter)
        ds.write_dataset(
            record_batches_iter,
            base_dir=path,
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

        # refresh dataset files
        self.dataset = self.load_pyarrow_dataset()

        # write metadata append deltas
        if write_append_deltas:
            for written_file in written_files:
                self.metadata.write_append_delta_duckdb(written_file.path)  # type: ignore[attr-defined]
            self.metadata.refresh()

        self.log_write_statistics(start_time, written_files)

        return written_files

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
        table: str = "records",
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[DatasetFilters],
    ) -> Iterator[pa.RecordBatch]:
        """Yield ETL records as pyarrow.RecordBatches.

        This is the base read method.  All read methods eventually drop down and use this
        for streaming batches of records.  This method performs a two-step process:

            1. Perform a "metadata" query that narrows down records and physical parquet
            files to read from.
            2. Perform a "data" query that retrieves actual rows, joining the metadata
            information to increase efficiency.

        More detail can be found here: docs/reading.md

        Args:
            - table: an available DuckDB view or table
            - columns: list of columns to return
            - limit: limit number of records yielded
            - where: raw SQL WHERE clause that can be used alone, or in combination with
            key/value DatasetFilters
            - filters: simple filtering based on key/value pairs from DatasetFilters
        """
        start_time = time.perf_counter()

        temp_table_name = "read_meta_chunk"
        total_yield_count = 0

        for i, meta_chunk_df in enumerate(
            self._iter_meta_chunks(
                table,
                limit=limit,
                where=where,
                **filters,
            )
        ):
            batch_time = time.perf_counter()
            batch_yield_count = len(meta_chunk_df)
            total_yield_count += batch_yield_count

            self.conn.register(
                temp_table_name,
                meta_chunk_df[
                    [
                        "timdex_record_id",
                        "run_id",
                        "run_record_offset",
                    ]
                ],
            )

            # build and perform data query, yield records
            # set in try/finally block to ensure we always deregister the meta table
            try:
                data_query = self._build_data_query_for_chunk(
                    columns,
                    meta_chunk_df,
                    registered_metadata_chunk=temp_table_name,
                )
                yield from self._iter_data_chunks(data_query)
            finally:
                self.conn.unregister(temp_table_name)

            batch_rps = int(batch_yield_count / (time.perf_counter() - batch_time))
            logger.debug(
                f"read_batches_iter batch {i+1}, yielded: {batch_yield_count} "
                f"@ {batch_rps} records/second, total yielded: {total_yield_count}"
            )

        logger.debug(
            f"read_batches_iter() elapsed: {round(time.perf_counter()-start_time, 2)}s"
        )

    def _iter_meta_chunks(
        self,
        table: str = "records",
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[DatasetFilters],
    ) -> Iterator[pd.DataFrame]:
        """Utility method to yield pandas Dataframe chunks of metadata query results.

        The approach here is to use "keyset" pagination, which means each paged result
        is a greater-than (>) check against a tuple of ordered values from the previous
        chunk.  This is more performant than a LIMIT + OFFSET.
        """
        # use duckdb_join_batch_size as the chunk size for keyset pagination
        chunk_size = self.config.duckdb_join_batch_size

        # init keyset value of zeros to begin with
        keyset_value = (0, 0, 0)

        total_yielded = 0
        while True:

            # enforce limit if passed
            if limit is not None:
                remaining = limit - total_yielded
                if remaining <= 0:
                    break
                chunk_limit = min(chunk_size, remaining)
            else:
                chunk_limit = chunk_size

            # perform chunk query and convert to pyarrow Table
            meta_query = self.metadata.build_keyset_paginated_metadata_query(
                table,
                limit=chunk_limit,  # pass chunk_limit instead of limit
                where=where,
                keyset_value=keyset_value,
                **filters,
            )
            meta_chunk_df = self.metadata.conn.query(meta_query).to_df()

            meta_chunk_count = len(meta_chunk_df)

            # an empty chunk signals end of pagination
            if meta_chunk_count == 0:
                break

            # yield this chunk of data
            total_yielded += meta_chunk_count
            yield meta_chunk_df[
                [
                    "timdex_record_id",
                    "run_id",
                    "run_record_offset",
                    "filename",
                ]
            ]

            # update keyset value using the last row from this chunk
            last_row = meta_chunk_df.iloc[-1]
            keyset_value = (
                int(last_row.filename_hash),
                int(last_row.run_id_hash),
                int(last_row.run_record_offset),
            )

    def _build_data_query_for_chunk(
        self,
        columns: list[str] | None,
        meta_chunk_df: pd.DataFrame,
        registered_metadata_chunk: str = "meta_chunk",
    ) -> str:
        """Build SQL query used for data retrieval, joining on passed metadata data."""
        # build select columns
        select_cols = ",".join(
            [f"ds.{col}" for col in (columns or TIMDEX_DATASET_SCHEMA.names)]
        )

        # build list of explicit parquet files to read from
        filenames = list(meta_chunk_df["filename"].unique())
        if self.location_scheme == "s3":
            filenames = [
                f"s3://{f.removeprefix('s3://')}" for f in filenames  # type: ignore[union-attr]
            ]
        parquet_list_sql = "[" + ",".join(f"'{f}'" for f in filenames) + "]"

        # build run_record_offset WHERE clause to leverage row group pruning
        rro_values = meta_chunk_df["run_record_offset"].unique()
        rro_values.sort()
        if len(rro_values) <= 1_000:  # noqa: PLR2004
            rro_clause = (
                f"and run_record_offset in ({','.join(str(rro) for rro in rro_values)})"
            )
        else:
            rro_clause = (
                f"and run_record_offset between {rro_values[0]} and {rro_values[-1]}"
            )

        return f"""
            select
                {select_cols}
            from read_parquet(
                {parquet_list_sql},
                hive_partitioning=true,
                filename=true
            ) as ds
            inner join {registered_metadata_chunk} mc using (
                timdex_record_id, run_id, run_record_offset
            )
            where true
            {rro_clause};
            """

    def _iter_data_chunks(self, data_query: str) -> Iterator[pa.RecordBatch]:
        """Perform a query to retrieve data and stream chunks."""
        if self.location_scheme == "s3":
            self.conn.execute("""set threads=16;""")
        try:
            cursor = self.conn.execute(data_query)
            yield from cursor.fetch_record_batch(
                rows_per_batch=self.config.read_batch_size
            )
        finally:
            if self.location_scheme == "s3":
                self.conn.execute(
                    f"""set threads={self.metadata.config.duckdb_connection_threads};"""
                )

    def read_dataframes_iter(
        self,
        table: str = "records",
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[DatasetFilters],
    ) -> Iterator[pd.DataFrame]:
        for record_batch in self.read_batches_iter(
            table=table,
            columns=columns,
            limit=limit,
            where=where,
            **filters,
        ):
            yield record_batch.to_pandas()

    def read_dataframe(
        self,
        table: str = "records",
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[DatasetFilters],
    ) -> pd.DataFrame | None:
        df_batches = [
            record_batch.to_pandas()
            for record_batch in self.read_batches_iter(
                table=table,
                columns=columns,
                limit=limit,
                where=where,
                **filters,
            )
        ]
        if not df_batches:
            return None
        return pd.concat(df_batches)

    def read_dicts_iter(
        self,
        table: str = "records",
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[DatasetFilters],
    ) -> Iterator[dict]:
        for record_batch in self.read_batches_iter(
            table=table,
            columns=columns,
            limit=limit,
            where=where,
            **filters,
        ):
            yield from record_batch.to_pylist()

    def read_transformed_records_iter(
        self,
        table: str = "records",
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[DatasetFilters],
    ) -> Iterator[dict]:
        for record_dict in self.read_dicts_iter(
            table=table,
            columns=["transformed_record"],
            limit=limit,
            where=where,
            **filters,
        ):
            if transformed_record := record_dict["transformed_record"]:
                yield json.loads(transformed_record)
