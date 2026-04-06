"""timdex_dataset_api/data_source.py

Abstract base class for TIMDEX data sources (records, embeddings, etc.).

Shared read/write orchestration lives here; subclasses provide schema definitions,
column contracts, and domain-specific hooks.
"""

import itertools
import logging
import time
import uuid
from abc import ABC
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from timdex_dataset_api.metadata import (
    CurrentMetadataViewSpec,
    DataTypeMetadataConfig,
    TIMDEXDatasetMetadata,
)

if TYPE_CHECKING:
    from timdex_dataset_api.dataset import TIMDEXDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidTable:
    """A table or view that a data source exposes for reading."""

    name: str
    """DuckDB table or view name, e.g. 'current_records'."""

    description: str
    """Human-readable explanation of what this table contains."""


@runtime_checkable
class DataSourceRow(Protocol):
    """Protocol for row objects that can be written to a data source."""

    def to_dict(self) -> dict: ...


class TIMDEXDataSource(ABC):
    """Abstract base class for TIMDEX data sources.

    Provides shared write, read, and column-contract logic.  Subclasses must
    define their schema and column contract class variables; metadata
    configuration is derived in ``__init_subclass__``.
    """

    # ------------------------------------------------------------------ #
    # Required sub-class class vars
    # ------------------------------------------------------------------ #

    # Short identifier, e.g. "records", "embeddings", etc.
    NAME: ClassVar[str]

    # Full pyarrow schema for parquet files of this data source
    SCHEMA: ClassVar[pa.Schema]

    # Heavy/payload columns read from parquet data files
    DATA_COLUMNS: ClassVar[list[str]]

    # Relative dataset path segment, e.g. "data/records"
    DATA_PATH_SEGMENT: ClassVar[str]

    # If True, metadata views are pre-joined to records for base columns
    PREJOIN_RECORDS: ClassVar[bool] = True

    # Metadata-layer configuration derived in ``__init_subclass__``
    METADATA_CONFIG: ClassVar[DataTypeMetadataConfig]

    # Tables and views this data source exposes for reading
    VALID_TABLES: ClassVar[list[ValidTable]]

    # ------------------------------------------------------------------ #
    # Optional sub-class class vars
    # ------------------------------------------------------------------ #

    # Hive-style partition columns (e.g. ``["year", "month", "day"]``)
    PARTITION_COLUMNS: ClassVar[list[str]] = [
        "year",
        "month",
        "day",
    ]

    # Current-metadata view specs owned by this data source
    CURRENT_VIEW_SPECS: ClassVar[list[CurrentMetadataViewSpec]] = []

    # Composite key columns used when joining metadata to parquet data.
    # filename is always included to physically disambiguate rows that share
    # the same logical key but reside in different parquet files (common for
    # bolt-on data sources like embeddings).
    JOIN_KEYS: ClassVar[list[str]] = [
        "timdex_record_id",
        "run_id",
        "run_record_offset",
        "filename",
    ]

    # ------------------------------------------------------------------ #
    # Derived class vars
    # ------------------------------------------------------------------ #

    ADDITIONAL_METADATA_COLUMNS: ClassVar[list[str]]
    DEFAULT_READ_COLUMNS: ClassVar[list[str]]
    VALID_READ_COLUMNS: ClassVar[set[str]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Instantiate DataSource subclasses."""
        super().__init_subclass__(**kwargs)

        # skip derivation for classes that haven't yet declared required contract vars
        required_class_vars = [
            "NAME",
            "SCHEMA",
            "PARTITION_COLUMNS",
            "DATA_COLUMNS",
            "DATA_PATH_SEGMENT",
            "VALID_TABLES",
        ]
        if not all(hasattr(cls, var_name) for var_name in required_class_vars):
            return

        cls.ADDITIONAL_METADATA_COLUMNS = cls.derive_additional_metadata_columns(
            cls.SCHEMA.names,
            cls.DATA_COLUMNS,
            TIMDEXDatasetMetadata.BASE_METADATA_COLUMNS,
            cls.PARTITION_COLUMNS,
        )

        cls.DEFAULT_READ_COLUMNS = (
            TIMDEXDatasetMetadata.BASE_METADATA_COLUMNS
            + cls.ADDITIONAL_METADATA_COLUMNS
            + cls.DATA_COLUMNS
        )

        cls.VALID_READ_COLUMNS = set(cls.DEFAULT_READ_COLUMNS)

        cls.METADATA_CONFIG = DataTypeMetadataConfig(
            name=cls.NAME,
            metadata_columns=cls.derive_metadata_columns(
                base_metadata_columns=TIMDEXDatasetMetadata.BASE_METADATA_COLUMNS,
                additional_metadata_columns=cls.ADDITIONAL_METADATA_COLUMNS,
                prejoin_records_columns=TIMDEXDatasetMetadata.PREJOIN_RECORDS_COLUMNS,
                prejoin_records=cls.PREJOIN_RECORDS,
            ),
            data_path_segment=cls.DATA_PATH_SEGMENT,
            prejoin_records=cls.PREJOIN_RECORDS,
        )

    def __init__(self, timdex_dataset: "TIMDEXDataset") -> None:
        """Instance instantiation; runs after sub-class instantiation."""
        self.timdex_dataset = timdex_dataset
        self.schema = self.SCHEMA
        self.partition_columns = self.PARTITION_COLUMNS
        self._ensure_data_root_exists()

    @property
    def data_root(self) -> str:
        """Root path for this source's parquet data."""
        return (
            f"{self.timdex_dataset.location.removesuffix('/')}"
            f"/{self.METADATA_CONFIG.data_path_segment}"
        )

    @property
    def default_table(self) -> str:
        """Default table name for read methods."""
        return self.NAME

    @staticmethod
    def derive_additional_metadata_columns(
        schema_names: list[str],
        data_columns: list[str],
        base_metadata_columns: list[str],
        partition_columns: list[str],
    ) -> list[str]:
        """Return additional metadata columns for a data source read contract.

        Derives columns from a physical parquet schema by excluding:
        - payload/data columns
        - shared/base metadata columns
        - partition helper columns
        """
        return [
            column_name
            for column_name in schema_names
            if column_name not in data_columns
            and column_name not in base_metadata_columns
            and column_name not in partition_columns
        ]

    @staticmethod
    def derive_metadata_columns(
        base_metadata_columns: list[str],
        additional_metadata_columns: list[str],
        prejoin_records_columns: list[str],
        *,
        prejoin_records: bool,
    ) -> list[str]:
        """Return metadata columns stored in static/delta metadata tables."""
        if not prejoin_records:
            return base_metadata_columns + additional_metadata_columns

        key_columns = [
            column_name
            for column_name in base_metadata_columns
            if column_name not in prejoin_records_columns and column_name != "filename"
        ]
        return key_columns + additional_metadata_columns + ["filename"]

    def create_data_structure(self) -> None:
        """Ensure source data root exists (idempotent for local datasets)."""
        self._ensure_data_root_exists()

    def _ensure_data_root_exists(self) -> None:
        """Ensure local data root directory exists for this source."""
        if self.timdex_dataset.location_scheme != "file":
            return
        Path(self.data_root).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Write pipeline
    # ------------------------------------------------------------------ #

    def write(
        self,
        rows_iter: Iterator[DataSourceRow],
        *,
        use_threads: bool = True,
        write_append_deltas: bool = True,
    ) -> list[ds.WrittenFile]:
        """Write rows to this data source's parquet dataset.

        Args:
            rows_iter: iterator of row objects (DatasetRecord, DatasetEmbedding, etc.)
                each must implement ``.to_dict()``
            use_threads: use threads for writing
            write_append_deltas: write append deltas for metadata tracking
        """
        start_time = time.perf_counter()
        written_files: list[ds.WrittenFile] = []

        filesystem, path = self.timdex_dataset.parse_location(self.data_root)

        batches_iter = self._create_batches(rows_iter)
        ds.write_dataset(
            batches_iter,
            base_dir=path,
            basename_template="%s-{i}.parquet" % (str(uuid.uuid4())),  # noqa: UP031
            existing_data_behavior="overwrite_or_ignore",
            filesystem=filesystem,
            file_visitor=lambda written_file: written_files.append(written_file),  # type: ignore[arg-type] # noqa: PLW0108
            format="parquet",
            max_open_files=500,
            max_rows_per_file=self.timdex_dataset.config.max_rows_per_file,
            max_rows_per_group=self.timdex_dataset.config.max_rows_per_group,
            partitioning=self.partition_columns,
            partitioning_flavor="hive",
            schema=self.schema,
            use_threads=use_threads,
        )

        # write metadata append deltas
        if write_append_deltas:
            for written_file in written_files:
                self.timdex_dataset.metadata.write_append_delta(
                    written_file.path,  # type: ignore[attr-defined]
                    self.METADATA_CONFIG,
                )
            self.timdex_dataset.refresh()

        self.log_write_statistics(start_time, written_files)

        return written_files

    def _create_batches(
        self,
        rows_iter: Iterator[DataSourceRow],
    ) -> Iterator[pa.RecordBatch]:
        """Yield ``pyarrow.RecordBatch`` objects from an iterator of row objects."""
        for i, batch in enumerate(
            itertools.batched(rows_iter, self.timdex_dataset.config.write_batch_size)
        ):
            row_dicts = [row.to_dict() for row in batch]
            record_batch = pa.RecordBatch.from_pylist(row_dicts)
            logger.debug(f"Yielding batch {i + 1} for dataset writing.")
            yield record_batch

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

    # ------------------------------------------------------------------ #
    # Read pipeline
    # ------------------------------------------------------------------ #

    def read_batches_iter(
        self,
        table: str | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Any,  # noqa: ANN401
    ) -> Iterator[pa.RecordBatch]:
        """Yield rows as ``pyarrow.RecordBatch`` via metadata-driven two-step reads.

        Args:
            table: DuckDB table/view name (defaults to ``self.default_table``)
            columns: columns to return (defaults to ``DEFAULT_READ_COLUMNS``)
            limit: max rows to yield
            where: raw SQL WHERE predicate
            **filters: key/value filter pairs
        """
        start_time = time.perf_counter()
        table = table or self.default_table

        valid_table_names = {vt.name for vt in self.VALID_TABLES}
        if table not in valid_table_names:
            valid = ", ".join(
                f"'{vt.name}' ({vt.description})" for vt in self.VALID_TABLES
            )
            raise ValueError(f"Invalid table: '{table}'. Valid tables: {valid}")

        try:
            self.timdex_dataset.get_sa_table("metadata", table)
        except ValueError as exc:
            raise ValueError(
                f"Table '{table}' not found in DuckDB context.  If this is a new "
                f"dataset, either {self.NAME} do not yet exist or a "
                "TIMDEXDataset.metadata.rebuild_dataset_metadata() may be required."
            ) from exc

        temp_table_name = "read_meta_chunk"
        total_yield_count = 0
        metadata_columns = self.timdex_dataset.metadata.get_metadata_columns_for_table(
            table
        )

        meta_chunks = self._iter_meta_chunks(
            table,
            limit=limit,
            where=where,
            **filters,
        )
        for i, meta_chunk_df in enumerate(meta_chunks):
            batch_time = time.perf_counter()
            batch_yield_count = len(meta_chunk_df)
            total_yield_count += batch_yield_count

            self.timdex_dataset.conn.register(
                temp_table_name,
                meta_chunk_df[metadata_columns],
            )

            try:
                data_query = self._build_data_query_for_chunk(
                    columns,
                    meta_chunk_df,
                    registered_metadata_chunk=temp_table_name,
                )
                yield from self._iter_data_chunks(data_query)
            finally:
                self.timdex_dataset.conn.unregister(temp_table_name)

            batch_rps = int(batch_yield_count / (time.perf_counter() - batch_time))
            logger.debug(
                f"read_batches_iter batch {i + 1}, "
                f"yielded: {batch_yield_count} "
                f"@ {batch_rps} records/second, "
                f"total yielded: {total_yield_count}"
            )

        logger.debug(
            f"read_batches_iter() elapsed: {round(time.perf_counter() - start_time, 2)}s"
        )

    def _iter_meta_chunks(
        self,
        table: str | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Any,  # noqa: ANN401
    ) -> Iterator[pd.DataFrame]:
        """Yield pandas DataFrames of metadata query results via keyset pagination."""
        table = table or self.default_table
        chunk_size = self.timdex_dataset.config.duckdb_join_batch_size

        keyset_value = (0, 0, 0)

        total_yielded = 0
        while True:
            if limit is not None:
                remaining = limit - total_yielded
                if remaining <= 0:
                    break
                chunk_limit = min(chunk_size, remaining)
            else:
                chunk_limit = chunk_size

            meta_query = (
                self.timdex_dataset.metadata.build_keyset_paginated_metadata_query(
                    table,
                    limit=chunk_limit,
                    where=where,
                    keyset_value=keyset_value,
                    **filters,
                )
            )
            meta_chunk_df = self.timdex_dataset.conn.query(meta_query).to_df()

            meta_chunk_count = len(meta_chunk_df)

            if meta_chunk_count == 0:
                break

            total_yielded += meta_chunk_count
            yield meta_chunk_df

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
        """Build SQL query for data retrieval, joining metadata chunk to parquet."""
        metadata_columns = (
            TIMDEXDatasetMetadata.BASE_METADATA_COLUMNS + self.ADDITIONAL_METADATA_COLUMNS
        )

        requested_columns = columns or self.DEFAULT_READ_COLUMNS
        invalid_columns = set(requested_columns) - self.VALID_READ_COLUMNS
        if invalid_columns:
            invalid = ", ".join(sorted(invalid_columns))
            raise ValueError(f"Invalid column: {invalid}")

        select_parts: list[str] = []
        for column_name in requested_columns:
            if column_name in metadata_columns:
                select_parts.append(f"mc.{column_name}")
                continue
            if column_name in self.DATA_COLUMNS:
                select_parts.append(f"ds.{column_name}")

        select_cols = ",".join(select_parts)

        filenames = list(meta_chunk_df["filename"].unique())
        if self.timdex_dataset.location_scheme == "s3":
            filenames = [
                f"s3://{f.removeprefix('s3://')}"
                for f in filenames  # type: ignore[union-attr]
            ]
        parquet_list_sql = "[" + ",".join(f"'{f}'" for f in filenames) + "]"

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

        join_keys = ", ".join(self.JOIN_KEYS)

        return f"""
            select
                {select_cols}
            from read_parquet(
                {parquet_list_sql},
                hive_partitioning=true,
                filename=true
            ) as ds
            inner join {registered_metadata_chunk} mc using (
                {join_keys}
            )
            where true
            {rro_clause};
            """

    def _iter_data_chunks(self, data_query: str) -> Iterator[pa.RecordBatch]:
        """Execute data query and stream ``pyarrow.RecordBatch`` results."""
        if self.timdex_dataset.location_scheme == "s3":
            self.timdex_dataset.conn.execute("""set threads=16;""")
        try:
            cursor = self.timdex_dataset.conn.execute(data_query)
            yield from cursor.to_arrow_reader(
                batch_size=self.timdex_dataset.config.read_batch_size
            )
        finally:
            if self.timdex_dataset.location_scheme == "s3":
                self.timdex_dataset.conn.execute(
                    f"""set threads={self.timdex_dataset.conn_factory.threads};"""
                )

    def read_dataframes_iter(
        self,
        table: str | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Any,  # noqa: ANN401
    ) -> Iterator[pd.DataFrame]:
        """Yield rows as pandas DataFrames."""
        for record_batch in self.read_batches_iter(
            table=table or self.default_table,
            columns=columns,
            limit=limit,
            where=where,
            **filters,
        ):
            yield record_batch.to_pandas()

    def read_dataframe(
        self,
        table: str | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Any,  # noqa: ANN401
    ) -> pd.DataFrame | None:
        """Read all matching rows into a single pandas DataFrame."""
        df_batches = [
            record_batch.to_pandas()
            for record_batch in self.read_batches_iter(
                table=table or self.default_table,
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
        table: str | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Any,  # noqa: ANN401
    ) -> Iterator[dict]:
        """Yield rows as Python dicts."""
        for record_batch in self.read_batches_iter(
            table=table or self.default_table,
            columns=columns,
            limit=limit,
            where=where,
            **filters,
        ):
            yield from record_batch.to_pylist()
