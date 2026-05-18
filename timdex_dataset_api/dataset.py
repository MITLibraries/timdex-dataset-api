"""timdex_dataset_api/dataset.py"""

import os
import time
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

import boto3
from duckdb_engine import ConnectionWrapper
from pyarrow import fs
from sqlalchemy import MetaData, Table, create_engine

from timdex_dataset_api.config import configure_logger
from timdex_dataset_api.data_types import (
    TIMDEXEmbeddings,
    TIMDEXFulltexts,
    TIMDEXRecords,
)
from timdex_dataset_api.metadata import TIMDEXDatasetMetadata
from timdex_dataset_api.utils import DuckDBConnectionFactory

logger = configure_logger(__name__)


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
    - duckdb_join_batch_size: batch size for keyset pagination when joining metadata

    Note: DuckDB connection settings (memory_limit, threads) are handled by
    DuckDBConnectionFactory via TDA_DUCKDB_MEMORY_LIMIT and TDA_DUCKDB_THREADS env vars.
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
    """Class to represent the TIMDEXDataset."""

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

        # create DuckDB connection used by all classes
        self.conn_factory = DuckDBConnectionFactory(location_scheme=self.location_scheme)
        self.conn = self.conn_factory.create_connection()

        # create schemas
        self._create_duckdb_schemas()

        self.data_type_classes = [TIMDEXRecords, TIMDEXEmbeddings, TIMDEXFulltexts]

        # define readable metadata-backed tables contributed by data types
        self.table_configs = [
            table_config
            for data_type_class in self.data_type_classes
            for table_config in data_type_class.TABLES
        ]

        # composed components receive self
        self.metadata = TIMDEXDatasetMetadata(self)
        self.records = TIMDEXRecords(self)
        self.embeddings = TIMDEXEmbeddings(self)
        self.fulltexts = TIMDEXFulltexts(self)

        # SQLAlchemy (SA) reflection after components have set up their views
        self.sa_tables: dict[str, dict[str, Table]] = {}
        self.reflect_sa_tables()

    @property
    def location_scheme(self) -> Literal["file", "s3"]:
        scheme = urlparse(self.location).scheme
        if scheme == "":
            return "file"
        if scheme == "s3":
            return "s3"
        raise ValueError(f"Location with scheme type '{scheme}' not supported.")

    def refresh(self) -> None:
        """Refresh dataset by fully reinitializing."""
        self.__init__(  # type: ignore[misc]
            self.location,
            config=self.config,
            preload_current_records=self.preload_current_records,
        )

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

    def _create_duckdb_schemas(self) -> None:
        """Create DuckDB schemas used by all components."""
        self.conn.execute("create schema metadata;")

    def reflect_sa_tables(self, schemas: list[str] | None = None) -> None:
        """Reflect SQLAlchemy metadata for DuckDB schemas.

        This centralizes SA reflection for all composed components. Reflected tables
        are stored in self.sa_tables as {schema: {table_name: Table}}.

        Args:
            schemas: list of schemas to reflect; defaults to ["metadata"]
        """
        start_time = time.perf_counter()
        schemas = schemas or ["metadata"]

        engine = create_engine(
            "duckdb://",
            creator=lambda: ConnectionWrapper(self.conn),
        )

        for schema in schemas:
            db_metadata = MetaData()
            db_metadata.reflect(bind=engine, schema=schema, views=True)

            self.sa_tables[schema] = {
                table_name.removeprefix(f"{schema}."): table
                for table_name, table in db_metadata.tables.items()
            }

        logger.debug(
            f"SQLAlchemy reflection complete for schemas {schemas}, "
            f"{round(time.perf_counter() - start_time, 3)}s"
        )

    def get_sa_table(self, schema: str, table: str) -> Table:
        """Get a reflected SQLAlchemy Table by schema and table name."""
        if schema not in self.sa_tables:
            raise ValueError(f"Schema '{schema}' not found in reflected SA tables.")
        if table not in self.sa_tables[schema]:
            raise ValueError(f"Table '{table}' not found in schema '{schema}'.")
        return self.sa_tables[schema][table]
