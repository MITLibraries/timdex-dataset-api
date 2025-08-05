"""timdex_dataset_api/metadata.py"""

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import duckdb
from duckdb import DuckDBPyConnection

from timdex_dataset_api.config import configure_logger
from timdex_dataset_api.utils import S3Client

logger = configure_logger(__name__)

ORDERED_METADATA_COLUMN_NAMES = [
    "timdex_record_id",
    "source",
    "run_date",
    "run_type",
    "action",
    "run_id",
    "run_record_offset",
    "run_timestamp",
    "filename",
]


@dataclass
class TIMDEXDatasetMetadataConfig:
    """Configurations for metadata operations.

    - duckdb_connection_memory_limit: Memory limit for DuckDB connection
    - duckdb_connection_threads: Thread limit for DuckDB connection
    """

    duckdb_connection_memory_limit: str = field(
        default_factory=lambda: os.getenv("TDA_DUCKDB_MEMORY_LIMIT", "4GB")
    )
    duckdb_connection_threads: int = field(
        default_factory=lambda: int(os.getenv("TDA_DUCKDB_THREADS", "8"))
    )


class TIMDEXDatasetMetadata:

    def __init__(
        self,
        location: str,
    ) -> None:
        """Init TIMDEXDatasetMetadata.

        Args:
            location: root location of TIMDEX dataset, e.g. 's3://timdex/dataset'
        """
        self.location = location
        self.config = TIMDEXDatasetMetadataConfig()

        self.create_metadata_structure()
        self.conn: DuckDBPyConnection = self.setup_duckdb_context()

    @property
    def location_scheme(self) -> Literal["file", "s3"]:
        scheme = urlparse(self.location).scheme
        if scheme == "":
            return "file"
        if scheme == "s3":
            return "s3"
        raise ValueError(f"Location with scheme type '{scheme}' not supported.")

    @property
    def metadata_root(self) -> str:
        return f"{self.location.removesuffix('/')}/metadata"

    @property
    def metadata_database_filename(self) -> str:
        return "metadata.duckdb"

    @property
    def metadata_database_path(self) -> str:
        return f"{self.metadata_root}/{self.metadata_database_filename}"

    @property
    def append_deltas_path(self) -> str:
        return f"{self.metadata_root}/append_deltas"

    def create_metadata_structure(self) -> None:
        """Ensure metadata structure exists in TIDMEX dataset.."""
        if self.location_scheme == "file":
            Path(self.metadata_database_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            Path(self.append_deltas_path).mkdir(
                parents=True,
                exist_ok=True,
            )

    def configure_duckdb_connection(self, conn: DuckDBPyConnection) -> None:
        """Configure a DuckDB connection/context.

        These configurations include things like memory settings, AWS authentication, etc.
        """
        self._configure_duckdb_s3_secret(conn)
        self._configure_duckdb_memory_profile(conn)

    def _configure_duckdb_s3_secret(
        self,
        conn: DuckDBPyConnection,
        scope: str | None = None,
    ) -> None:
        """Configure a secret in a DuckDB connection for S3 access.

        If a scope is provided, e.g. an S3 URI prefix like 's3://timdex', set a scope
        parameter in the config.  Else, leave it blank.
        """
        # establish scope string
        scope_str = f", scope '{scope}'" if scope else ""

        if os.getenv("MINIO_S3_ENDPOINT_URL"):
            conn.execute(
                f"""
                create or replace secret minio_s3_secret (
                    type s3,
                    endpoint '{urlparse(os.environ["MINIO_S3_ENDPOINT_URL"]).netloc}',
                    key_id '{os.environ["MINIO_USERNAME"]}',
                    secret '{os.environ["MINIO_PASSWORD"]}',
                    region 'us-east-1',
                    url_style 'path',
                    use_ssl false
                    {scope_str}
                );
                """
            )

        else:
            conn.execute(
                f"""
                create or replace secret aws_s3_secret (
                    type s3,
                    provider credential_chain,
                    chain 'sso;env;config',
                    refresh true
                    {scope_str}
                );
                """
            )

    def _configure_duckdb_memory_profile(self, conn: DuckDBPyConnection) -> None:
        conn.execute(
            f"""
            set enable_external_file_cache = false;
            set memory_limit = '{self.config.duckdb_connection_memory_limit}';
            set threads = {self.config.duckdb_connection_threads};
            set preserve_insertion_order=false;
            """
        )

    def database_exists(self) -> bool:
        """Check if static metadata database file exists."""
        if self.location_scheme == "s3":
            s3_client = S3Client()
            return s3_client.object_exists(self.metadata_database_path)
        return os.path.exists(self.metadata_database_path)

    def refresh(self) -> None:
        """Refresh DuckDB connection on self."""
        self.conn = self.setup_duckdb_context()

    def recreate_static_database_file(self) -> None:
        """Create/recreate the static metadata database file.

        The following work is performed:
            1. Create a local working directory
            2. Open a DuckDB connection with a database file in this local working dir
            3. Create tables and views by scanning ETL data in dataset/data/records
            4. Close DuckDB connection ensuring a fully formed, local database file
            5. Upload DuckDB database file to target destination, making that the new
            static metadata database file
        """
        if self.location_scheme == "s3":
            s3_client = S3Client()
            s3_client.delete_folder(self.append_deltas_path)
        else:
            shutil.rmtree(self.append_deltas_path, ignore_errors=True)

        # build database locally
        with tempfile.TemporaryDirectory() as temp_dir:
            local_db_path = str(Path(temp_dir) / self.metadata_database_filename)

            with duckdb.connect(local_db_path) as conn:
                self.configure_duckdb_connection(conn)
                conn.execute("""SET threads = 64;""")

                self._create_full_dataset_table(conn)

            # copy local database file to remote location
            if self.location_scheme == "s3":
                s3_client = S3Client()
                s3_client.upload_file(
                    local_db_path,
                    self.metadata_database_path,
                )
            else:
                shutil.copy(local_db_path, self.metadata_database_path)

        # refresh DuckDB connection
        self.conn = self.setup_duckdb_context()

    def _create_full_dataset_table(self, conn: DuckDBPyConnection) -> None:
        """Create a table of metadata for all records in the ETL parquet dataset.

        This is one of the few times we fully materialize data in a DuckDB connection.
        This is most commonly used when recreating the baseline static metadata database
        file.
        """
        start_time = time.perf_counter()
        logger.info("creating table of full dataset metadata")

        query = f"""
            create or replace table records as (
                select
                    {','.join(ORDERED_METADATA_COLUMN_NAMES)}
                from read_parquet(
                    '{self.location}/data/records/**/*.parquet',
                     hive_partitioning=true,
                     filename=true
                 )
            );
            """
        conn.execute(query)

        row_count = conn.query("""select count(*) from records;""").fetchone()[0]  # type: ignore[index]
        logger.info(
            f"'records' table created - rows: {row_count}, "
            f"elapsed: {time.perf_counter() - start_time}"
        )

    def setup_duckdb_context(self) -> DuckDBPyConnection:
        """Create a DuckDB connection that provides full dataset metadata information.

        The following work is performed:
            1. Attach to static metadata database file.
            2. Create views that union static metadata with any append deltas.
            3. Create additional metadata views as needed.

        The resulting, in-memory DuckDB connection is used for all metadata queries.
        """
        conn = duckdb.connect()
        self.configure_duckdb_connection(conn)

        if self.database_exists():
            self._attach_database_file(conn)
        else:
            logger.warning(
                f"Static metadata database not found @ '{self.metadata_database_path}'. "
                "Please recreate via TIMDEXDatasetMetadata.recreate_database_file()."
            )

        return conn

    def _attach_database_file(self, conn: DuckDBPyConnection) -> None:
        """Readonly attach to static metadata database.

        Attaching to a remote DuckDB database file is supported, but only in readonly
        mode: https://duckdb.org/docs/stable/sql/statements/attach.html, though it does
        support multiple, concurrent attachments.
        """
        logger.debug(f"Attaching to static database file: {self.metadata_database_path}")
        conn.execute(
            f"""attach '{self.metadata_database_path}' AS static_db (READ_ONLY);"""
        )

    def write_append_delta_duckdb(self, filepath: str) -> None:
        """Write an append delta for an ETL parquet file.

        A DuckDB context is used to both read metadata-only columns from the ETL parquet
        file, then write an append delta parquet file to /metadata/append_deltas.  The
        write is performed by DuckDB's COPY function.

        Note: this operation is safe in parallel with other possible append delta writes.
        """
        start_time = time.perf_counter()

        output_path = f"{self.append_deltas_path}/append_delta-{filepath.split('/')[-1]}"

        # ensure s3:// schema prefix is present
        if self.location_scheme == "s3":
            filepath = f"s3://{filepath.removeprefix("s3://")}"

        # perform query + write as one SQL statement
        sql = f"""
        copy (
            select
                {','.join(ORDERED_METADATA_COLUMN_NAMES)}
            from read_parquet(
                '{filepath}',
                hive_partitioning=true,
                filename=true
            )
        ) to '{output_path}'
        (FORMAT parquet);
        """
        self.conn.execute(sql)

        logger.debug(
            f"Append delta written: {output_path}, {time.perf_counter()-start_time}s"
        )
