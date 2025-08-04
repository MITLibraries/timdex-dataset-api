"""timdex_dataset_api/metadata.py"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import duckdb
from duckdb import DuckDBPyConnection

from timdex_dataset_api.config import configure_logger
from timdex_dataset_api.utils import S3Client, configure_duckdb_s3_secret

logger = configure_logger(__name__)

ORDERED_DATASET_COLUMN_NAMES = [
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
        self.conn: None | DuckDBPyConnection = self.setup_duckdb_context()

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

    def database_exists(self) -> bool:
        """Check if static metadata database file exists."""
        if self.location_scheme == "s3":
            s3_client = S3Client()
            return s3_client.object_exists(self.metadata_database_path)
        return os.path.exists(self.metadata_database_path)

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
                conn.execute("""SET threads = 64;""")
                configure_duckdb_s3_secret(conn)

                self._create_full_dataset_table(conn)

            # copy local database file to remote location
            if self.location_scheme == "s3":
                s3_client = S3Client()
                s3_client.upload_file(
                    local_db_path,
                    self.metadata_database_path,
                )
            else:
                Path(self.metadata_database_path).parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
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
                    {','.join(ORDERED_DATASET_COLUMN_NAMES)}
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

    def setup_duckdb_context(self) -> DuckDBPyConnection | None:
        """Create a DuckDB connection that provides full dataset metadata information.

        The following work is performed:
            1. Attach to static metadata database file.
            2. Create views that union static metadata with any append deltas.
            3. Create additional metadata views as needed.

        The resulting, in-memory DuckDB connection is used for all metadata queries.
        """
        if not self.database_exists():
            logger.warning(
                f"Static metadata database not found @ '{self.metadata_database_path}'. "
                "Please recreate via TIMDEXDatasetMetadata.recreate_database_file()."
            )
            return None

        conn = duckdb.connect()
        configure_duckdb_s3_secret(conn)

        self._attach_database_file(conn)

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
