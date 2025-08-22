"""timdex_dataset_api/metadata.py"""

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Unpack
from urllib.parse import urlparse

import duckdb
from duckdb import DuckDBPyConnection
from duckdb_engine import Dialect as DuckDBDialect
from sqlalchemy import Table, and_, select, text

from timdex_dataset_api.config import configure_logger
from timdex_dataset_api.utils import (
    S3Client,
    build_filter_expr_sa,
    sa_reflect_duckdb_conn,
)

if TYPE_CHECKING:
    from timdex_dataset_api.dataset import DatasetFilters

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
        self._sa_metadata = sa_reflect_duckdb_conn(self.conn, schema="metadata")

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

    @property
    def records_count(self) -> int:
        """Count of all records in dataset."""
        return self.conn.query(
            """
            select count(*) from metadata.records;
            """
        ).fetchone()[
            0
        ]  # type: ignore[index]

    @property
    def current_records_count(self) -> int:
        """Count of all current records in dataset."""
        return self.conn.query(
            """
            select count(*) from metadata.current_records;
            """
        ).fetchone()[
            0
        ]  # type: ignore[index]

    @property
    def append_deltas_count(self) -> int:
        """Count of all append deltas."""
        return self.conn.query(
            """
            select count(*) from metadata.append_deltas;
            """
        ).fetchone()[
            0
        ]  # type: ignore[index]

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
        self._install_duckdb_extensions(conn)
        self._configure_duckdb_s3_secret(conn)
        self._configure_duckdb_memory_profile(conn)

    def _install_duckdb_extensions(self, conn: DuckDBPyConnection) -> None:
        """Ensure DuckDB capable of installing extensions and install any required."""
        # ensure secrets and extensions paths are accessible
        home_env = os.getenv("HOME")
        use_fallback_home = not home_env or not Path(home_env).is_dir()

        if use_fallback_home:
            duckdb_home = Path("/tmp/.duckdb")  # noqa: S108
            secrets_dir = duckdb_home / "secrets"
            extensions_dir = duckdb_home / "extensions"

            secrets_dir.mkdir(parents=True, exist_ok=True)
            extensions_dir.mkdir(parents=True, exist_ok=True)

            conn.execute(f"set secret_directory='{secrets_dir.as_posix()}';")
            conn.execute(f"set extension_directory='{extensions_dir.as_posix()}';")

        # install HTTPFS extension
        conn.execute(
            """
            install httpfs;
            load httpfs;
            """
        )

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

    def get_sa_table(self, table: str) -> Table:
        """Get SQLAlchemy Table from reflected SQLAlchemy metadata."""
        schema_table = f"metadata.{table}"
        if schema_table not in self._sa_metadata.tables:
            raise ValueError(
                f"Could not find table '{table}' in DuckDB schema 'metadata'."
            )
        return self._sa_metadata.tables[schema_table]

    def refresh(self) -> None:
        """Refresh DuckDB connection and reflected SQLAlchemy metadata on self."""
        self.conn = self.setup_duckdb_context()
        self._sa_metadata = sa_reflect_duckdb_conn(self.conn, schema="metadata")

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

        If a static database file is not found, a configured DuckDB connection is still
        returned.
        """
        start_time = time.perf_counter()

        conn = duckdb.connect()
        conn.execute("""SET enable_progress_bar = false;""")
        self.configure_duckdb_connection(conn)

        if not self.database_exists():
            logger.warning(
                f"Static metadata database not found @ '{self.metadata_database_path}'. "
                "Please recreate via TIMDEXDatasetMetadata.recreate_database_file()."
            )
            return conn

        # create metadata schema
        conn.execute("create schema metadata;")

        self._attach_database_file(conn)
        self._create_append_deltas_view(conn)
        self._create_records_union_view(conn)
        self._create_current_records_view(conn)

        logger.debug(
            f"DuckDB metadata context created, {round(time.perf_counter()-start_time,2)}s"
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

    def _create_append_deltas_view(self, conn: DuckDBPyConnection) -> None:
        """Create a view that projects over append delta parquet files.

        If when run there are NO append deltas, which could be true immediately after a
        metadata base create/recreate or append delta merge, we still create a view by
        utilizing the schema from static_db.records but without any rows.  This allows us
        to build additional downstream views on top of *this* view.  Also noting that a
        call to .refresh() will recreate this view.
        """
        logger.debug("creating view of append deltas")

        # get current append delta count
        append_delta_count = conn.execute(
            f"""
            select count(*) as file_count
            from glob('{self.append_deltas_path}/*.parquet')
            """
        ).fetchone()[
            0
        ]  # type: ignore[index]
        logger.debug(f"{append_delta_count} append deltas found")

        # if deltas, create view projecting over those parquet files
        if append_delta_count > 0:
            query = f"""
            create or replace view metadata.append_deltas as (
                select *
                from read_parquet(
                    '{self.append_deltas_path}/*.parquet',
                    filename = 'append_delta_filename'
                )
            );
            """

        # if not, create a view that mirrors the structure of static_db.records
        else:
            query = """
            create or replace view metadata.append_deltas as (
                select *
                from static_db.records
                where 1 = 0
            );"""

        conn.execute(query)

    def _create_records_union_view(self, conn: DuckDBPyConnection) -> None:
        logger.debug("creating view of unioned records")

        conn.execute(
            f"""
            create or replace view metadata.records as
            (
                select
                    {','.join(ORDERED_METADATA_COLUMN_NAMES)}
                from static_db.records
                union all
                select
                    {','.join(ORDERED_METADATA_COLUMN_NAMES)}
                from metadata.append_deltas
            );
            """
        )

    def _create_current_records_view(self, conn: DuckDBPyConnection) -> None:
        """Create a view of current records.

        This view builds on the table `records`.

        This metadata view includes only the most current version of each record in the
        dataset.  With the metadata provided from this view, we can streamline data
        retrievals in TIMDEXDataset read methods.
        """
        logger.info("creating view of current records metadata")

        conn.execute(
            """
            set temp_directory = '/tmp';
            """
        )

        conn.execute(
            """
            -- create temp table with current records using CTEs
            create or replace temp table temp.main.current_records as
            with
                -- CTE of run_timestamp for last source full run
                cr_source_last_full as (
                    select
                        source,
                        max(run_timestamp) as last_full_ts
                    from metadata.records
                    where run_type = 'full'
                    group by source
                ),

                -- CTE of all records, per source, on or after last full run
                cr_since_last_full as (
                    select
                        r.*
                    from metadata.records r
                    join cr_source_last_full f using (source)
                    where r.run_timestamp >= f.last_full_ts
                ),

                -- CTE of records ranked by run_timestamp, with tie breaker
                cr_ranked_records as (
                    select
                        r.*,
                        row_number() over (
                            partition by r.source, r.timdex_record_id
                            order by
                                r.run_timestamp desc nulls last,
                                r.run_id desc nulls last,
                                r.run_record_offset desc nulls last
                        ) as rn
                    from cr_since_last_full r
                )

            -- final select for current records (rn = 1)
            select
                * exclude (rn)
            from cr_ranked_records
            where rn = 1;

            -- create view in metadata schema
            create or replace view metadata.current_records as
            select * from temp.main.current_records;
            """
        )

    def merge_append_deltas(self) -> None:
        """Merge append deltas into the static metadata database file."""
        logger.info("merging append deltas into static metadata database file")

        start_time = time.perf_counter()

        s3_client = S3Client()

        # get filenames of append deltas
        append_delta_filenames = (
            self.conn.query(
                """
                select distinct(append_delta_filename)
                from metadata.append_deltas
                """
            )
            .to_df()["append_delta_filename"]
            .to_list()
        )

        if len(append_delta_filenames) == 0:
            logger.info("no append deltas found")
            return

        logger.debug(f"{len(append_delta_filenames)} append deltas found")

        with tempfile.TemporaryDirectory() as temp_dir:
            # create local copy of the static metadata database (static db) file
            local_db_path = str(Path(temp_dir) / self.metadata_database_filename)
            if self.location_scheme == "s3":
                s3_client.download_file(
                    s3_uri=self.metadata_database_path, local_path=local_db_path
                )
            else:
                shutil.copy(src=self.metadata_database_path, dst=local_db_path)

            # attach to local static db
            self.conn.execute(f"""attach '{local_db_path}' AS local_static_db;""")

            # insert records from append deltas to local static db
            self.conn.execute(
                f"""
                insert into local_static_db.records
                select
                    {','.join(ORDERED_METADATA_COLUMN_NAMES)}
                from metadata.append_deltas
                """
            )

            # detach from local static db
            self.conn.execute("""detach local_static_db;""")

            # overwrite static db file with local version
            if self.location_scheme == "s3":
                s3_client.upload_file(
                    local_db_path,
                    self.metadata_database_path,
                )
            else:
                shutil.copy(src=local_db_path, dst=self.metadata_database_path)

        # delete append deltas
        for append_delta_filename in append_delta_filenames:
            if self.location_scheme == "s3":
                s3_client.delete_file(s3_uri=append_delta_filename)
            else:
                os.remove(append_delta_filename)

        logger.debug(
            "append deltas merged into the static metadata database file: "
            f"{self.metadata_database_path}, {time.perf_counter()-start_time}s"
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

    def build_meta_query(
        self, table: str, where: str | None, **filters: Unpack["DatasetFilters"]
    ) -> str:
        """Build SQL query using SQLAlchemy against metadata schema tables and views."""
        sa_table = self.get_sa_table(table)

        # build WHERE clause filter expression based on any passed key/value filters
        # and/or an explicit WHERE string
        filter_expr = build_filter_expr_sa(sa_table, **filters)
        if where is not None and where.strip():
            text_where = text(where)
            combined = (
                and_(filter_expr, text_where) if filter_expr is not None else text_where
            )
        else:
            combined = filter_expr

        # create SQL statement object
        stmt = select(
            sa_table.c.timdex_record_id,
            sa_table.c.run_id,
            sa_table.c.run_record_offset,
            sa_table.c.filename,
        ).select_from(sa_table)
        if combined is not None:
            stmt = stmt.where(combined)

        # using DuckDB dialect, compile to SQL string
        compiled = stmt.compile(
            dialect=DuckDBDialect(),
            compile_kwargs={"literal_binds": True},
        )
        compiled_str = str(compiled)
        logger.debug(compiled_str)

        return compiled_str
