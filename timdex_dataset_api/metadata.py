"""timdex_dataset_api/metadata.py"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Unpack, cast

from duckdb import DuckDBPyConnection
from duckdb_engine import Dialect as DuckDBDialect
from sqlalchemy import func, literal, select, text, tuple_

from timdex_dataset_api.config import configure_logger
from timdex_dataset_api.utils import (
    DuckDBConnectionFactory,
    S3Client,
    build_filter_expr_sa,
)

if TYPE_CHECKING:
    from timdex_dataset_api.dataset import DatasetFilters, TIMDEXDataset

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


class TIMDEXDatasetMetadata:
    def __init__(self, timdex_dataset: "TIMDEXDataset") -> None:
        """Init TIMDEXDatasetMetadata.

        Args:
            timdex_dataset: parent TIMDEXDataset instance
        """
        self.timdex_dataset = timdex_dataset
        self.conn = timdex_dataset.conn

        self.create_metadata_structure()
        self._setup_metadata_schema()

    @property
    def location(self) -> str:
        return self.timdex_dataset.location

    @property
    def location_scheme(self) -> Literal["file", "s3"]:
        return self.timdex_dataset.location_scheme

    @property
    def config(self) -> "TIMDEXDataset.config":  # type: ignore[name-defined]
        return self.timdex_dataset.config

    @property
    def preload_current_records(self) -> bool:
        return self.timdex_dataset.preload_current_records

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
        return self.conn.query("""
            select count(*) from metadata.records;
            """).fetchone()[0]  # type: ignore[index]

    @property
    def current_records_count(self) -> int:
        """Count of all current records in dataset."""
        return self.conn.query("""
            select count(*) from metadata.current_records;
            """).fetchone()[0]  # type: ignore[index]

    @property
    def append_deltas_count(self) -> int:
        """Count of all append deltas."""
        return self.conn.query("""
            select count(*) from metadata.append_deltas;
            """).fetchone()[0]  # type: ignore[index]

    def create_metadata_structure(self) -> None:
        """Ensure metadata structure exists in TIMDEX dataset."""
        if self.location_scheme == "file":
            Path(self.metadata_database_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            Path(self.append_deltas_path).mkdir(
                parents=True,
                exist_ok=True,
            )

    def database_exists(self) -> bool:
        """Check if static metadata database file exists."""
        if self.location_scheme == "s3":
            s3_client = S3Client()
            return s3_client.object_exists(self.metadata_database_path)
        return os.path.exists(self.metadata_database_path)

    def rebuild_dataset_metadata(self) -> None:
        """Fully rebuild dataset metadata.

        Work includes:
            - remove any append deltas, understanding a full metadata rebuild
                will pickup that data from the ETL records themselves
            - build a local, temporary static metadata database file, then overwrite the
                canonical version in the dataset (e.g. in S3)
        """
        if self.location_scheme == "s3":
            s3_client = S3Client()
            s3_client.delete_folder(self.append_deltas_path)
        else:
            shutil.rmtree(self.append_deltas_path, ignore_errors=True)

        # build database locally
        with tempfile.TemporaryDirectory() as temp_dir:
            local_db_path = str(Path(temp_dir) / self.metadata_database_filename)

            factory = DuckDBConnectionFactory(location_scheme=self.location_scheme)
            with factory.create_connection(local_db_path) as conn:
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

        # refresh dataset to pick up new metadata
        self.timdex_dataset.refresh()

    def _create_full_dataset_table(self, conn: DuckDBPyConnection) -> None:
        """Create a table of metadata for all records in the ETL parquet dataset.

        This is one of the few times we fully materialize data in a DuckDB connection.
        This is most commonly used when recreating the baseline static metadata database
        file.
        """
        start_time = time.perf_counter()
        logger.debug("creating table static_db.main.records")

        # temporarily increase thread count
        conn.execute("""SET threads = 64;""")

        query = f"""
            create or replace table records as (
                select
                    {",".join(ORDERED_METADATA_COLUMN_NAMES)}
                from read_parquet(
                    '{self.location}/data/records/**/*.parquet',
                     hive_partitioning=true,
                     filename=true
                 )
            );
            """
        conn.execute(query)

        # reset thread count
        conn.execute(f"""SET threads = {self.timdex_dataset.conn_factory.threads};""")

        row_count = conn.query("""select count(*) from records;""").fetchone()[0]  # type: ignore[index]
        logger.info(
            f"'records' table created - rows: {row_count}, "
            f"elapsed: {time.perf_counter() - start_time}"
        )

    def _setup_metadata_schema(self) -> None:
        """Set up metadata schema views in the DuckDB connection.

        Creates views for accessing static metadata DB and append deltas.
        If static DB doesn't exist, logs warning but doesn't fail.
        """
        start_time = time.perf_counter()

        if not self.database_exists():
            logger.warning(
                f"Static metadata database not found @ '{self.metadata_database_path}'. "
                "Consider rebuild via TIMDEXDataset.metadata.rebuild_dataset_metadata()."
            )
            return

        self._attach_database_file(self.conn)
        self._create_append_deltas_view(self.conn)
        self._create_records_union_view(self.conn)
        self._create_current_records_view(self.conn)

        logger.debug(
            "Metadata schema setup for TIMDEXDatasetMetadata, "
            f"{round(time.perf_counter() - start_time, 2)}s"
        )

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
        logger.debug("creating view metadata.append_deltas")

        # get current append delta count
        append_delta_count = conn.execute(f"""
            select count(*) as file_count
            from glob('{self.append_deltas_path}/*.parquet')
            """).fetchone()[0]  # type: ignore[index]
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
        logger.debug("creating view metadata.records")

        conn.execute(f"""
            create or replace view metadata.records as
            (
                select
                    {",".join(ORDERED_METADATA_COLUMN_NAMES)}
                from static_db.records
                union all
                select
                    {",".join(ORDERED_METADATA_COLUMN_NAMES)}
                from metadata.append_deltas
            );
            """)

    def _create_current_records_view(self, conn: DuckDBPyConnection) -> None:
        """Create a view of current records.

        This view builds on the table `records`.

        This metadata view includes only the most current version of each record in the
        dataset.  With the metadata provided from this view, we can streamline data
        retrievals in TIMDEXDataset read methods.

        By default, creates a view only (lazy evaluation). If
        preload_current_records=True, creates a temp table for better performance
        for repeated queries.

        For temp table mode, the data is mostly in memory but has the ability to spill to
        disk if we risk getting too close to our memory constraints. We explicitly set the
        temporary location on disk for DuckDB at "/tmp" to play nice with contexts like
        AWS ECS or Lambda, where sometimes the $HOME env var is missing; DuckDB often
        tries to utilize the user's home directory and this works around that.
        """
        logger.debug("creating view metadata.current_records")

        # SQL for the current records logic (CTEs)
        current_records_query = """
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

                -- CTE of records ranked by run_timestamp
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
            where rn = 1
        """

        # create temp table (materializes in memory)
        if self.preload_current_records:
            logger.debug("creating temp table temp.main.current_records")
            conn.execute("set temp_directory = '/tmp';")
            conn.execute(f"""
                create or replace temp table temp.main.current_records as
                {current_records_query};

                -- create view in metadata schema that points to temp table
                create or replace view metadata.current_records as
                select * from temp.main.current_records;
                """)

        # create view only (lazy evaluation)
        else:
            conn.execute(f"""
                create or replace view metadata.current_records as
                {current_records_query};
                """)

    def merge_append_deltas(self) -> None:
        """Merge append deltas into the static metadata database file."""
        logger.info("merging append deltas into static metadata database file")

        start_time = time.perf_counter()

        s3_client = S3Client()

        # get filenames of append deltas
        append_delta_filenames = (
            self.conn.query("""
                select distinct(append_delta_filename)
                from metadata.append_deltas
                """)
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
            self.conn.execute(f"""
                insert into local_static_db.records
                select
                    {",".join(ORDERED_METADATA_COLUMN_NAMES)}
                from metadata.append_deltas
                """)

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
            f"{self.metadata_database_path}, {time.perf_counter() - start_time}s"
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
            filepath = f"s3://{filepath.removeprefix('s3://')}"

        # perform query + write as one SQL statement
        sql = f"""
        copy (
            select
                {",".join(ORDERED_METADATA_COLUMN_NAMES)}
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
            f"Append delta written: {output_path}, {time.perf_counter() - start_time}s"
        )

    def build_keyset_paginated_metadata_query(
        self,
        table: str,
        *,
        limit: int | None = None,
        where: str | None = None,
        keyset_value: tuple[int, int, int] = (0, 0, 0),
        **filters: Unpack["DatasetFilters"],
    ) -> str:
        """Build SQL query using SQLAlchemy against metadata schema tables and views."""
        sa_table = self.timdex_dataset.get_sa_table("metadata", table)

        # create SQL statement object
        stmt = select(
            sa_table.c.timdex_record_id,
            sa_table.c.run_id,
            func.hash(sa_table.c.run_id).label("run_id_hash"),
            sa_table.c.run_record_offset,
            sa_table.c.filename,
            func.hash(sa_table.c.filename).label("filename_hash"),
        ).select_from(sa_table)

        # filter expressions from key/value filters (may return None)
        filter_expr = build_filter_expr_sa(sa_table, **cast("dict", filters))
        if filter_expr is not None:
            stmt = stmt.where(filter_expr)

        # explicit raw WHERE string
        if where is not None and where.strip():
            stmt = stmt.where(text(where))

        # keyset pagination
        filename_has, run_id_hash, run_record_offset_ = keyset_value
        stmt = stmt.where(
            tuple_(
                func.hash(sa_table.c.filename),
                func.hash(sa_table.c.run_id),
                sa_table.c.run_record_offset,
            )
            > tuple_(
                literal(filename_has),
                literal(run_id_hash),
                literal(run_record_offset_),
            )
        )

        # order by filename + run_record_offset
        stmt = stmt.order_by(
            func.hash(sa_table.c.filename),
            func.hash(sa_table.c.run_id),
            sa_table.c.run_record_offset,
        )

        # apply limit if present
        if limit:
            stmt = stmt.limit(limit)

        # using DuckDB dialect, compile to SQL string
        compiled = stmt.compile(
            dialect=DuckDBDialect(),
            compile_kwargs={"literal_binds": True},
        )
        return str(compiled)
