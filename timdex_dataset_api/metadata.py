"""timdex_dataset_api/metadata.py"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Unpack, cast

from duckdb import BinderException as DuckDBBinderException
from duckdb import CatalogException as DuckDBCatalogException
from duckdb import DuckDBPyConnection
from duckdb import HTTPException as DuckDBHTTPException
from duckdb import IOException as DuckDBIOException
from duckdb_engine import Dialect as DuckDBDialect
from sqlalchemy import func, literal, select, text, tuple_

from timdex_dataset_api.config import configure_logger
from timdex_dataset_api.utils import (
    DuckDBConnectionFactory,
    S3Client,
    build_filter_expr_sa,
)

if TYPE_CHECKING:
    from timdex_dataset_api.data_type import DataTypeTableConfig, TIMDEXDataType
    from timdex_dataset_api.data_types.records import RecordsFilters
    from timdex_dataset_api.dataset import TIMDEXDataset

logger = configure_logger(__name__)


class TIMDEXDatasetMetadata:
    """Class to handle metadata for all data types in the TIMDEXDataset."""

    BASE_METADATA_COLUMNS: ClassVar[list[str]] = [
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

    def __init__(self, timdex_dataset: "TIMDEXDataset") -> None:
        """Init TIMDEXDatasetMetadata.

        Args:
            timdex_dataset: parent TIMDEXDataset instance
        """
        self.timdex_dataset = timdex_dataset
        self.data_type_classes = timdex_dataset.data_type_classes
        self.table_configs = timdex_dataset.table_configs

        self.create_metadata_structure()
        self._setup_metadata_schema()

    @property
    def metadata_root(self) -> str:
        return f"{self.timdex_dataset.location.removesuffix('/')}/metadata"

    @property
    def metadata_database_filename(self) -> str:
        return "metadata.duckdb"

    @property
    def metadata_database_path(self) -> str:
        return f"{self.metadata_root}/{self.metadata_database_filename}"

    def append_deltas_path_for(self, data_type_class: type["TIMDEXDataType"]) -> str:
        """Return the append deltas path for a specific data type."""
        return f"{self.metadata_root}/append_deltas/{data_type_class.NAME}"

    def resolve_data_type_class_for_table(self, table: str) -> type["TIMDEXDataType"]:
        """Resolve a metadata table/view name to its owning data type class."""
        for data_type_class in self.data_type_classes:
            if table == data_type_class.NAME or table.endswith(
                f"_{data_type_class.NAME}"
            ):
                return data_type_class

        raise ValueError(f"Could not resolve data type for metadata table '{table}'.")

    def data_type_metadata_columns_for(
        self, data_type_class: type["TIMDEXDataType"]
    ) -> list[str]:
        """Return the full metadata column surface for a data type."""
        return data_type_class.METADATA_COLUMNS

    def get_metadata_columns_for_table(self, table: str) -> list[str]:
        """Return canonical metadata columns projected by read keyset queries.

        The returned columns are derived from the owning data type's metadata surface
        and filtered to columns actually available on the requested table or view.
        """
        sa_table = self.timdex_dataset.get_sa_table("metadata", table)
        available_columns = set(sa_table.c.keys())

        data_type_class = self.resolve_data_type_class_for_table(table)
        expected_columns = self.data_type_metadata_columns_for(data_type_class)

        projected_columns: list[str] = []
        for column_name in expected_columns:
            if column_name not in available_columns:
                continue
            if column_name in projected_columns:
                continue
            projected_columns.append(column_name)

        return projected_columns

    @property
    def records_count(self) -> int:
        """Count of all records in dataset."""
        return self.timdex_dataset.conn.query("""
            select count(*) from metadata.records;
            """).fetchone()[0]  # type: ignore[index]

    @property
    def current_records_count(self) -> int:
        """Count of all current records in dataset."""
        return self.timdex_dataset.conn.query("""
            select count(*) from metadata.current_records;
            """).fetchone()[0]  # type: ignore[index]

    def append_deltas_count_for(self, data_type_class: type["TIMDEXDataType"]) -> int:
        """Count append deltas rows for a single data type."""
        view_name = f"{data_type_class.NAME}_append_deltas"
        return self.timdex_dataset.conn.query(f"""
            select count(*) from metadata.{view_name};
        """).fetchone()[0]  # type: ignore[index]

    @property
    def append_deltas_count(self) -> int:
        """Count of append deltas rows across all registered data types."""
        total = 0
        for data_type_class in self.data_type_classes:
            try:
                total += self.append_deltas_count_for(data_type_class)
            except (DuckDBCatalogException, DuckDBBinderException):
                continue
        return total

    def create_metadata_structure(self) -> None:
        """Ensure metadata structure exists in TIMDEX dataset."""
        if self.timdex_dataset.location_scheme == "file":
            Path(self.metadata_database_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            for data_type_class in self.data_type_classes:
                Path(self.append_deltas_path_for(data_type_class)).mkdir(
                    parents=True,
                    exist_ok=True,
                )

    def database_exists(self) -> bool:
        """Check if static metadata database file exists."""
        if self.timdex_dataset.location_scheme == "s3":
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
        for data_type_class in self.data_type_classes:
            deltas_path = self.append_deltas_path_for(data_type_class)
            if self.timdex_dataset.location_scheme == "s3":
                s3_client = S3Client()
                s3_client.delete_folder(deltas_path)
            else:
                shutil.rmtree(deltas_path, ignore_errors=True)

        # build database locally
        with tempfile.TemporaryDirectory() as temp_dir:
            local_db_path = str(Path(temp_dir) / self.metadata_database_filename)

            factory = DuckDBConnectionFactory(
                location_scheme=self.timdex_dataset.location_scheme
            )
            with factory.create_connection(local_db_path) as conn:
                self._create_full_dataset_table(conn)

            # copy local database file to remote location
            if self.timdex_dataset.location_scheme == "s3":
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
        """Create metadata tables for all data types in the static database.

        Iterates over registered data type classes and creates one table per data type.
        Gracefully skips data types whose parquet data does not yet exist.
        """
        for data_type_class in self.data_type_classes:
            self._create_metadata_table(conn, data_type_class)

    def _create_metadata_table(
        self, conn: DuckDBPyConnection, data_type_class: type["TIMDEXDataType"]
    ) -> None:
        """Create a metadata table for a single data type in the static database."""
        start_time = time.perf_counter()
        dataset_location = self.timdex_dataset.location.removesuffix("/")
        data_path = f"{dataset_location}/{data_type_class.DATA_PATH}"

        logger.debug(f"creating table static_db.main.{data_type_class.NAME}")

        # temporarily increase thread count for parallel parquet file scanning
        conn.execute("SET threads = 64;")

        try:
            sql_query = f"""
                create or replace table {data_type_class.NAME} as (
                    select {",".join(data_type_class.DATATYPE_METADATA_COLUMNS)}
                    from read_parquet(
                        '{data_path}/**/*.parquet',
                        hive_partitioning=true,
                        filename=true
                    )
                );
            """
            conn.execute(sql_query)
        except DuckDBIOException:
            logger.warning(
                f"Could not create metadata table for '{data_type_class.NAME}' "
                f"(no parquet data at '{data_path}'). Skipping."
            )
            return
        finally:
            # always reset thread count, even if parquet data is missing
            conn.execute(f"""SET threads = {self.timdex_dataset.conn_factory.threads};""")

        row_count = conn.query(
            f"select count(*) from {data_type_class.NAME};"
        ).fetchone()[0]  # type: ignore[index]
        logger.info(
            f"'{data_type_class.NAME}' table created - rows: {row_count}, "
            f"elapsed: {time.perf_counter() - start_time}"
        )

    def _setup_metadata_schema(self) -> None:
        """Set up metadata schema views in the DuckDB connection.

        Creates views for accessing static metadata DB and append deltas.
        If the static DB does not exist yet, bootstrap metadata views from append
        deltas when available.
        """
        start_time = time.perf_counter()

        if self.database_exists():
            self._attach_database_file(self.timdex_dataset.conn)
        else:
            bootstrap_data_types = []
            for data_type_class in self.data_type_classes:
                append_delta_count = self._append_delta_count(
                    self.timdex_dataset.conn, data_type_class
                )
                if append_delta_count > 0:
                    bootstrap_data_types.append(data_type_class.NAME)

            if bootstrap_data_types:
                logger.warning(
                    "Static metadata database not found @ "
                    f"'{self.metadata_database_path}'. "
                    "Bootstrapping metadata views from append deltas for: "
                    f"{', '.join(bootstrap_data_types)}. "
                    "Consider rebuild via "
                    "TIMDEXDataset.metadata.rebuild_dataset_metadata()."
                )
            else:
                logger.warning(
                    "Static metadata database not found @ "
                    f"'{self.metadata_database_path}'. "
                    "Consider rebuild via "
                    "TIMDEXDataset.metadata.rebuild_dataset_metadata()."
                )

        for data_type_class in self.data_type_classes:
            self._create_append_deltas_view(self.timdex_dataset.conn, data_type_class)
            self._create_union_view(self.timdex_dataset.conn, data_type_class)

        for table_config in self.table_configs:
            if table_config.kind != "custom":
                continue
            self._create_custom_metadata_table(self.timdex_dataset.conn, table_config)

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

    def _create_append_deltas_view(
        self, conn: DuckDBPyConnection, data_type_class: type["TIMDEXDataType"]
    ) -> None:
        """Create a view that projects over append delta parquet files for a data type.

        If there are NO append deltas (e.g. after a rebuild or merge), we still create a
        view by utilizing the schema from the static DB table but without any rows. This
        allows downstream views to be built on top of this view.

        The view is named ``metadata.{data_type_class.NAME}_append_deltas``.
        """
        view_name = f"{data_type_class.NAME}_append_deltas"
        deltas_path = self.append_deltas_path_for(data_type_class)
        static_table = f"static_db.{data_type_class.NAME}"

        logger.debug(f"creating view metadata.{view_name}")

        # get current append delta count
        append_delta_count = self._append_delta_count(conn, data_type_class)
        logger.debug(
            f"{append_delta_count} append deltas found for '{data_type_class.NAME}'"
        )

        # if deltas exist, always create this view from parquet files
        if append_delta_count > 0:
            conn.execute(f"""
                create or replace view metadata.{view_name} as (
                    select *
                    from read_parquet(
                        '{deltas_path}/*.parquet',
                        filename = 'append_delta_filename'
                    )
                );
            """)
            return

        # no deltas: if static table exists, create zero-row mirror
        table_exists = conn.execute(f"""
            select count(*) from information_schema.tables
            where table_catalog = 'static_db'
            and table_name = '{data_type_class.NAME}'
        """).fetchone()[0]  # type: ignore[index]

        if table_exists:
            conn.execute(f"""
                create or replace view metadata.{view_name} as (
                    select *,
                        null::varchar as append_delta_filename
                    from {static_table}
                    where 1 = 0
                );
            """)
            return

        # no static table and no deltas, so no view to create
        logger.debug(
            f"No static table or append deltas found for '{data_type_class.NAME}'; "
            f"skipping append deltas view for '{data_type_class.NAME}'."
        )

    # columns added to bolt-on types via pre-join to metadata.records
    PREJOIN_RECORDS_COLUMNS: ClassVar[list[str]] = [
        "source",
        "run_date",
        "run_type",
        "action",
        "run_timestamp",
    ]

    def _create_union_view(
        self, conn: DuckDBPyConnection, data_type_class: type["TIMDEXDataType"]
    ) -> None:
        """Create a union view combining static DB and append deltas for a data type.

        The view is named ``metadata.{data_type_class.NAME}`` and unions
        `static_db.{data_type_class.NAME}` with
        `metadata.{data_type_class.NAME}_append_deltas`.

        For bolt-on data types (`data_type_class.PREJOIN_RECORDS=True`), the view
        pre-joins to `metadata.records` so that `source`, `run_date`, `run_type`,
        `action`, and `run_timestamp` are available as filterable columns.
        """
        view_name = data_type_class.NAME
        static_table = f"static_db.{data_type_class.NAME}"
        deltas_view = f"metadata.{data_type_class.NAME}_append_deltas"
        columns = ",".join(data_type_class.DATATYPE_METADATA_COLUMNS)

        logger.debug(f"creating view metadata.{view_name}")

        static_table_exists = conn.execute(f"""
            select count(*) from information_schema.tables
            where table_catalog = 'static_db'
            and table_name = '{data_type_class.NAME}'
        """).fetchone()[0]  # type: ignore[index]

        deltas_view_exists = conn.execute(f"""
            select count(*) from information_schema.tables
            where table_schema = 'metadata'
            and table_name = '{data_type_class.NAME}_append_deltas'
            and table_type = 'VIEW'
        """).fetchone()[0]  # type: ignore[index]

        # build the base union (or single-input) subquery
        base_subquery = self._build_base_union_sql(
            static_table,
            deltas_view,
            columns,
            static_table_exists,
            deltas_view_exists,
        )

        if base_subquery is None:
            logger.debug(
                "No static table or append deltas view found for "
                f"'{data_type_class.NAME}'; skipping union view for "
                f"'{data_type_class.NAME}'."
            )
            return

        if data_type_class.PREJOIN_RECORDS:
            if not self._metadata_table_exists(conn, "records"):
                logger.warning(
                    f"Skipping metadata.{view_name} view creation because missing "
                    "dependency: records"
                )
                return

            prejoin_cols = ",".join(f"r.{c}" for c in self.PREJOIN_RECORDS_COLUMNS)
            join_keys = "timdex_record_id, run_id, run_record_offset"
            conn.execute(f"""
                create or replace view metadata.{view_name} as
                select
                    e.*,
                    {prejoin_cols}
                from ({base_subquery}) e
                join metadata.records r using ({join_keys});
            """)
        else:
            conn.execute(f"""
                create or replace view metadata.{view_name} as
                {base_subquery};
            """)

    @staticmethod
    def _build_base_union_sql(
        static_table: str,
        deltas_view: str,
        columns: str,
        static_table_exists: int,
        deltas_view_exists: int,
    ) -> str | None:
        """Return base union SQL or None if neither input exists."""
        if static_table_exists and deltas_view_exists:
            return f"""
                select {columns} from {static_table}
                union all
                select {columns} from {deltas_view}
            """
        if static_table_exists:
            return f"select {columns} from {static_table}"
        if deltas_view_exists:
            return f"select {columns} from {deltas_view}"
        return None

    def _create_custom_metadata_table(
        self, conn: DuckDBPyConnection, table_config: "DataTypeTableConfig"
    ) -> None:
        """Create a custom metadata view from a data-type table config."""
        missing_tables = [
            table_name
            for table_name in table_config.required_metadata_tables
            if not self._metadata_table_exists(conn, table_name)
        ]

        if missing_tables:
            logger.warning(
                f"Skipping metadata.{table_config.name} view creation because missing "
                f"dependencies: {', '.join(missing_tables)}"
            )
            return

        if table_config.query_sql is None:
            raise ValueError(
                f"Custom metadata table '{table_config.name}' must define query_sql."
            )

        logger.debug(f"creating view metadata.{table_config.name}")

        if self._should_preload_table(table_config):
            logger.debug(f"creating temp table temp.main.{table_config.name}")
            conn.execute("set temp_directory = '/tmp';")
            conn.execute(f"""
                create or replace temp table temp.main.{table_config.name} as
                {table_config.query_sql};

                create or replace view metadata.{table_config.name} as
                select * from temp.main.{table_config.name};
            """)
            return

        conn.execute(f"""
            create or replace view metadata.{table_config.name} as
            {table_config.query_sql};
        """)

    def _should_preload_table(self, table_config: "DataTypeTableConfig") -> bool:
        """Return True when a table config is configured for temp-table preloading."""
        if table_config.preload_setting_attribute is None:
            return False
        return bool(
            getattr(self.timdex_dataset, table_config.preload_setting_attribute, False)
        )

    def _append_delta_count(
        self, conn: DuckDBPyConnection, data_type_class: type["TIMDEXDataType"]
    ) -> int:
        """Return append delta parquet file count for a single data type."""
        deltas_glob = f"{self.append_deltas_path_for(data_type_class)}/*.parquet"

        try:
            return cast(
                "int",
                conn.execute(f"""
                    select count(*) as file_count
                    from glob('{deltas_glob}')
                """).fetchone()[0],  # type: ignore[index]
            )
        except (DuckDBHTTPException, DuckDBIOException):
            logger.debug(
                "Could not inspect append deltas for "
                f"'{data_type_class.NAME}' at '{deltas_glob}'; assuming none exist."
            )
            return 0

    def _metadata_table_exists(self, conn: DuckDBPyConnection, table_name: str) -> bool:
        """Return True if a metadata schema table or view exists by name."""
        table_exists = conn.execute(f"""
            select count(*) from information_schema.tables
            where table_schema = 'metadata'
            and table_name = '{table_name}'
        """).fetchone()[0]  # type: ignore[index]
        return bool(table_exists)

    def merge_append_deltas(self) -> None:
        """Merge append deltas into the static metadata database file.

        Iterates over all data type configs, merging each data type's deltas into its
        corresponding table in the static database.
        """
        logger.info("merging append deltas into static metadata database file")

        start_time = time.perf_counter()

        s3_client = S3Client()

        # collect all append delta filenames across all data types
        all_delta_filenames: dict[str, list[str]] = {}
        has_any_deltas = False
        for data_type_class in self.data_type_classes:
            deltas_view = f"{data_type_class.NAME}_append_deltas"
            try:
                filenames = (
                    self.timdex_dataset.conn.query(f"""
                        select distinct(append_delta_filename)
                        from metadata.{deltas_view}
                    """)
                    .to_df()["append_delta_filename"]
                    .to_list()
                )
            except (
                DuckDBIOException,
                DuckDBCatalogException,
                DuckDBBinderException,
                KeyError,
            ):
                filenames = []
            all_delta_filenames[data_type_class.NAME] = filenames
            if filenames:
                has_any_deltas = True

        if not has_any_deltas:
            logger.info("no append deltas found")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            # create local copy of the static metadata database (static db) file
            local_db_path = str(Path(temp_dir) / self.metadata_database_filename)
            if self.timdex_dataset.location_scheme == "s3":
                s3_client.download_file(
                    s3_uri=self.metadata_database_path, local_path=local_db_path
                )
            else:
                shutil.copy(src=self.metadata_database_path, dst=local_db_path)

            # attach to local static db
            self.timdex_dataset.conn.execute(
                f"""attach '{local_db_path}' AS local_static_db;"""
            )

            # merge deltas for each data type
            for data_type_class in self.data_type_classes:
                if not all_delta_filenames[data_type_class.NAME]:
                    continue
                self._merge_deltas_for_data_type(data_type_class)

            # detach from local static db
            self.timdex_dataset.conn.execute("""detach local_static_db;""")

            # overwrite static db file with local version
            if self.timdex_dataset.location_scheme == "s3":
                s3_client.upload_file(
                    local_db_path,
                    self.metadata_database_path,
                )
            else:
                shutil.copy(src=local_db_path, dst=self.metadata_database_path)

        # delete append deltas for all data types
        for data_type_class in self.data_type_classes:
            for delta_filename in all_delta_filenames[data_type_class.NAME]:
                if self.timdex_dataset.location_scheme == "s3":
                    s3_client.delete_file(s3_uri=delta_filename)
                else:
                    os.remove(delta_filename)

        logger.debug(
            "append deltas merged into the static metadata database file: "
            f"{self.metadata_database_path}, {time.perf_counter() - start_time}s"
        )

    def _merge_deltas_for_data_type(
        self, data_type_class: type["TIMDEXDataType"]
    ) -> None:
        """Insert rows from append deltas into the local static DB for one data type."""
        columns = ",".join(data_type_class.DATATYPE_METADATA_COLUMNS)
        deltas_view = f"metadata.{data_type_class.NAME}_append_deltas"

        logger.debug(f"merging append deltas for '{data_type_class.NAME}'")

        # if data type table doesn't yet exist in static DB, initialize from deltas schema
        table_exists = self.timdex_dataset.conn.execute(f"""
            select count(*) from information_schema.tables
            where table_catalog = 'local_static_db'
            and table_name = '{data_type_class.NAME}'
        """).fetchone()[0]  # type: ignore[index]

        if not table_exists:
            self.timdex_dataset.conn.execute(f"""
                create table local_static_db.{data_type_class.NAME} as
                select {columns}
                from {deltas_view}
                where 1 = 0
            """)

        self.timdex_dataset.conn.execute(f"""
            insert into local_static_db.{data_type_class.NAME}
            select {columns}
            from {deltas_view}
        """)

    def write_append_delta(
        self,
        filepath: str,
        data_type_class: type["TIMDEXDataType"],
    ) -> None:
        """Write an append delta for a parquet file.

        A DuckDB context is used to read metadata-only columns from the parquet
        file, then write an append delta parquet file to
        ``metadata/append_deltas/{data_type_class.NAME}/``.

        Note: this operation is safe in parallel with other possible append delta writes.

        Args:
            filepath: path to the parquet file to extract metadata from
            data_type_class: the data type class owning this parquet file
        """
        start_time = time.perf_counter()

        deltas_path = self.append_deltas_path_for(data_type_class)
        output_path = f"{deltas_path}/append_delta-{filepath.split('/')[-1]}"

        # ensure s3:// schema prefix is present
        if self.timdex_dataset.location_scheme == "s3":
            filepath = f"s3://{filepath.removeprefix('s3://')}"

        sql = f"""
        copy (
            select
                {",".join(data_type_class.DATATYPE_METADATA_COLUMNS)}
            from read_parquet(
                '{filepath}',
                hive_partitioning=true,
                filename=true
            )
        ) to '{output_path}'
        (FORMAT parquet);
        """
        self.timdex_dataset.conn.execute(sql)

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
        **filters: Unpack["RecordsFilters"],
    ) -> str:
        """Build SQL query using SQLAlchemy against metadata schema tables and views.

        Args:
            table: metadata table/view name
            limit: max rows to return
            where: raw SQL WHERE clause
            keyset_value: tuple of (filename_hash, run_id_hash, run_record_offset)
                for keyset pagination
            **filters: key/value filter pairs
        """
        sa_table = self.timdex_dataset.get_sa_table("metadata", table)

        required_keyset_columns = {"filename", "run_id", "run_record_offset"}
        missing_keyset_columns = required_keyset_columns - set(sa_table.c.keys())
        if missing_keyset_columns:
            missing = ", ".join(sorted(missing_keyset_columns))
            raise ValueError(
                f"Table '{table}' missing required keyset column(s): {missing}"
            )

        metadata_columns = self.get_metadata_columns_for_table(table)

        # create SQL statement object
        select_columns: list[Any] = [
            sa_table.c[column_name] for column_name in metadata_columns
        ]
        select_columns.extend(
            [
                func.hash(sa_table.c.run_id).label("run_id_hash"),
                func.hash(sa_table.c.filename).label("filename_hash"),
            ]
        )

        stmt = select(*select_columns).select_from(sa_table)

        # filter expressions from key/value filters (may return None)
        filter_expr = build_filter_expr_sa(sa_table, **cast("dict", filters))
        if filter_expr is not None:
            stmt = stmt.where(filter_expr)

        # explicit raw WHERE string
        if where is not None and where.strip():
            stmt = stmt.where(text(where))

        # keyset pagination
        filename_hash, run_id_hash, run_record_offset_ = keyset_value
        stmt = stmt.where(
            tuple_(
                func.hash(sa_table.c.filename),
                func.hash(sa_table.c.run_id),
                sa_table.c.run_record_offset,
            )
            > tuple_(
                literal(filename_hash),
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
