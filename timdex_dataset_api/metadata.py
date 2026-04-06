"""timdex_dataset_api/metadata.py"""

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Unpack, cast

from duckdb import BinderException as DuckDBBinderException
from duckdb import CatalogException as DuckDBCatalogException
from duckdb import DuckDBPyConnection
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
    from timdex_dataset_api.dataset import TIMDEXDataset
    from timdex_dataset_api.records import RecordsFilters

logger = configure_logger(__name__)


@dataclass(frozen=True)
class DataTypeMetadataConfig:
    """Configuration for a data type's participation in the metadata layer."""

    name: str
    """Identifier and static DB table name, e.g. 'records', 'embeddings'."""

    metadata_columns: list[str]
    """Ordered column names for the static DB table and append deltas.
    These are the lightweight metadata columns — no large payloads
    (no source_record, transformed_record, embedding_vector, etc.)."""

    data_path_segment: str
    """Relative path segment under the dataset location, e.g. 'data/records'."""

    prejoin_records: bool = True
    """If True, metadata union views pre-join to metadata.records, adding
    source, run_date, run_type, action, run_timestamp as columns.
    Set to False for 'records' (columns are native); True for bolt-on types."""


@dataclass(frozen=True)
class CurrentMetadataViewSpec:
    """Domain-owned definition for a current-metadata view."""

    name: str
    """View name created in metadata schema, e.g. 'current_records'."""

    query_sql: str
    """SQL query body used to create the view."""

    required_metadata_tables: list[str]
    """Metadata tables/views that must exist before creating this view."""

    preload_setting_attribute: str | None = None
    """Optional TIMDEXDataset bool attribute controlling temp-table preload."""


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
        self.data_type_configs = timdex_dataset.data_type_configs
        self.current_metadata_view_specs = timdex_dataset.current_metadata_view_specs

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

    def append_deltas_path_for(self, config: DataTypeMetadataConfig) -> str:
        """Return the append deltas path for a specific data type."""
        return f"{self.metadata_root}/append_deltas/{config.name}"

    def get_config(self, name: str) -> DataTypeMetadataConfig:
        """Lookup a DataTypeMetadataConfig by name."""
        for config in self.data_type_configs:
            if config.name == name:
                return config
        raise ValueError(f"No metadata config for data type: {name}")

    def resolve_data_type_name_for_table(self, table: str) -> str:
        """Resolve a metadata table/view name to its owning data type config name."""
        for config in self.data_type_configs:
            if table == config.name or table.endswith(f"_{config.name}"):
                return config.name

        raise ValueError(f"Could not resolve data type for metadata table '{table}'.")

    def data_type_metadata_columns_for(self, data_type_name: str) -> list[str]:
        """Return type-specific metadata columns (excluding shared base metadata)."""
        config = self.get_config(data_type_name)
        return [
            column_name
            for column_name in config.metadata_columns
            if column_name not in self.BASE_METADATA_COLUMNS
        ]

    def get_metadata_columns_for_table(self, table: str) -> list[str]:
        """Return canonical metadata columns projected by read keyset queries.

        This method combines self.BASE_METADATA_COLUMNS with metadata columns specific to,
        and identified by, the data class.
        """
        sa_table = self.timdex_dataset.get_sa_table("metadata", table)
        available_columns = set(sa_table.c.keys())

        data_type_name = self.resolve_data_type_name_for_table(table)
        type_metadata_columns = self.data_type_metadata_columns_for(data_type_name)

        expected_columns = self.BASE_METADATA_COLUMNS + type_metadata_columns

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

    def append_deltas_count_for(self, config: DataTypeMetadataConfig) -> int:
        """Count append deltas rows for a single data type."""
        view_name = f"{config.name}_append_deltas"
        return self.timdex_dataset.conn.query(f"""
            select count(*) from metadata.{view_name};
        """).fetchone()[0]  # type: ignore[index]

    @property
    def append_deltas_count(self) -> int:
        """Count of append deltas rows across all registered data types."""
        total = 0
        for config in self.data_type_configs:
            try:
                total += self.append_deltas_count_for(config)
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
            for config in self.data_type_configs:
                Path(self.append_deltas_path_for(config)).mkdir(
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
        for config in self.data_type_configs:
            deltas_path = self.append_deltas_path_for(config)
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

        Iterates over registered data type configs and creates one table per type.
        Gracefully skips data types whose parquet data does not yet exist.
        """
        for config in self.data_type_configs:
            self._create_metadata_table(conn, config)

    def _create_metadata_table(
        self, conn: DuckDBPyConnection, config: DataTypeMetadataConfig
    ) -> None:
        """Create a metadata table for a single data type in the static database."""
        start_time = time.perf_counter()
        data_path = (
            f"{self.timdex_dataset.location.removesuffix('/')}/{config.data_path_segment}"
        )

        logger.debug(f"creating table static_db.main.{config.name}")

        try:
            sql_query = f"""
                create or replace table {config.name} as (
                    select {",".join(config.metadata_columns)}
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
                f"Could not create metadata table for '{config.name}' "
                f"(no parquet data at '{data_path}'). Skipping."
            )
            return

        row_count = conn.query(f"select count(*) from {config.name};").fetchone()[0]  # type: ignore[index]
        logger.info(
            f"'{config.name}' table created - rows: {row_count}, "
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

        self._attach_database_file(self.timdex_dataset.conn)

        for config in self.data_type_configs:
            self._create_append_deltas_view(self.timdex_dataset.conn, config)
            self._create_union_view(self.timdex_dataset.conn, config)

        for spec in self.current_metadata_view_specs:
            self._create_current_metadata_view(self.timdex_dataset.conn, spec)

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
        self, conn: DuckDBPyConnection, config: DataTypeMetadataConfig
    ) -> None:
        """Create a view that projects over append delta parquet files for a data type.

        If there are NO append deltas (e.g. after a rebuild or merge), we still create a
        view by utilizing the schema from the static DB table but without any rows.  This
        allows downstream views to be built on top of this view.

        The view is named ``metadata.{config.name}_append_deltas``.
        """
        view_name = f"{config.name}_append_deltas"
        deltas_path = self.append_deltas_path_for(config)
        static_table = f"static_db.{config.name}"

        logger.debug(f"creating view metadata.{view_name}")

        # get current append delta count
        append_delta_count = conn.execute(f"""
            select count(*) as file_count
            from glob('{deltas_path}/*.parquet')
        """).fetchone()[0]  # type: ignore[index]
        logger.debug(f"{append_delta_count} append deltas found for '{config.name}'")

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
            and table_name = '{config.name}'
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
            f"No static table or append deltas found for '{config.name}'; "
            f"skipping append deltas view for '{config.name}'."
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
        self, conn: DuckDBPyConnection, config: DataTypeMetadataConfig
    ) -> None:
        """Create a union view combining static DB and append deltas for a data type.

        The view is named ``metadata.{config.name}`` and unions
        ``static_db.{config.name}`` with ``metadata.{config.name}_append_deltas``.

        For bolt-on data types (``config.prejoin_records=True``), the view pre-joins
        to ``metadata.records`` so that ``source``, ``run_date``, ``run_type``,
        ``action``, and ``run_timestamp`` are available as filterable columns.
        """
        view_name = config.name
        static_table = f"static_db.{config.name}"
        deltas_view = f"metadata.{config.name}_append_deltas"
        columns = ",".join(config.metadata_columns)

        logger.debug(f"creating view metadata.{view_name}")

        static_table_exists = conn.execute(f"""
            select count(*) from information_schema.tables
            where table_catalog = 'static_db'
            and table_name = '{config.name}'
        """).fetchone()[0]  # type: ignore[index]

        deltas_view_exists = conn.execute(f"""
            select count(*) from information_schema.tables
            where table_schema = 'metadata'
            and table_name = '{config.name}_append_deltas'
            and table_type = 'VIEW'
        """).fetchone()[0]  # type: ignore[index]

        # build the base union (or single-source) subquery
        base_subquery = self._build_base_union_sql(
            static_table,
            deltas_view,
            columns,
            static_table_exists,
            deltas_view_exists,
        )

        if base_subquery is None:
            logger.debug(
                f"No static table or append deltas view found for '{config.name}'; "
                f"skipping union view for '{config.name}'."
            )
            return

        if config.prejoin_records:
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
        """Return base union SQL or None if neither source exists."""
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

    def _create_current_metadata_view(
        self, conn: DuckDBPyConnection, spec: CurrentMetadataViewSpec
    ) -> None:
        """Create a current metadata view from a registered domain-owned spec."""
        missing_tables = [
            table_name
            for table_name in spec.required_metadata_tables
            if not self._metadata_table_exists(conn, table_name)
        ]

        if missing_tables:
            logger.warning(
                f"Skipping metadata.{spec.name} view creation because missing "
                f"dependencies: {', '.join(missing_tables)}"
            )
            return

        logger.debug(f"creating view metadata.{spec.name}")

        if self._should_preload_current_view(spec):
            logger.debug(f"creating temp table temp.main.{spec.name}")
            conn.execute("set temp_directory = '/tmp';")
            conn.execute(f"""
                create or replace temp table temp.main.{spec.name} as
                {spec.query_sql};

                create or replace view metadata.{spec.name} as
                select * from temp.main.{spec.name};
            """)
            return

        conn.execute(f"""
            create or replace view metadata.{spec.name} as
            {spec.query_sql};
        """)

    def _should_preload_current_view(self, spec: CurrentMetadataViewSpec) -> bool:
        """Return True when a view spec is configured for temp-table preloading."""
        if spec.preload_setting_attribute is None:
            return False
        return bool(getattr(self.timdex_dataset, spec.preload_setting_attribute, False))

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

        Iterates over all data type configs, merging each type's deltas into its
        corresponding table in the static database.
        """
        logger.info("merging append deltas into static metadata database file")

        start_time = time.perf_counter()

        s3_client = S3Client()

        # collect all append delta filenames across all types
        all_delta_filenames: dict[str, list[str]] = {}
        has_any_deltas = False
        for config in self.data_type_configs:
            deltas_view = f"{config.name}_append_deltas"
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
            all_delta_filenames[config.name] = filenames
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
            for config in self.data_type_configs:
                if not all_delta_filenames[config.name]:
                    continue
                self._merge_deltas_for_type(config)

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

        # delete append deltas for all types
        for config in self.data_type_configs:
            for delta_filename in all_delta_filenames[config.name]:
                if self.timdex_dataset.location_scheme == "s3":
                    s3_client.delete_file(s3_uri=delta_filename)
                else:
                    os.remove(delta_filename)

        logger.debug(
            "append deltas merged into the static metadata database file: "
            f"{self.metadata_database_path}, {time.perf_counter() - start_time}s"
        )

    def _merge_deltas_for_type(self, config: DataTypeMetadataConfig) -> None:
        """Insert rows from append deltas into the local static DB for one data type."""
        columns = ",".join(config.metadata_columns)
        deltas_view = f"metadata.{config.name}_append_deltas"

        logger.debug(f"merging append deltas for '{config.name}'")

        # if type table doesn't yet exist in static DB, initialize it from deltas schema
        table_exists = self.timdex_dataset.conn.execute(f"""
            select count(*) from information_schema.tables
            where table_catalog = 'local_static_db'
            and table_name = '{config.name}'
        """).fetchone()[0]  # type: ignore[index]

        if not table_exists:
            self.timdex_dataset.conn.execute(f"""
                create table local_static_db.{config.name} as
                select {columns}
                from {deltas_view}
                where 1 = 0
            """)

        self.timdex_dataset.conn.execute(f"""
            insert into local_static_db.{config.name}
            select {columns}
            from {deltas_view}
        """)

    def write_append_delta(
        self,
        filepath: str,
        config: DataTypeMetadataConfig,
    ) -> None:
        """Write an append delta for a parquet file.

        A DuckDB context is used to read metadata-only columns from the parquet
        file, then write an append delta parquet file to
        ``metadata/append_deltas/{config.name}/``.

        Note: this operation is safe in parallel with other possible append delta writes.

        Args:
            filepath: path to the parquet file to extract metadata from
            config: the DataTypeMetadataConfig for this data type
        """
        start_time = time.perf_counter()

        deltas_path = self.append_deltas_path_for(config)
        output_path = f"{deltas_path}/append_delta-{filepath.split('/')[-1]}"

        # ensure s3:// schema prefix is present
        if self.timdex_dataset.location_scheme == "s3":
            filepath = f"s3://{filepath.removeprefix('s3://')}"

        sql = f"""
        copy (
            select
                {",".join(config.metadata_columns)}
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
