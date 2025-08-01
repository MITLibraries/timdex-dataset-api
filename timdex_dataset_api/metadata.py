"""timdex_dataset_api/metadata.py"""

import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import TYPE_CHECKING, Unpack
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
import duckdb
from duckdb import DuckDBPyConnection

from timdex_dataset_api.config import configure_logger
from timdex_dataset_api.utils import S3Client

if TYPE_CHECKING:
    from timdex_dataset_api.dataset import DatasetFilters, TIMDEXDataset

logger = configure_logger(__name__)


class TIMDEXDatasetMetadata:

    def __init__(
        self,
        location: str,
    ):
        """

        Args:
            location: root location of TIMDEXDataset
        """
        self.location = location

        self.conn = None
        if self.database_exists():
            self.conn = self.setup_duckdb_connection()
        else:
            logger.warning(
                f"Expected metadata database not found: '{self.metadata_database_path}'. "
                "Consider recreating via recreate_database_file()."
            )

    @property
    def metadata_root_path(self):
        return f"{self.location.removesuffix('/')}/metadata"

    @property
    def metadata_database_path(self):
        return f"{self.metadata_root_path}/metadata.duckdb"

    @property
    def append_deltas_path(self):
        return f"{self.metadata_root_path}/append_deltas"

    def database_exists(self):
        components = urlparse(self.metadata_database_path)
        if components.scheme == "s3":
            return self.s3_object_exists(self.metadata_database_path)
        else:
            return os.path.exists(self.metadata_database_path)

    def get_s3_client(self):
        s3_client_kwargs = {}
        if os.getenv("MINIO_S3_ENDPOINT_URL"):
            s3_client_kwargs.update(
                {
                    "endpoint_url": os.environ["MINIO_S3_ENDPOINT_URL"],
                    "aws_access_key_id": os.environ["MINIO_USERNAME"],
                    "aws_secret_access_key": os.environ["MINIO_PASSWORD"],
                    "region_name": "us-east-1",  # MinIO default
                }
            )

        s3_client = boto3.client("s3", **s3_client_kwargs)
        return s3_client

    def s3_object_exists(self, s3_uri: str):

        components = urlparse(s3_uri)
        bucket, key = components.netloc, components.path.removesuffix("/")
        s3_client = self.get_s3_client()
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                raise

    def recreate_database_file(self):
        """Fully recreated the metadata DuckDB file.

        This uses the MetadataDuckDBDatabase class which fully encapsulates the structure
        and schema of the static metadata database.
        """
        # remove any append deltas before build as those rows in the data to be read
        self._remove_all_append_deltas()

        with tempfile.TemporaryDirectory() as temp_dir:

            # create local, temporary path for physical database file
            local_db_path = str(Path(temp_dir) / "metadata.duckdb")

            # build database file
            metadata_database = MetadataDuckDBDatabase(
                location=self.location,
                local_db_path=local_db_path,
            )
            metadata_database.build()

            # copy local database file to destination
            self._write_database_file(
                local_db_path,
                self.metadata_database_path,
            )

        self.conn = self.setup_duckdb_connection()

    # TODO: will want a method like _get_current_append_deltas()
    #   this will allow a smarter delete that only targets delta created before NOW
    def _remove_all_append_deltas(self):
        components = urlparse(self.append_deltas_path)
        logger.debug(f"Checking for append deltas at: '{self.append_deltas_path}'")

        if components.scheme == "s3":
            bucket, prefix = components.netloc, components.path.removeprefix("/")
            s3_client = self.get_s3_client()
            paginator = s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
                        logger.info(f"Deleted append delta: {obj['Key']}")
        else:
            files_to_delete = list(Path(self.append_deltas_path).glob("append_delta*"))
            for file_path in files_to_delete:
                if file_path.is_file():  # Only delete files, not directories
                    file_path.unlink()
                    logger.info(f"Deleted append delta: {file_path}")

    def _write_database_file(self, source_location: str, target_location: str):
        """Used for full recreate and compaction."""
        target_components = urlparse(target_location)

        # s3 target
        if target_components.scheme == "s3":
            s3_client = self.get_s3_client()
            s3_client.upload_file(
                source_location,
                target_components.netloc,
                target_components.path.removeprefix("/"),
            )

        # local target
        else:
            target_dir = os.path.dirname(target_location)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(source_location, target_location)

    def refresh(self):
        logger.info("Refreshing dataset metadata connection")
        self.conn = self.setup_duckdb_connection()

    # TODO: this method could use a rename
    def setup_duckdb_connection(self) -> DuckDBPyConnection:
        """Setup a DuckDB connection to dataset metadata.

        A DuckDB connection is created, configured, and returned.  This connection
        attaches to the metadata DuckDB file for the dataset, and projects over any
        write deltas, providing read only views unioning the two.
        """
        conn = duckdb.connect()

        configure_s3_connection(conn)
        self._attach_database_file(conn)
        self._create_append_deltas_view(conn)
        self._create_records_union_view(conn)
        self._create_metadata_views(conn)

        logger.info("Dataset metadata DuckDB context created.")
        return conn

    def _attach_database_file(self, conn: DuckDBPyConnection):
        logger.debug("attaching to remote database")
        conn.execute(
            f"""
            attach '{self.metadata_root_path}/metadata.duckdb' AS remote_db (READ_ONLY);
            """
        )

    def _create_append_deltas_view(self, conn: DuckDBPyConnection):
        logger.debug("creating view of append deltas")

        # check if any append deltas
        append_delta_count = conn.execute(
            f"""
            select count(*) as file_count
            from glob('{self.metadata_root_path}/append_deltas/*.parquet')
            """
        ).fetchone()[0]

        logger.debug(f"{append_delta_count} append deltas found")

        # if deltas, create view projecting over those parquet files
        # TODO: use new properties...
        if append_delta_count > 0:
            query = f"""
            create view append_deltas as (
                select *
                from read_parquet(
                    '{self.metadata_root_path}/append_deltas/*.parquet'
                )
            );
            """

        # if not, create a view that mirrors the structure of remote_db.records
        else:
            query = """
            create view append_deltas as (
                select * from remote_db.records where 1=0
            );"""

        conn.execute(query)

    def _create_records_union_view(self, conn: DuckDBPyConnection):
        logger.debug("creating view of unioned records")
        conn.execute(
            """
            create view records as (
                select *
                from remote_db.records
                union all
                select *
                from append_deltas
            );
            """
        )

    def _create_metadata_views(self, conn: DuckDBPyConnection):
        """Create all required metadata views from MetadataDuckDBDatabase schema."""
        MetadataDuckDBDatabase.create_current_records_view(conn)

    # DEBUG: direct port from v1 --------------------------------------------------------
    # TODO: revisit these, maybe streamline?

    def get_current_parquet_files(
        self,
        *,
        strip_protocol_prefix: bool = True,
        **filters: Unpack["DatasetFilters"],
    ) -> list[str]:
        """Provide a list of parquet files that contain one or more current records.

        Args:
            - strip_protocol_prefix: boolean if the file protocol should be removed,
                e.g. "s3://"
            - **filters: keyword dataset filters like `source="alma"` or
                `run_date="2025-05-01"`
        """
        where_clause = self._prepare_where_clause_from_dataset_filters(**filters)

        query = f"""
        select distinct
            filename as parquet_filename
        from current_records
        {where_clause}
        order by run_timestamp desc;
        """
        parquet_files_df = self.conn.query(query).to_df()

        if strip_protocol_prefix:
            parquet_files_df["parquet_filename"] = parquet_files_df[
                "parquet_filename"
            ].apply(lambda x: x.removeprefix("s3://"))

        return list(parquet_files_df["parquet_filename"])

    def get_current_record_to_run_map(self, **filters: Unpack["DatasetFilters"]) -> dict:
        """Provide a dictionary of timdex_record_id --> run_id for current records.

        This dictionary is all that read methods in TIMDEXDataset would require to ensure
        they only yield the current version of a record.

        Args:
            - **filters: keyword dataset filters like `source="alma"` or
                `run_date="2025-05-01"`
        """
        start_time = time.perf_counter()

        where_clause = self._prepare_where_clause_from_dataset_filters(**filters)

        query = f"""
        select
            timdex_record_id,
            run_id
        from current_records
        {where_clause}
        ;
        """
        mapper_df = self.conn.query(query).to_df()
        mapper_dict = mapper_df.set_index("timdex_record_id")["run_id"].to_dict()
        logger.info(
            f"Record-to-run mapper dict created elapsed: {time.perf_counter()-start_time}"
        )
        return mapper_dict

    def _prepare_where_clause_from_dataset_filters(
        self, **filters: Unpack["DatasetFilters"]
    ) -> str:
        """Given keyword filters from DatasetFilters, provide a SQL WHERE clause.

        Note: this implementation of translating TIMDEXDataset DatasetFilters to a single
        SQL WHERE clause is quite naive.  This does the trick for now, supporting filters
        like `source` or `run_date`, but this should be revisited if more robust filtering
        is needed.
        """
        conditions = [f"{column} = '{value}'" for column, value in filters.items()]

        if conditions:
            return f"where {' and '.join(conditions)}"
        return ""

    # DEBUG: direct port from v1 --------------------------------------------------------
    # TODO: revisit these, maybe streamline?

    # DEBUG: development debugging ------------------------------------------------------
    # TODO: handy for now, but need work (e.g. no assert)
    def run_integrity_checks(self):
        """No news is good news."""
        self._check_all_etl_parquet_files_accounted_for()

    def _check_all_etl_parquet_files_accounted_for(self):
        """Check that all ETL parquet files are represented in the metadata layer.

        Either the static metadata db file, or the append deltas, should contain a
        'filename' column value that accounts for all ETL parquet files under
        /data/records.
        """
        assert self.conn.query(
            f"""
            select
                (
                (
                    select count(distinct filename) as metadata_db_files from remote_db.records 
                ) 
                + (
                    select count(distinct filename) as append_delta_files from append_deltas 
                )
                ) = (
                    select count(*) as etl_files from glob('{self.location}/data/records/**/*.parquet')
                );
            """
        ).fetchone()[0]

    def get_data_source_counts(self):
        query = f"""
        select
            (select count(*) from remote_db.records) as metadata_file_rows,
            (select count(*) from append_deltas) as append_delta_rows,
            (select count(*) from records) as records_view_rows,
            (select count(*) from current_records) as current_records_view_rows
        """
        return self.conn.query(query)

    # TODO: sloppy method, but functional
    def merge_append_deltas(self):
        # TODO: make this work for local filesystem too...

        s3_client = S3Client()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = str(Path(temp_dir) / "metadata.duckdb")

            # download metadata file
            s3_client.download_file(self.metadata_database_path, temp_path)

            # attach temp, local DB
            self.conn.execute(f"""ATTACH '{temp_path}' as temp_local_db;""")

            # insert all data from append_deltas
            self.conn.execute(
                """
                insert into temp_local_db.records (
                    select * from append_deltas
                );
                """
            )

            # detach temp, local DB
            self.conn.execute("""DETACH temp_local_db;""")

            # upload back to S3
            s3_client.upload_file(temp_path, self.metadata_database_path)

        # delete append deltas
        s3_client.delete_folder(self.append_deltas_path)

        # TODO: only needed because append_deltas directory is gone...
        self.refresh()


class MetadataDuckDBDatabase:
    """This class represents the physical dataset metadata DuckDB file.

    This class provides the schema for the database and methods to manage it (e.g. create,
    full refresh, etc.).
    """

    def __init__(self, location: str, local_db_path: str):
        self.location = location
        self.local_db_path = local_db_path

    def build(self):
        start_time = time.perf_counter()

        timdex_dataset = self.load_timdex_dataset()

        with duckdb.connect(self.local_db_path) as conn:
            conn.execute(f"""SET threads = 64;""")
            configure_s3_connection(conn)
            self.create_full_dataset_table(conn, timdex_dataset)
            self.create_current_records_view(conn)

        logger.info(
            "Local DuckDB metadata database built, "
            f"elapsed: {time.perf_counter() - start_time}"
        )

        return self.local_db_path

    def load_timdex_dataset(self):
        from timdex_dataset_api import TIMDEXDataset

        timdex_dataset = TIMDEXDataset(self.location)
        timdex_dataset.load()
        return timdex_dataset

    @staticmethod
    def create_full_dataset_table(
        conn: DuckDBPyConnection,
        timdex_dataset: "TIMDEXDataset",
    ) -> None:
        """Create a table of metadata about all records in the parquet dataset.

        While this table will obviously have a high number of rows, the data is small.
        Testing has shown around 20 million records results in 1gb in memory or ~150mb on
        disk.
        """
        start_time = time.perf_counter()
        logger.info("creating table of full dataset metadata")

        parquet_glob_pattern = f"'{timdex_dataset.records_location}/**/*.parquet'"
        query = f"""
        create or replace table records as (
            select
                timdex_record_id,
                source,
                run_date,
                run_type,
                action,
                run_id,
                run_record_offset,
                run_timestamp,
                filename,
            from read_parquet(
                {parquet_glob_pattern},
                 hive_partitioning=true,
                 filename=true
             )
        );
        """
        conn.execute(query)

        row_count = conn.query("""select count(*) from records;""").fetchone()[0]
        logger.info(
            f"'records' table created - rows: {row_count}, "
            f"elapsed: {time.perf_counter() - start_time}"
        )

    @staticmethod
    def create_current_records_view(conn: DuckDBPyConnection) -> None:
        """Create a view of current records.

        This view builds on the table `records`.

        This view includes only the most current version of each record in the dataset.
        Because it includes the `timdex_record_id` and `run_id`, it makes yielding the
        current version of a record via a TIMDEXDataset instance trivial: for any given
        `timdex_record_id` if the `run_id` doesn't match, it's not the current version.
        """
        logger.info("creating view of current records metadata")

        query = """
        create or replace view current_records as
        with ranked_records as (
            select
                r.*,
                row_number() over (
                    partition by r.timdex_record_id
                    order by r.run_timestamp desc
                ) as rn
            from records r
            where r.run_timestamp >= (
                select max(r2.run_timestamp)
                from records r2
                where r2.source = r.source
                and r2.run_type = 'full'
            )
        )
        -- NOTE: important this order matches TIMDEX_DATASET_SCHEMA 
        select
            timdex_record_id,
            source,
            run_date,
            run_type,
            action,
            run_id,
            run_record_offset,
            run_timestamp,
            filename
        from ranked_records
        where rn = 1;
        """
        conn.execute(query)


def configure_s3_connection(conn: DuckDBPyConnection):
    logger.info("configuring S3 connection")

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
            );
            """
        )

    else:
        conn.execute(
            """
            create or replace secret aws_s3_secret (
                type s3,
                provider credential_chain,
                chain 'sso;env;config',
                refresh true
            );
            """
        )
