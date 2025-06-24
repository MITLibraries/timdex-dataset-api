"""timdex_dataset_api/metadata.py"""

import time
from typing import TYPE_CHECKING, Unpack

import duckdb

from timdex_dataset_api.config import configure_logger

if TYPE_CHECKING:
    from timdex_dataset_api.dataset import DatasetFilters, TIMDEXDataset

logger = configure_logger(__name__)


class TIMDEXDatasetMetadata:
    """Collect and provide access to metadata about the parquet dataset.

    The ETL parquet dataset is essentially parquet files in S3.  This class utilizes
    DuckDB to generate metadata about the parquet dataset, down to the individual record
    layer.  This is somewhat similar to how other data lakes like Apache Iceberg or
    DuckDB DuckLake provide a metadata layer over the stored, large, raw files.

    Because this metadata is somewhat infrequently needed, e.g. only for bulk operations
    or analysis, the architectural decision has been made to pay an initial time penalty
    of crawling the dataset to generate metadata which is then used to dramatically
    speed up and simplify other operations.  In the event this dataset-wide metadata
    is needed more often, it may be worth exploring storing it in S3 alongside the data
    and updating it for each write; very much mirroring other data lake frameworks.
    """

    def __init__(
        self,
        timdex_dataset: "TIMDEXDataset",
        db_path: str = ":memory:",
    ):
        """Initialize TIMDEXDatasetMetadata.

        Args:
            timdex_dataset: The TIMDEX dataset instance to extract metadata from
            db_path: Path to the DuckDB database file. Defaults to ":memory:" for
                in-memory database
        """
        self.timdex_dataset = timdex_dataset
        self.db_path = db_path

        self.conn = self.get_connection()
        self._setup_database()

    @classmethod
    def from_dataset_location(
        cls,
        timdex_dataset_location: str,
        **kwargs: str,
    ) -> "TIMDEXDatasetMetadata":
        """Factory method to init TIMDEXDatasetMetadata from a dataset location.

        This first instantiates and loads a TIMDEXDataset instance, then instantiates this
        class using that.  While this class will likely most commonly be used by
        TIMDEXDataset to limit to current records, it is hoped and expected this dataset
        metadata client will be increasingly useful in its own right, thus this method.

        Args:
            timdex_dataset_location: S3 path or local path to the TIMDEX dataset
            **kwargs: Additional keyword arguments passed to the class constructor,
                such as db_path
        """
        # avoids circular import dependency
        from .dataset import TIMDEXDataset  # noqa: PLC0415

        timdex_dataset = TIMDEXDataset(timdex_dataset_location)
        timdex_dataset.load()
        return cls(timdex_dataset, **kwargs)

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a DuckDB connection to the metadata database."""
        return duckdb.connect(self.db_path)

    def set_database_thread_usage(self, thread_count: int) -> None:
        """Set the number of threads for DuckDB operations."""
        self.conn.execute(f"""SET threads = {thread_count};""")

    def _setup_database(self) -> None:
        """Initialize DuckDB database with AWS credentials and base tables and views."""
        start_time = time.perf_counter()

        # bump threads for high parallelization of lightweight data calls for metadata
        self.set_database_thread_usage(64)

        # setup AWS credentials chain
        self._create_aws_credential_chain()

        # create a table of metadata about all rows in dataset
        self._create_full_dataset_table()

        # create a view for current records
        self._create_current_records_view()

        logger.info(
            f"metadata database setup elapsed: {time.perf_counter()-start_time}, "
            f"path: '{self.db_path}'"
        )

    def _create_aws_credential_chain(self) -> None:
        """Setup AWS credentials chain in database connection.

        https://duckdb.org/docs/stable/core_extensions/aws.html
        """
        logger.info("setting up AWS credentials chain")
        query = """
        create or replace secret secret (
            type s3,
            provider credential_chain,
            chain 'sso;env;config',
            refresh true
        );
        """
        self.conn.execute(query)

    def _create_full_dataset_table(self) -> None:
        """Create a table of metadata about all records in the parquet dataset.

        While this table will obviously have a high number of rows, the data is small.
        Testing has shown around 20 million records results in 1gb in memory or ~150mb on
        disk.
        """
        start_time = time.perf_counter()
        logger.info("creating table of full dataset metadata")

        parquet_glob_pattern = self._prepare_parquet_file_glob_pattern()
        query = f"""
        create or replace table records as (
            select
                timdex_record_id,
                source,
                run_date,
                run_type,
                run_id,
                action,
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
        self.conn.execute(query)

        row_count = self.conn.query("""select count(*) from records;""").fetchone()[0]  # type: ignore[index]
        logger.info(
            f"'records' table created - rows: {row_count}, "
            f"elapsed: {time.perf_counter() - start_time}"
        )

    def _prepare_parquet_file_glob_pattern(self) -> str:
        """Prepare a parquet file glob pattern suitable for DuckDB read_parquet()."""
        if isinstance(self.timdex_dataset.location, list):
            return ",".join([f"'{file}'" for file in self.timdex_dataset.location])

        prefix = self.timdex_dataset.location.removesuffix("/")
        return f"'{prefix}/**/*.parquet'"

    def _create_current_records_view(self) -> None:
        """Create a view of current records.

        This view builds on the table `records`.

        This view includes only the most current version of each record in the dataset.
        Because it includes the `timdex_record_id` and `run_id`, it makes yielding the
        current version of a record via a TIMDEXDataset instance trivial: for any given
        `timdex_record_id` if the `run_id` doesn't match, it's not the current version.
        """
        start_time = time.perf_counter()
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
        select
            timdex_record_id,
            source,
            run_date,
            run_type,
            run_id,
            action,
            run_record_offset,
            run_timestamp,
            filename
        from ranked_records
        where rn = 1;
        """
        self.conn.execute(query)

        row_count = self.conn.query(  # type: ignore[index]
            """select count(*) from current_records;"""
        ).fetchone()[0]
        logger.info(
            f"'current_records' view created - rows: {row_count}, "
            f"elapsed: {time.perf_counter() - start_time}"
        )

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
