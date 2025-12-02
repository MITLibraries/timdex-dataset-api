import itertools
import logging
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict, Unpack, cast

import attrs
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from attrs import asdict, define, field
from duckdb import DuckDBPyConnection
from duckdb import IOException as DuckDBIOException
from duckdb_engine import Dialect as DuckDBDialect
from sqlalchemy import Table, select, text
from sqlalchemy.types import ARRAY, FLOAT

from timdex_dataset_api.record import datetime_iso_parse
from timdex_dataset_api.utils import build_filter_expr_sa, sa_reflect_duckdb_conn

if TYPE_CHECKING:
    from timdex_dataset_api import TIMDEXDataset


logger = logging.getLogger(__name__)

TIMDEX_DATASET_EMBEDDINGS_SCHEMA = pa.schema(
    (
        pa.field("timdex_record_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("run_record_offset", pa.int32()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("embedding_model", pa.string()),
        pa.field("embedding_strategy", pa.string()),
        pa.field("embedding_vector", pa.list_(pa.float32())),
        pa.field("embedding_object", pa.binary()),
        pa.field("year", pa.string()),
        pa.field("month", pa.string()),
        pa.field("day", pa.string()),
    )
)


class EmbeddingsFilters(TypedDict, total=False):
    timdex_record_id: str
    run_id: str
    run_record_offset: int
    timestamp: str | datetime
    embedding_model: str
    embedding_strategy: str


@define
class DatasetEmbedding:
    """Container for single record embedding.

    Fields:
        timdex_record_id: Fields (timdex_record_id, run_id, run_record_offset) combine to
            form a composite key that points to a single, distinct record version in the
            records data.
        run_id: ...
        run_record_offset: ...
        embedding_model: Embedding model name, e.g. HuggingFace URI
        embedding_strategy: Strategy used to create embedding
            - this correlates to a transformation strategy in the timdex-embeddings CLI
            application, e.g. "full_record"
        timestamp: Timestamp when embedding was created
        embedding_vector: Numerical vector representation of embedding
            - preferred form for storing embedding as a numerical array
        embedding_object: Object representation of the embedding
            - example: {token:weight, ...} representation for sparse vector
            - flexible enough to hold other representations
    """

    timdex_record_id: str = field()
    run_id: str = field()
    run_record_offset: int = field()
    embedding_model: str = field()
    embedding_strategy: str = field()
    timestamp: datetime = field(  # type: ignore[assignment]
        converter=datetime_iso_parse,
        default=attrs.Factory(lambda: datetime.now(tz=UTC).isoformat()),
    )
    embedding_vector: list[float] | None = field(default=None)
    embedding_object: bytes | None = field(default=None)

    @property
    def year(self) -> str:
        return self.timestamp.strftime("%Y")

    @property
    def month(self) -> str:
        return self.timestamp.strftime("%m")

    @property
    def day(self) -> str:
        return self.timestamp.strftime("%d")

    def to_dict(
        self,
    ) -> dict:
        """Serialize instance as dictionary."""
        return {
            **asdict(self),
            "year": self.year,
            "month": self.month,
            "day": self.day,
        }


class TIMDEXEmbeddings:

    def __init__(self, timdex_dataset: "TIMDEXDataset"):
        """Init TIMDEXEmbeddings.

        Class to handle the writing and readings of embeddings associated with TIMDEX
        records.

        Args:
            - timdex_dataset: instance of TIMDEXDataset
        """
        self.timdex_dataset = timdex_dataset

        self.schema = TIMDEX_DATASET_EMBEDDINGS_SCHEMA
        self.partition_columns = ["year", "month", "day"]

        # DuckDB context
        self.conn = self.setup_duckdb_context()
        self._sa_metadata_data_schema = sa_reflect_duckdb_conn(self.conn, schema="data")

        # resolve data type for 'embedding_vector' column
        if "data.embeddings" in self._sa_metadata_data_schema.tables:
            sa_metadata_data_embeddings_table = self._sa_metadata_data_schema.tables[
                "data.embeddings"
            ]
            sa_metadata_data_embeddings_table.c.embedding_vector.type = ARRAY(FLOAT)

    @property
    def data_embeddings_root(self) -> str:
        return f"{self.timdex_dataset.location.removesuffix('/')}/data/embeddings"

    def get_sa_table(self, table: str) -> Table:
        """Get SQLAlchemy Table from reflected SQLAlchemy metadata."""
        schema_table = f"data.{table}"
        if schema_table not in self._sa_metadata_data_schema.tables:
            raise ValueError(f"Could not find table '{table}' in DuckDB schema 'data'.")
        return self._sa_metadata_data_schema.tables[schema_table]

    def setup_duckdb_context(self) -> DuckDBPyConnection:
        """Create a DuckDB connection for embeddings query and retrieval.

        This method extends TIMDEXDatasetMetadata's pre-existing DuckDB connection
        (via the attached TIMDEXDataset), creating views in the 'data' schema.
        """
        start_time = time.perf_counter()

        conn = self.timdex_dataset.conn

        try:
            self._create_embeddings_view(conn)
            self._create_current_embeddings_view(conn)
            self._create_current_run_embeddings_view(conn)
        except DuckDBIOException:
            logger.warning("No embeddings found")
        except Exception as exception:  # noqa: BLE001
            logger.warning(f"An error occurred while creating views: {exception}")

        logger.debug(
            "DuckDB context created for TIMDEXEmbeddings, "
            f"{round(time.perf_counter()-start_time,2)}s"
        )
        return conn

    def _create_embeddings_view(self, conn: DuckDBPyConnection) -> None:
        """Create a view that projects over embeddings parquet files."""
        logger.debug("creating view data.embeddings")

        conn.execute(
            f"""
            create or replace view data.embeddings as
            (
                select *
                from read_parquet(
                    '{self.data_embeddings_root}/**/*.parquet',
                    hive_partitioning=true,
                    filename=true
                )
            );
            """
        )

    def _create_current_embeddings_view(self, conn: DuckDBPyConnection) -> None:
        """Create a view of current embedding records.

        This builds on the 'data.embeddings' view. This view includes only
        the most current version of each embedding grouped by
        [timdex_record_id, embedding_strategy].
        """
        logger.debug("creating view data.current_embeddings")

        # SQL for the current records logic (CTEs)
        conn.execute(
            """
            create or replace view data.current_embeddings as
            (
                with
                    -- CTE of embeddings ranked by timestamp
                    ce_ranked_embeddings as
                    (
                        select
                            *,
                            row_number() over (
                                partition by timdex_record_id, embedding_strategy
                                order by
                                    timestamp desc nulls last,
                                    run_record_offset desc nulls last
                            ) as rn
                        from data.embeddings
                    )
                -- final select for current records (rn = 1)
                select
                    * exclude (rn)
                from ce_ranked_embeddings
                where rn = 1
            );
            """
        )

    def _create_current_run_embeddings_view(self, conn: DuckDBPyConnection) -> None:
        """Create a view of current embedding records per run.

        This builds on the 'data.embeddings' view. This view includes only
        the most current version of each embedding per run grouped by
        [timdex_record_id, run_id, embedding_strategy,].
        """
        logger.debug("creating view data.current_run_embeddings")

        # SQL for the current records logic (CTEs)
        conn.execute(
            """
            create or replace view data.current_run_embeddings as
            (
                with
                    -- CTE of embeddings ranked by timestamp
                    ce_ranked_embeddings as
                    (
                        select
                            *,
                            row_number() over (
                                partition by timdex_record_id, run_id, embedding_strategy
                                order by
                                    timestamp desc nulls last,
                                    run_id desc nulls last,
                                    run_record_offset desc nulls last
                            ) as rn
                        from data.embeddings
                    )
                -- final select for current records (rn = 1)
                select
                    * exclude (rn)
                from ce_ranked_embeddings
                where rn = 1
            );
            """
        )

    def write(
        self,
        embeddings_iter: Iterator[DatasetEmbedding],
        *,
        use_threads: bool = True,
    ) -> list[ds.WrittenFile]:
        """Write embeddings as parquet files to /data/embeddings.

        Approach is similar to TIMDEXDataset.write() for Records:
            - use self.data_embeddings_root for location of embeddings parquet files
            - use pyarrow Dataset to write rows
        """
        start_time = time.perf_counter()
        written_files: list[ds.WrittenFile] = []

        filesystem, path = self.timdex_dataset.parse_location(self.data_embeddings_root)

        embedding_batches_iter = self.create_embedding_batches(embeddings_iter)
        ds.write_dataset(
            embedding_batches_iter,
            base_dir=path,
            basename_template="%s-{i}.parquet" % (str(uuid.uuid4())),  # noqa: UP031
            existing_data_behavior="overwrite_or_ignore",
            filesystem=filesystem,
            file_visitor=lambda written_file: written_files.append(written_file),  # type: ignore[arg-type]
            format="parquet",
            max_open_files=500,
            max_rows_per_file=self.timdex_dataset.config.max_rows_per_file,
            max_rows_per_group=self.timdex_dataset.config.max_rows_per_group,
            partitioning=self.partition_columns,
            partitioning_flavor="hive",
            schema=self.schema,
            use_threads=use_threads,
        )

        self.log_write_statistics(start_time, written_files)

        return written_files

    def create_embedding_batches(
        self, embeddings_iter: Iterator["DatasetEmbedding"]
    ) -> Iterator[pa.RecordBatch]:
        for i, embedding_batch in enumerate(
            itertools.batched(
                embeddings_iter, self.timdex_dataset.config.write_batch_size
            )
        ):
            embedding_dicts = [embedding.to_dict() for embedding in embedding_batch]
            batch = pa.RecordBatch.from_pylist(embedding_dicts)
            logger.debug(f"Yielding batch {i + 1} for dataset writing.")
            yield batch

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

    def read_batches_iter(
        self,
        table: str = "embeddings",
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[EmbeddingsFilters],
    ) -> Iterator[pa.RecordBatch]:
        """Yield ETL records as pyarrow.RecordBatches.

        This is the base read method. All read methods use this for streaming
        batches of records. This method relies on DuckDB to project over all
        embeddings parquet files (i.e., no "metadata layer") and filter data.
        """
        start_time = time.perf_counter()

        data_query = self._build_query(
            table,
            columns or TIMDEX_DATASET_EMBEDDINGS_SCHEMA.names,
            limit,
            where,
            **filters,
        )
        cursor = self.conn.execute(data_query)
        yield from cursor.fetch_record_batch(
            rows_per_batch=self.timdex_dataset.config.read_batch_size
        )

        logger.debug(f"read() elapsed: {round(time.perf_counter()-start_time, 2)}s")

    def _build_query(
        self,
        table: str = "embeddings",
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[EmbeddingsFilters],
    ) -> str:
        """Build SQL query using SQLAlchemy.

        The method returns a SQL query string, which SQLAlchemy executes to
        fetch results.
        """
        sa_table = self.get_sa_table(table)

        # create SQL statement object
        # select named columns
        if columns:
            stmt = select(*sa_table.c[*columns]).select_from(sa_table)
        else:
            stmt = select(sa_table).select_from(sa_table)

        # filter expressions from key/value filters (may return None)
        filter_expr = build_filter_expr_sa(sa_table, **cast("dict", filters))
        if filter_expr is not None:
            stmt = stmt.where(filter_expr)

        # explicit raw WHERE string
        if where is not None and where.strip():
            stmt = stmt.where(text(where))

        # apply limit if present
        if limit:
            stmt = stmt.limit(limit)

        # using DuckDB dialect, compile to SQL string
        compiled = stmt.compile(
            dialect=DuckDBDialect(),
            compile_kwargs={"literal_binds": True},
        )
        return str(compiled)

    def read_dataframes_iter(
        self,
        table: str = "embeddings",
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[EmbeddingsFilters],
    ) -> Iterator[pd.DataFrame]:
        for record_batch in self.read_batches_iter(
            table=table, columns=columns, limit=limit, where=where, **filters
        ):
            yield record_batch.to_pandas()

    def read_dataframe(
        self,
        table: str = "embeddings",
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[EmbeddingsFilters],
    ) -> pd.DataFrame | None:
        df_batches = [
            record_batch.to_pandas()
            for record_batch in self.read_batches_iter(
                table=table,
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
        table: str = "embeddings",
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[EmbeddingsFilters],
    ) -> Iterator[dict]:
        for record_batch in self.read_batches_iter(
            table=table,
            columns=columns,
            limit=limit,
            where=where,
            **filters,
        ):
            yield from record_batch.to_pylist()
