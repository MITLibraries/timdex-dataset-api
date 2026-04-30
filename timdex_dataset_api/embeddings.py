from datetime import UTC, datetime
from typing import ClassVar

import attrs
import pyarrow as pa
from attrs import asdict, define, field

from timdex_dataset_api.data_type import DataTypeTableConfig, TIMDEXDataType
from timdex_dataset_api.utils import datetime_iso_parse


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
        embedding_timestamp: Timestamp when embedding was created
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
    embedding_timestamp: datetime = field(  # type: ignore[assignment]
        converter=datetime_iso_parse,
        default=attrs.Factory(lambda: datetime.now(tz=UTC).isoformat()),
    )
    embedding_vector: list[float] | None = field(default=None)
    embedding_object: bytes | None = field(default=None)

    @property
    def year(self) -> str:
        return self.embedding_timestamp.strftime("%Y")

    @property
    def month(self) -> str:
        return self.embedding_timestamp.strftime("%m")

    @property
    def day(self) -> str:
        return self.embedding_timestamp.strftime("%d")

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


class TIMDEXEmbeddings(TIMDEXDataType):
    """Class to handle record embeddings in the TIMDEXDataset."""

    NAME: ClassVar[str] = "embeddings"

    SCHEMA: ClassVar[pa.Schema] = pa.schema(
        (
            pa.field("timdex_record_id", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("run_record_offset", pa.int32()),
            pa.field("embedding_timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("embedding_model", pa.string()),
            pa.field("embedding_strategy", pa.string()),
            pa.field("embedding_vector", pa.list_(pa.float32())),
            pa.field("embedding_object", pa.binary()),
            pa.field("year", pa.string()),
            pa.field("month", pa.string()),
            pa.field("day", pa.string()),
        )
    )

    DATA_COLUMNS: ClassVar[list[str]] = [
        "embedding_vector",
        "embedding_object",
    ]

    DATA_PATH: ClassVar[str] = "data/embeddings"

    CURRENT_METADATA_VIEW_QUERY: ClassVar[str] = """
        with
            -- CTE of embeddings attached to current record versions only
            ce_current_record_embeddings as
            (
                select
                    e.*
                from metadata.embeddings e
                join metadata.current_records r using (
                    source,
                    timdex_record_id,
                    run_id,
                    run_record_offset
                )
            ),

            -- CTE of current-record embeddings ranked by embedding recency
            ce_ranked_embeddings as
            (
                select
                    e.*,
                    row_number() over (
                        partition by
                            e.timdex_record_id,
                            e.embedding_model,
                            e.embedding_strategy
                        order by
                            e.embedding_timestamp desc nulls last,
                            e.filename desc nulls last
                    ) as rn
                from ce_current_record_embeddings e
            )
        -- final select for current embeddings (rn = 1)
        select
            * exclude (rn)
        from ce_ranked_embeddings
        where rn = 1
    """

    CURRENT_RUN_METADATA_VIEW_QUERY: ClassVar[str] = """
        with
            -- CTE of embeddings ranked by embedding recency within a run and family
            -- keep run_timestamp because the same run_id can be written more than once
            -- keep run_record_offset as an intra-run tie-break when the same logical
            -- record appears more than once within a run
            crce_ranked_embeddings as
            (
                select
                    e.*,
                    row_number() over (
                        partition by
                            e.timdex_record_id,
                            e.run_id,
                            e.embedding_model,
                            e.embedding_strategy
                        order by
                            e.run_timestamp desc nulls last,
                            e.embedding_timestamp desc nulls last,
                            e.run_record_offset desc nulls last,
                            e.filename desc nulls last
                    ) as rn
                from metadata.embeddings e
            )
        -- final select for current run embeddings (rn = 1)
        select
            * exclude (rn)
        from crce_ranked_embeddings
        where rn = 1
    """

    TABLES: ClassVar[list[DataTypeTableConfig]] = [
        DataTypeTableConfig(
            name="embeddings",
            description="All embedding versions across all runs.",
            kind="base",
        ),
        DataTypeTableConfig(
            name="current_embeddings",
            description=(
                "One row per (timdex_record_id, embedding_model, "
                "embedding_strategy) representing the most recent embedding "
                "for each current record."
            ),
            kind="custom",
            query_sql=CURRENT_METADATA_VIEW_QUERY,
            required_metadata_tables=["embeddings", "current_records"],
        ),
        DataTypeTableConfig(
            name="current_run_embeddings",
            description=(
                "One row per (timdex_record_id, run_id, embedding_model, "
                "embedding_strategy) representing the most recent embedding "
                "within each run, regardless of whether the record is current."
            ),
            kind="custom",
            query_sql=CURRENT_RUN_METADATA_VIEW_QUERY,
            required_metadata_tables=["embeddings", "records"],
        ),
    ]
