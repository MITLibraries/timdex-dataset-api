import hashlib
from datetime import UTC, datetime
from typing import ClassVar

import attrs
import pyarrow as pa
from attrs import asdict, define, field

from timdex_dataset_api.data_type import DataTypeTableConfig, TIMDEXDataType
from timdex_dataset_api.utils import datetime_iso_parse


@define
class DatasetFulltext:
    """Container for single record fulltext.

    Fields:
        timdex_record_id: Fields (timdex_record_id, run_id, run_record_offset) combine to
            form a composite key that points to a single, distinct record version in the
            records data.
        run_id: ...
        run_record_offset: ...
        fulltext_timestamp: Timestamp when fulltext was extracted / added to TIMDEX
            dataset
        fulltext_md5: MD5 checksum for the fulltext payload. If omitted, generated
            from fulltext when serializing.
        fulltext: Fulltext for the record.
    """

    timdex_record_id: str = field()
    run_id: str = field()
    run_record_offset: int = field()

    fulltext_timestamp: datetime = field(  # type: ignore[assignment]
        converter=datetime_iso_parse,
        default=attrs.Factory(lambda: datetime.now(tz=UTC).isoformat()),
    )
    fulltext_md5: str | None = field(default=None)
    fulltext: bytes | None = field(default=None)

    @property
    def year(self) -> str:
        return self.fulltext_timestamp.strftime("%Y")

    @property
    def month(self) -> str:
        return self.fulltext_timestamp.strftime("%m")

    @property
    def day(self) -> str:
        return self.fulltext_timestamp.strftime("%d")

    def to_dict(
        self,
    ) -> dict:
        """Serialize instance as dictionary."""
        return {
            **asdict(self),
            "fulltext_md5": self._fulltext_md5(),
            "year": self.year,
            "month": self.month,
            "day": self.day,
        }

    def _fulltext_md5(self) -> str | None:
        """Calculate an MD5 hash on the fulltext bytes if not explicitly set already."""
        if self.fulltext_md5:
            return self.fulltext_md5
        if self.fulltext is None:
            return None
        return hashlib.md5(self.fulltext, usedforsecurity=False).hexdigest()


class TIMDEXFulltexts(TIMDEXDataType):
    """Class to handle record fulltexts in the TIMDEXDataset."""

    NAME: ClassVar[str] = "fulltexts"

    SCHEMA: ClassVar[pa.Schema] = pa.schema(
        (
            pa.field("timdex_record_id", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("run_record_offset", pa.int32()),
            pa.field("fulltext_timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("fulltext_md5", pa.string()),
            pa.field("fulltext", pa.binary()),
            pa.field("year", pa.string()),
            pa.field("month", pa.string()),
            pa.field("day", pa.string()),
        )
    )

    DATA_COLUMNS: ClassVar[list[str]] = ["fulltext"]

    DATA_PATH: ClassVar[str] = "data/fulltexts"

    CURRENT_METADATA_VIEW_QUERY: ClassVar[str] = """
        with
            -- CTE of fulltexts attached to current record versions only
            cf_current_record_fulltexts as
            (
                select
                    f.*
                from metadata.fulltexts f
                join metadata.current_records r using (
                    source,
                    timdex_record_id,
                    run_id,
                    run_record_offset
                )
            ),

            -- CTE of current-record fulltexts ranked by fulltext recency
            cf_ranked_fulltexts as
            (
                select
                    f.*,
                    row_number() over (
                        partition by
                            f.timdex_record_id
                        order by
                            f.fulltext_timestamp desc nulls last,
                            f.filename desc nulls last
                    ) as rn
                from cf_current_record_fulltexts f
            )
        -- final select for current fulltexts (rn = 1)
        select
            * exclude (rn)
        from cf_ranked_fulltexts
        where rn = 1
    """

    CURRENT_RUN_METADATA_VIEW_QUERY: ClassVar[str] = """
        with
            -- CTE of fulltexts ranked by fulltext recency within a run
            -- keep run_timestamp because the same run_id can be written more than once
            -- keep run_record_offset as an intra-run tie-break when the same logical
            -- record appears more than once within a run
            crcf_ranked_fulltexts as
            (
                select
                    f.*,
                    row_number() over (
                        partition by
                            f.timdex_record_id,
                            f.run_id
                        order by
                            f.run_timestamp desc nulls last,
                            f.fulltext_timestamp desc nulls last,
                            f.run_record_offset desc nulls last,
                            f.filename desc nulls last
                    ) as rn
                from metadata.fulltexts f
            )
        -- final select for current run fulltexts (rn = 1)
        select
            * exclude (rn)
        from crcf_ranked_fulltexts
        where rn = 1
    """

    TABLES: ClassVar[list[DataTypeTableConfig]] = [
        DataTypeTableConfig(
            name="fulltexts",
            description="All fulltext versions across all runs.",
            kind="base",
        ),
        DataTypeTableConfig(
            name="current_fulltexts",
            description=(
                "One row per timdex_record_id representing the most recent fulltext "
                "for each current record."
            ),
            kind="custom",
            query_sql=CURRENT_METADATA_VIEW_QUERY,
            required_metadata_tables=["fulltexts", "current_records"],
        ),
        DataTypeTableConfig(
            name="current_run_fulltexts",
            description=(
                "One row per (timdex_record_id, run_id) representing the most "
                "recent fulltext within each run, regardless of whether the "
                "record is current."
            ),
            kind="custom",
            query_sql=CURRENT_RUN_METADATA_VIEW_QUERY,
            required_metadata_tables=["fulltexts", "records"],
        ),
    ]
