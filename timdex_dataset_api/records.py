"""timdex_dataset_api/records.py"""

import json
from collections.abc import Iterator
from datetime import date, datetime
from typing import ClassVar, TypedDict, Unpack

import attrs
import pyarrow as pa
from attrs import asdict, define, field

from timdex_dataset_api.data_source import TIMDEXDataSource, ValidTable
from timdex_dataset_api.metadata import CurrentMetadataViewSpec
from timdex_dataset_api.utils import (
    datetime_iso_parse,
    strict_date_parse,
)


class RecordsFilters(TypedDict, total=False):
    timdex_record_id: str | list[str] | None
    source: str | list[str] | None
    run_date: str | date | list[str | date] | None
    run_type: str | list[str] | None
    action: str | list[str] | None
    run_id: str | list[str] | None
    run_record_offset: int | list[int] | None
    run_timestamp: str | datetime | list[str | datetime] | None


@define
class DatasetRecord:
    """Container for single dataset record.

    An iterator of these are passed to the TIMDEXRecords.write() method, where they are
    first serialized into dictionaries, and then grouped into pyarrow.RecordBatches for
    writing.
    """

    timdex_record_id: str = field()
    source_record: bytes = field()
    transformed_record: bytes = field()
    source: str = field()
    run_date: date = field(converter=strict_date_parse)
    run_type: str = field()
    action: str = field()
    run_id: str = field()
    run_timestamp: datetime = field(
        converter=datetime_iso_parse,
        default=attrs.Factory(
            lambda self: self.run_date.isoformat(),
            takes_self=True,
        ),
    )
    run_record_offset: int = field(default=None)

    @property
    def year(self) -> str:
        return self.run_date.strftime("%Y")

    @property
    def month(self) -> str:
        return self.run_date.strftime("%m")

    @property
    def day(self) -> str:
        return self.run_date.strftime("%d")

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


class TIMDEXRecords(TIMDEXDataSource):
    """Class to handle records in the TIMDEXDataset."""

    NAME: ClassVar[str] = "records"

    SCHEMA: ClassVar[pa.Schema] = pa.schema(
        (
            pa.field("timdex_record_id", pa.string()),
            pa.field("source_record", pa.binary()),
            pa.field("transformed_record", pa.binary()),
            pa.field("source", pa.string()),
            pa.field("run_date", pa.date32()),
            pa.field("run_type", pa.string()),
            pa.field("action", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("run_record_offset", pa.int32()),
            pa.field("year", pa.string()),
            pa.field("month", pa.string()),
            pa.field("day", pa.string()),
            pa.field("run_timestamp", pa.timestamp("us", tz="UTC")),
        )
    )

    DATA_COLUMNS: ClassVar[list[str]] = [
        "source_record",
        "transformed_record",
    ]

    DATA_PATH_SEGMENT: ClassVar[str] = "data/records"

    PREJOIN_RECORDS: ClassVar[bool] = False

    VALID_TABLES: ClassVar[list[ValidTable]] = [
        ValidTable(
            name="records",
            description="All record versions across all runs.",
        ),
        ValidTable(
            name="current_records",
            description=(
                "One row per (source, timdex_record_id) representing the"
                " most recent version of each record since the last full"
                " run."
            ),
        ),
    ]

    CURRENT_METADATA_VIEW_QUERY: ClassVar[str] = """
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

    CURRENT_METADATA_VIEW_SPEC: ClassVar[CurrentMetadataViewSpec] = (
        CurrentMetadataViewSpec(
            name="current_records",
            query_sql=CURRENT_METADATA_VIEW_QUERY,
            required_metadata_tables=["records"],
            preload_setting_attribute="preload_current_records",
        )
    )

    CURRENT_VIEW_SPECS: ClassVar[list[CurrentMetadataViewSpec]] = [
        CURRENT_METADATA_VIEW_SPEC
    ]

    @property
    def data_records_root(self) -> str:
        return self.data_root

    def read_transformed_records_iter(
        self,
        table: str = "records",
        limit: int | None = None,
        where: str | None = None,
        **filters: Unpack[RecordsFilters],
    ) -> Iterator[dict]:
        """Custom read method to yield parsed transformed TIMDEX JSON records only."""
        for record_dict in self.read_dicts_iter(
            table=table,
            columns=["transformed_record"],
            limit=limit,
            where=where,
            **filters,
        ):
            if transformed_record := record_dict["transformed_record"]:
                yield json.loads(transformed_record)
