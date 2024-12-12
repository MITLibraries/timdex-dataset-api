"""timdex_dataset_api/record.py"""

from datetime import UTC, datetime

from attrs import asdict, define, field


def strict_date_parse(date_string: str) -> datetime:
    return datetime.strptime(date_string, "%Y-%m-%d").astimezone(UTC)


@define
class DatasetRecord:
    """Container for single dataset record.

    An iterator of these are passed to the TIMDEXDataset.write() method, where they are
    first serialized into dictionaries, and then grouped into pyarrow.RecordBatches for
    writing.
    """

    # primary columns
    timdex_record_id: str = field()
    source_record: bytes = field()
    transformed_record: bytes = field()
    source: str = field()
    run_date: datetime = field(converter=strict_date_parse)
    run_type: str = field()
    run_id: str = field()
    action: str = field()

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
