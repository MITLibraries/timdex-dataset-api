"""timdex_dataset_api/record.py"""

import datetime
from dataclasses import asdict, dataclass

from timdex_dataset_api.exceptions import InvalidDatasetRecordError


@dataclass
class DatasetRecord:
    """Container for single dataset record.

    An iterator of these are passed to the TIMDEXDataset.write() method, where they are
    first serialized into dictionaries, and then grouped into pyarrow.RecordBatches for
    writing.
    """

    # primary columns
    timdex_record_id: str
    source_record: bytes
    transformed_record: bytes
    source: str
    run_date: str | datetime.datetime
    run_type: str
    run_id: str
    action: str

    # partition columns
    year: str | None = None
    month: str | None = None
    day: str | None = None

    def __post_init__(self) -> None:
        """Post init method to derive partition values from self.run_date"""
        run_date = self.run_date

        if isinstance(run_date, str):
            try:
                run_date = datetime.datetime.strptime(run_date, "%Y-%m-%d").astimezone(
                    datetime.UTC
                )
            except ValueError as exception:
                raise InvalidDatasetRecordError(
                    "Cannot parse partition values [year, month, date] from invalid 'run-date' string."  # noqa: E501
                ) from exception

        self.year = run_date.strftime("%Y")
        self.month = run_date.strftime("%m")
        self.day = run_date.strftime("%d")

    def to_dict(
        self,
        *,
        validate: bool = True,
    ) -> dict:
        """Serialize instance as dictionary."""
        if validate:
            self.validate()

        return asdict(self)

    def validate(self) -> None:
        """Validate DatasetRecord for writing."""
        # ensure all partition columns are set
        missing_partition_values = [
            field for field in ["year", "month", "day"] if getattr(self, field) is None
        ]
        if missing_partition_values:
            raise InvalidDatasetRecordError(
                f"Partition values are missing: {', '.join(missing_partition_values)}"
            )
