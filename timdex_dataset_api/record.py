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

    # partition columns
    source: str | None = None
    run_date: str | datetime.datetime | None = None
    run_type: str | None = None
    action: str | None = None
    run_id: str | None = None

    def to_dict(
        self,
        *,
        partition_values: dict[str, str | datetime.datetime] | None = None,
        validate: bool = True,
    ) -> dict:
        """Serialize instance as dictionary, setting partition values if passed."""
        if partition_values:
            for key, value in partition_values.items():
                setattr(self, key, value)
        if validate:
            self.validate()
        return asdict(self)

    def validate(self) -> None:
        """Validate DatasetRecord for writing."""
        # ensure all partition columns are set
        missing_partition_values = [
            field
            for field in ["source", "run_date", "run_type", "action", "run_id"]
            if getattr(self, field) is None
        ]
        if missing_partition_values:
            raise InvalidDatasetRecordError(
                f"Partition values are missing: {', '.join(missing_partition_values)}"
            )
