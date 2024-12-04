"""timdex_dataset_api/record.py"""

import datetime
from dataclasses import asdict, dataclass


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
        partition_values: dict[str, str | datetime.datetime] | None = None,
    ) -> dict:
        """Serialize instance as dictionary, setting partition values if passed."""
        if partition_values:
            for key, value in partition_values.items():
                setattr(self, key, value)
        return asdict(self)
