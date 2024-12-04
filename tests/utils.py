"""tests/utils.py"""

# ruff: noqa: S311

import random
import uuid
from collections.abc import Iterator

from timdex_dataset_api import DatasetRecord


def generate_sample_records(
    num_records: int,
    timdex_record_id_prefix: str = "alma",
    source: str | None = "alma",
    run_date: str | None = "2024-12-01",
    run_type: str | None = "daily",
    action: str | None = "index",
    run_id: str | None = None,
) -> Iterator[DatasetRecord]:
    """Generate sample DatasetRecords."""
    if not run_id:
        run_id = str(uuid.uuid4())

    for x in range(num_records):
        yield DatasetRecord(
            timdex_record_id=f"{timdex_record_id_prefix}:{x}",
            source_record=b"<record><title>Hello World.</title></record>",
            transformed_record=b"""{"title":["Hello World."]}""",
            source=source,
            run_date=run_date,
            run_type=run_type,
            action=action,
            run_id=run_id,
        )


def generate_sample_records_with_simulated_partitions(
    num_records: int, num_run_ids: int = 4
) -> Iterator[DatasetRecord]:
    """Generate sample DatasetRecords, with simulated sampling of partitions."""
    sources = ["alma", "dspsace", "aspace", "libguides", "gismit", "gisogm"]
    run_dates = ["2024-01-01", "2024-06-15", "2024-12-31"]
    run_types = ["full", "daily"]
    actions = ["index", "delete"]
    run_ids = [str(uuid.uuid4()) for x in range(num_run_ids)]

    records_remaining = num_records
    while records_remaining > 0:
        batch_size = random.randint(1, min(100, records_remaining))
        yield from generate_sample_records(
            num_records=batch_size,
            timdex_record_id_prefix=random.choice(sources),
            source=random.choice(sources),
            run_date=random.choice(run_dates),
            run_type=random.choice(run_types),
            action=random.choice(actions),
            run_id=random.choice(run_ids),
        )
        records_remaining -= batch_size
