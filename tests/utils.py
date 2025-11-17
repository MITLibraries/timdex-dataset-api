"""tests/utils.py"""

# ruff: noqa: S311

import json
import random
import uuid
from collections.abc import Iterator

from timdex_dataset_api import DatasetRecord
from timdex_dataset_api.embeddings import DatasetEmbedding


def generate_sample_records(
    num_records: int,
    source: str | None = "alma",
    run_date: str | None = "2024-12-01",
    run_type: str | None = "daily",
    action: str | None = "index",
    run_id: str | None = None,
    run_timestamp: str | None = None,
) -> Iterator[DatasetRecord]:
    """Generate sample DatasetRecords."""
    if not run_id:
        run_id = str(uuid.uuid4())

    for x in range(num_records):
        yield DatasetRecord(
            timdex_record_id=f"{source}:{x}",
            source_record=b"<record><title>Hello World.</title></record>",
            transformed_record=b"""{"title":["Hello World."]}""",
            source=source,
            run_date=run_date,
            run_type=run_type,
            action=action,
            run_id=run_id,
            run_record_offset=x,
            run_timestamp=run_timestamp or run_date,
        )


def generate_sample_records_with_simulated_partitions(
    num_records: int, num_run_ids: int = 4
) -> Iterator[DatasetRecord]:
    """Generate sample DatasetRecords, with simulated sampling of partitions."""
    sources = ["alma", "dspsace", "aspace", "libguides", "gismit", "gisogm"]
    run_dates = ["2024-01-01", "2024-06-15", "2024-12-31"]
    run_types = ["full", "daily"]
    actions = ["index", "delete"]

    records_remaining = num_records
    while records_remaining > 0:
        batch_size = random.randint(1, min(100, records_remaining))
        source = random.choice(sources)
        yield from generate_sample_records(
            num_records=batch_size,
            source=source,
            run_date=random.choice(run_dates),
            run_type=random.choice(run_types),
            action=random.choice(actions),
        )
        records_remaining -= batch_size


def generate_sample_embeddings(
    num_embeddings: int,
    source: str | None = "alma",
    embedding_model: str | None = "super-org/amazing-model",
    embedding_strategy: str | None = "full_record",
    run_id: str | None = None,
    timestamp: str | None = "2024-12-01T00:00:00+00:00",
) -> Iterator[DatasetEmbedding]:
    """Generate sample DatasetEmbeddings."""
    if not run_id:
        run_id = str(uuid.uuid4())

    for x in range(num_embeddings):
        embedding_vector = [random.random() for _ in range(768)]
        embedding_object = json.dumps(
            {
                "token1": 0.1,
                "token2": 0.2,
                "token3": 0.3,
            }
        ).encode()

        yield DatasetEmbedding(
            timdex_record_id=f"{source}:{x}",
            run_id=run_id,
            run_record_offset=x,
            embedding_model=embedding_model,
            embedding_strategy=embedding_strategy,
            timestamp=timestamp,
            embedding_vector=embedding_vector,
            embedding_object=embedding_object,
        )
