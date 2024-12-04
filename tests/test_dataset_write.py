# ruff: noqa: S105, S106, SLF001, PLR2004, PD901, D209, D205

import datetime
import math
import os

import pyarrow.dataset as ds
import pytest

from timdex_dataset_api.dataset import (
    MAX_ROWS_PER_FILE,
    TIMDEX_DATASET_SCHEMA,
    DatasetNotLoadedError,
    TIMDEXDataset,
)
from timdex_dataset_api.record import DatasetRecord


def test_dataset_record_serialization():
    dataset_record = DatasetRecord(
        timdex_record_id="alma:123",
        source_record=b"<record><title>Hello World.</title></record>",
        transformed_record=b"""{"title":["Hello World."]}""",
    )
    assert dataset_record.to_dict() == {
        "timdex_record_id": "alma:123",
        "source_record": b"<record><title>Hello World.</title></record>",
        "transformed_record": b"""{"title":["Hello World."]}""",
        "source": None,
        "run_date": None,
        "run_type": None,
        "action": None,
        "run_id": None,
    }


def test_dataset_record_serialization_with_partition_values_provided():
    dataset_record = DatasetRecord(
        timdex_record_id="alma:123",
        source_record=b"<record><title>Hello World.</title></record>",
        transformed_record=b"""{"title":["Hello World."]}""",
    )
    partition_values = {
        "source": "alma",
        "run_date": "2024-12-01",
        "run_type": "daily",
        "action": "index",
        "run_id": "000-111-aaa-bbb",
    }
    assert dataset_record.to_dict(partition_values=partition_values) == {
        "timdex_record_id": "alma:123",
        "source_record": b"<record><title>Hello World.</title></record>",
        "transformed_record": b"""{"title":["Hello World."]}""",
        "source": "alma",
        "run_date": "2024-12-01",
        "run_type": "daily",
        "action": "index",
        "run_id": "000-111-aaa-bbb",
    }


def test_dataset_write_records_to_new_dataset(new_temp_dataset, small_records_iter):
    files_written = new_temp_dataset.write(small_records_iter(10_000))
    assert len(files_written) == 1
    assert os.path.exists(new_temp_dataset.location)

    # load newly created dataset as new TIMDEXDataset instance
    dataset = TIMDEXDataset.load(new_temp_dataset.location)
    assert dataset.row_count == 10_000


def test_dataset_reload_after_write(new_temp_dataset, small_records_iter):
    files_written = new_temp_dataset.write(small_records_iter(10_000))
    assert len(files_written) == 1
    assert os.path.exists(new_temp_dataset.location)

    # attempt row count before reload
    with pytest.raises(DatasetNotLoadedError):
        _ = new_temp_dataset.row_count

    # attempt row count after reload
    new_temp_dataset.reload()
    assert new_temp_dataset.row_count == 10_000


def test_dataset_write_default_max_rows_per_file(new_temp_dataset, small_records_iter):
    """Default is 100k rows per file, therefore writing 200,033 records should result in
    3 files (x2 @ 100k rows, x1 @ 33 rows)."""
    total_records = 200_033

    new_temp_dataset.write(small_records_iter(total_records))
    new_temp_dataset.reload()

    assert new_temp_dataset.row_count == total_records
    assert len(new_temp_dataset.dataset.files) == math.ceil(
        total_records / MAX_ROWS_PER_FILE
    )


def test_dataset_write_record_batches_uses_batch_size(
    new_temp_dataset, small_records_iter
):
    total_records = 101
    batch_size = 50
    batches = list(
        new_temp_dataset.get_dataset_record_batches(
            small_records_iter(total_records), batch_size=batch_size
        )
    )
    assert len(batches) == math.ceil(total_records / batch_size)


def test_dataset_write_to_multiple_locations_raise_error(small_records_iter):
    timdex_dataset = TIMDEXDataset(
        location=["/path/to/records-1.parquet", "/path/to/records-2.parquet"]
    )
    with pytest.raises(
        TypeError,
        match="Dataset location must be the root of a single dataset for writing",
    ):
        timdex_dataset.write(small_records_iter(10))


def test_dataset_write_mixin_partition_values_used(
    new_temp_dataset, small_records_iter_without_partitions
):
    partition_values = {
        "source": "alma",
        "run_date": "2024-12-01",
        "run_type": "daily",
        "action": "index",
        "run_id": "000-111-aaa-bbb",
    }
    _written_files = new_temp_dataset.write(
        small_records_iter_without_partitions(10),
        partition_values=partition_values,
    )
    new_temp_dataset.reload()

    # load as pandas dataframe and assert column values
    df = new_temp_dataset.dataset.to_table().to_pandas()
    row = df.iloc[0]
    assert row.source == partition_values["source"]
    assert row.run_date == datetime.date(2024, 12, 1)
    assert row.run_type == partition_values["run_type"]
    assert row.action == partition_values["action"]
    assert row.action == partition_values["action"]


def test_dataset_write_schema_partitions_correctly_ordered(
    new_temp_dataset, small_records_iter
):
    written_files = new_temp_dataset.write(
        small_records_iter(10),
        partition_values={
            "source": "alma",
            "run_date": "2024-12-01",
            "run_type": "daily",
            "action": "index",
            "run_id": "000-111-aaa-bbb",
        },
    )
    file = written_files[0]
    assert (
        "/source=alma/run_date=2024-12-01/run_type=daily"
        "/action=index/run_id=000-111-aaa-bbb" in file.path
    )


def test_dataset_write_schema_applied_to_dataset(new_temp_dataset, small_records_iter):
    new_temp_dataset.write(small_records_iter(10))

    # manually load dataset to confirm schema without TIMDEXDataset projecting schema
    # during load
    dataset = ds.dataset(
        new_temp_dataset.location,
        format="parquet",
        partitioning="hive",
    )

    assert set(dataset.schema.names) == set(TIMDEX_DATASET_SCHEMA.names)
