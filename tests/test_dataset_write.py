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
from timdex_dataset_api.exceptions import InvalidDatasetRecordError
from timdex_dataset_api.record import DatasetRecord


def test_dataset_record_serialization():
    values = {
        "timdex_record_id": "alma:123",
        "source_record": b"<record><title>Hello World.</title></record>",
        "transformed_record": b"""{"title":["Hello World."]}""",
        "source": "libguides",
        "run_date": "2024-12-01",
        "run_type": "full",
        "action": "index",
        "run_id": "abc123",
    }
    dataset_record = DatasetRecord(**values)
    assert dataset_record.to_dict() == values


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


def test_dataset_record_serialization_missing_partition_raise_error():
    values = {
        "timdex_record_id": "alma:123",
        "source_record": b"<record><title>Hello World.</title></record>",
        "transformed_record": b"""{"title":["Hello World."]}""",
        "source": "libguides",
        "run_date": "2024-12-01",
        "run_type": "full",
        "action": "index",
        "run_id": None,  # <------ missing partition here
    }
    dataset_record = DatasetRecord(**values)
    with pytest.raises(
        InvalidDatasetRecordError,
        match="Partition values are missing: run_id",
    ):
        assert dataset_record.to_dict() == values


def test_dataset_write_records_to_new_dataset(new_dataset, sample_records_iter):
    files_written = new_dataset.write(sample_records_iter(10_000))
    assert len(files_written) == 1
    assert os.path.exists(new_dataset.location)

    # load newly created dataset as new TIMDEXDataset instance
    dataset = TIMDEXDataset.load(new_dataset.location)
    assert dataset.row_count == 10_000


def test_dataset_reload_after_write(new_dataset, sample_records_iter):
    files_written = new_dataset.write(sample_records_iter(10_000))
    assert len(files_written) == 1
    assert os.path.exists(new_dataset.location)

    # attempt row count before reload
    with pytest.raises(DatasetNotLoadedError):
        _ = new_dataset.row_count

    # attempt row count after reload
    new_dataset.reload()
    assert new_dataset.row_count == 10_000


def test_dataset_write_default_max_rows_per_file(new_dataset, sample_records_iter):
    """Default is 100k rows per file, therefore writing 200,033 records should result in
    3 files (x2 @ 100k rows, x1 @ 33 rows)."""
    total_records = 200_033

    new_dataset.write(sample_records_iter(total_records))
    new_dataset.reload()

    assert new_dataset.row_count == total_records
    assert len(new_dataset.dataset.files) == math.ceil(total_records / MAX_ROWS_PER_FILE)


def test_dataset_write_record_batches_uses_batch_size(new_dataset, sample_records_iter):
    total_records = 101
    batch_size = 50
    batches = list(
        new_dataset.get_dataset_record_batches(
            sample_records_iter(total_records), batch_size=batch_size
        )
    )
    assert len(batches) == math.ceil(total_records / batch_size)


def test_dataset_write_to_multiple_locations_raise_error(sample_records_iter):
    timdex_dataset = TIMDEXDataset(
        location=["/path/to/records-1.parquet", "/path/to/records-2.parquet"]
    )
    with pytest.raises(
        TypeError,
        match="Dataset location must be the root of a single dataset for writing",
    ):
        timdex_dataset.write(sample_records_iter(10))


def test_dataset_write_mixin_partition_values_used(
    new_dataset, sample_records_iter_without_partitions
):
    partition_values = {
        "source": "alma",
        "run_date": "2024-12-01",
        "run_type": "daily",
        "action": "index",
        "run_id": "000-111-aaa-bbb",
    }
    _written_files = new_dataset.write(
        sample_records_iter_without_partitions(10),
        partition_values=partition_values,
    )
    new_dataset.reload()

    # load as pandas dataframe and assert column values
    df = new_dataset.dataset.to_table().to_pandas()
    row = df.iloc[0]
    assert row.source == partition_values["source"]
    assert row.run_date == datetime.date(2024, 12, 1)
    assert row.run_type == partition_values["run_type"]
    assert row.action == partition_values["action"]
    assert row.action == partition_values["action"]


def test_dataset_write_schema_partitions_correctly_ordered(
    new_dataset, sample_records_iter
):
    written_files = new_dataset.write(
        sample_records_iter(10),
        partition_values={
            "source": "alma",
            "run_date": "2024-12-01",
            "run_type": "daily",
            "run_id": "000-111-aaa-bbb",
            "action": "index",
        },
    )
    file = written_files[0]
    assert (
        "/source=alma/run_date=2024-12-01/run_type=daily"
        "/run_id=000-111-aaa-bbb/action=index/" in file.path
    )


def test_dataset_write_schema_applied_to_dataset(new_dataset, sample_records_iter):
    new_dataset.write(sample_records_iter(10))

    # manually load dataset to confirm schema without TIMDEXDataset projecting schema
    # during load
    dataset = ds.dataset(
        new_dataset.location,
        format="parquet",
        partitioning="hive",
    )

    assert set(dataset.schema.names) == set(TIMDEX_DATASET_SCHEMA.names)


def test_dataset_write_partition_deleted_when_written_to_again(
    new_dataset, sample_records_iter
):
    """This tests the existing_data_behavior="delete_matching" configuration when writing
    to a dataset."""
    partition_values = {
        "source": "alma",
        "run_date": "2024-12-01",
        "run_type": "daily",
        "action": "index",
        "run_id": "000-111-aaa-bbb",
    }

    # perform FIRST write to run_date="2024-12-01"
    written_files_1 = new_dataset.write(
        sample_records_iter(10),
        partition_values=partition_values,
    )

    # assert that files from first write are present at this time
    assert os.path.exists(written_files_1[0].path)

    # perform unrelated write with new run_date to confirm this is untouched during delete
    new_partition_values = partition_values.copy()
    new_partition_values["run_date"] = "2024-12-15"
    new_partition_values["run_id"] = "222-333-ccc-ddd"
    written_files_x = new_dataset.write(
        sample_records_iter(7),
        partition_values=new_partition_values,
    )

    # perform SECOND write to run_date="2024-12-01", expecting this to delete everything
    # under this combination of partitions (i.e. the first write)
    written_files_2 = new_dataset.write(
        sample_records_iter(10),
        partition_values=partition_values,
    )

    new_dataset.reload()

    # assert 17 rows: second write for run_date="2024-12-01" @ 10 rows +
    # run_date="2024-12-15" @ 5 rows
    assert new_dataset.row_count == 17

    # assert that files from first run_date="2024-12-01" are gone, second exist
    # and files from run_date="2024-12-15" also exist
    assert not os.path.exists(written_files_1[0].path)
    assert os.path.exists(written_files_2[0].path)
    assert os.path.exists(written_files_x[0].path)


def test_dataset_write_missing_partitions_raise_error(new_dataset, sample_records_iter):
    missing_partition_values = {
        "source": "libguides",
        "run_date": None,
        "run_type": None,
        "action": None,
        "run_id": None,
    }
    with pytest.raises(InvalidDatasetRecordError, match="Partition values are missing"):
        _ = new_dataset.write(
            sample_records_iter(10),
            partition_values=missing_partition_values,
        )
