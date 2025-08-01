# ruff: noqa: PLR2004, D209, D205
import math
import os
from unittest.mock import patch

import pyarrow.dataset as ds
import pytest

from tests.utils import generate_sample_records
from timdex_dataset_api.dataset import (
    TIMDEX_DATASET_SCHEMA,
    TIMDEXDataset,
)


def test_dataset_write_records_to_new_local_dataset(
    new_local_dataset, sample_records_iter
):
    written_files = new_local_dataset.write(sample_records_iter(10_000))
    new_local_dataset.load()

    assert len(written_files) == 1
    assert os.path.exists(new_local_dataset.location)
    assert new_local_dataset.row_count == 10_000


def test_dataset_write_default_max_rows_per_file(new_local_dataset, sample_records_iter):
    """Default is 100k rows per file, therefore writing 200,033 records should result in
    3 files (x2 @ 100k rows, x1 @ 33 rows)."""
    default_max_rows_per_file = new_local_dataset.config.max_rows_per_file
    total_records = 200_033

    new_local_dataset.write(sample_records_iter(total_records))
    new_local_dataset.load()

    assert new_local_dataset.row_count == total_records
    assert len(new_local_dataset.dataset.files) == math.ceil(
        total_records / default_max_rows_per_file
    )


def test_dataset_write_record_batches_uses_batch_size(
    new_local_dataset, sample_records_iter
):
    total_records = 101
    new_local_dataset.config.write_batch_size = 50
    batches = list(
        new_local_dataset.create_record_batches(sample_records_iter(total_records))
    )
    assert len(batches) == math.ceil(
        total_records / new_local_dataset.config.write_batch_size
    )


def test_dataset_write_to_multiple_locations_raise_error(sample_records_iter):
    timdex_dataset = TIMDEXDataset(
        location=["/path/to/records-1.parquet", "/path/to/records-2.parquet"]
    )
    with pytest.raises(
        TypeError,
        match="Dataset location must be the root of a single dataset for writing",
    ):
        timdex_dataset.write(sample_records_iter(10))


def test_dataset_write_schema_applied_to_dataset(new_local_dataset, sample_records_iter):
    new_local_dataset.write(sample_records_iter(10))

    # manually load dataset to confirm schema without TIMDEXDataset projecting schema
    # during load
    dataset = ds.dataset(
        new_local_dataset.location,
        format="parquet",
        partitioning="hive",
    )

    assert set(dataset.schema.names) == set(TIMDEX_DATASET_SCHEMA.names)


def test_dataset_write_partition_for_single_source(
    new_local_dataset, sample_records_iter
):
    written_files = new_local_dataset.write(sample_records_iter(10))
    assert len(written_files) == 1
    assert os.path.exists(new_local_dataset.location)
    assert "year=2024/month=12/day=01" in written_files[0].path


def test_dataset_write_partition_for_multiple_sources(
    new_local_dataset, sample_records_iter
):
    # perform write for source="alma" and run_date="2024-12-01"
    written_files_source_a = new_local_dataset.write(sample_records_iter(10))
    new_local_dataset.load()

    assert os.path.exists(written_files_source_a[0].path)
    assert new_local_dataset.row_count == 10

    # perform write for source="libguides" and run_date="2024-12-01"
    written_files_source_b = new_local_dataset.write(
        generate_sample_records(num_records=7, source="libguides")
    )
    new_local_dataset.load()

    assert os.path.exists(written_files_source_b[0].path)
    assert os.path.exists(written_files_source_a[0].path)
    assert new_local_dataset.row_count == 17


def test_dataset_write_partition_ignore_existing_data(
    new_local_dataset, sample_records_iter
):
    # perform two (2) writes for source="alma" and run_date="2024-12-01"
    written_files_source_a0 = new_local_dataset.write(sample_records_iter(10))
    written_files_source_a1 = new_local_dataset.write(sample_records_iter(10))
    new_local_dataset.load()

    # assert that both files exist and no overwriting occurs
    assert os.path.exists(written_files_source_a0[0].path)
    assert os.path.exists(written_files_source_a1[0].path)
    assert new_local_dataset.row_count == 20


@patch("timdex_dataset_api.dataset.uuid.uuid4")
def test_dataset_write_partition_overwrite_files_with_same_name(
    mock_uuid, new_local_dataset, sample_records_iter
):
    """This test is to demonstrate existing_data_behavior="overwrite_or_ignore".

    It is extremely unlikely for the uuid.uuid4 method to generate duplicate values,
    so for testing purposes, this method is patched to return the same value
    and therefore generate similarly named files.
    """
    mock_uuid.return_value = "abc"

    # perform two (2) writes for source="alma" and run_date="2024-12-01"
    _ = new_local_dataset.write(sample_records_iter(10))
    written_files_source_a1 = new_local_dataset.write(sample_records_iter(7))
    new_local_dataset.load()

    # assert that only the second file exists and overwriting occurs
    assert os.path.exists(written_files_source_a1[0].path)
    assert new_local_dataset.row_count == 7
