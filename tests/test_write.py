# ruff: noqa: PLR2004, D209, D205
import math
import os
from pathlib import Path
from unittest.mock import patch

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from tests.utils import generate_sample_records
from timdex_dataset_api.dataset import (
    TIMDEX_DATASET_SCHEMA,
)
from timdex_dataset_api.metadata import ORDERED_METADATA_COLUMN_NAMES


def test_dataset_write_records_to_timdex_dataset_empty(
    timdex_dataset_empty, sample_records_generator
):
    written_files = timdex_dataset_empty.write(sample_records_generator(10_000))

    assert len(written_files) == 1
    assert os.path.exists(timdex_dataset_empty.location)
    assert timdex_dataset_empty.dataset.count_rows() == 10_000


def test_dataset_write_default_max_rows_per_file(
    timdex_dataset_empty, sample_records_generator
):
    """Default is 100k rows per file, therefore writing 200,033 records should result in
    3 files (x2 @ 100k rows, x1 @ 33 rows)."""
    default_max_rows_per_file = timdex_dataset_empty.config.max_rows_per_file
    total_records = 200_033

    timdex_dataset_empty.write(sample_records_generator(total_records))

    assert timdex_dataset_empty.dataset.count_rows() == total_records
    assert len(timdex_dataset_empty.dataset.files) == math.ceil(
        total_records / default_max_rows_per_file
    )


def test_dataset_write_record_batches_uses_batch_size(
    timdex_dataset_empty, sample_records_generator
):
    total_records = 101
    timdex_dataset_empty.config.write_batch_size = 50
    batches = list(
        timdex_dataset_empty.create_record_batches(
            sample_records_generator(total_records)
        )
    )
    assert len(batches) == math.ceil(
        total_records / timdex_dataset_empty.config.write_batch_size
    )


def test_dataset_write_schema_applied_to_dataset(
    timdex_dataset_empty, sample_records_generator
):
    timdex_dataset_empty.write(sample_records_generator(10))

    # manually load dataset to confirm schema without TIMDEXDataset projecting schema
    # during load
    dataset = ds.dataset(
        timdex_dataset_empty.location,
        format="parquet",
        partitioning="hive",
    )

    assert set(dataset.schema.names) == set(TIMDEX_DATASET_SCHEMA.names)


def test_dataset_write_partition_for_single_source(
    timdex_dataset_empty, sample_records_generator
):
    written_files = timdex_dataset_empty.write(sample_records_generator(10))
    assert len(written_files) == 1
    assert os.path.exists(timdex_dataset_empty.location)
    assert "year=2024/month=12/day=01" in written_files[0].path


def test_dataset_write_partition_for_multiple_sources(
    timdex_dataset_empty, sample_records_generator
):
    # perform write for source="alma" and run_date="2024-12-01"
    written_files_source_a = timdex_dataset_empty.write(sample_records_generator(10))

    assert os.path.exists(written_files_source_a[0].path)
    assert timdex_dataset_empty.dataset.count_rows() == 10

    # perform write for source="libguides" and run_date="2024-12-01"
    written_files_source_b = timdex_dataset_empty.write(
        generate_sample_records(num_records=7, source="libguides")
    )

    assert os.path.exists(written_files_source_b[0].path)
    assert os.path.exists(written_files_source_a[0].path)
    assert timdex_dataset_empty.dataset.count_rows() == 17


def test_dataset_write_partition_ignore_existing_data(
    timdex_dataset_empty, sample_records_generator
):
    # perform two (2) writes for source="alma" and run_date="2024-12-01"
    written_files_source_a0 = timdex_dataset_empty.write(sample_records_generator(10))
    written_files_source_a1 = timdex_dataset_empty.write(sample_records_generator(10))

    # assert that both files exist and no overwriting occurs
    assert os.path.exists(written_files_source_a0[0].path)
    assert os.path.exists(written_files_source_a1[0].path)
    assert timdex_dataset_empty.dataset.count_rows() == 20


@patch("timdex_dataset_api.dataset.uuid.uuid4")
def test_dataset_write_partition_overwrite_files_with_same_name(
    mock_uuid, timdex_dataset_empty, sample_records_generator
):
    """This test is to demonstrate existing_data_behavior="overwrite_or_ignore".

    It is extremely unlikely for the uuid.uuid4 method to generate duplicate values,
    so for testing purposes, this method is patched to return the same value
    and therefore generate similarly named files.
    """
    mock_uuid.return_value = "abc"

    # perform two (2) writes for source="alma" and run_date="2024-12-01"
    _ = timdex_dataset_empty.write(sample_records_generator(10))
    written_files_source_a1 = timdex_dataset_empty.write(sample_records_generator(7))

    # assert that only the second file exists and overwriting occurs
    assert os.path.exists(written_files_source_a1[0].path)
    assert timdex_dataset_empty.dataset.count_rows() == 7


def test_dataset_write_single_append_delta_success(
    timdex_dataset_empty, sample_records_generator
):
    written_files = timdex_dataset_empty.write(sample_records_generator(1_000))
    append_deltas = os.listdir(timdex_dataset_empty.metadata.append_deltas_path)

    assert len(append_deltas) == len(written_files)


def test_dataset_write_multiple_append_deltas_success(
    timdex_dataset_empty, sample_records_generator
):
    """Expecting 10 ETL parquet files written, and so 10 append deltas as well."""
    timdex_dataset_empty.config.max_rows_per_file = 100
    timdex_dataset_empty.config.max_rows_per_group = 100

    written_files = timdex_dataset_empty.write(sample_records_generator(1_000))
    append_deltas = os.listdir(timdex_dataset_empty.metadata.append_deltas_path)

    assert len(written_files) == 10
    assert len(append_deltas) == len(written_files)


def test_dataset_write_append_delta_expected_metadata_columns(
    timdex_dataset_empty, sample_records_generator
):
    timdex_dataset_empty.write(sample_records_generator(1_000))
    append_delta_filepath = os.listdir(timdex_dataset_empty.metadata.append_deltas_path)[
        0
    ]

    append_delta = pq.ParquetFile(
        timdex_dataset_empty.metadata.append_deltas_path / Path(append_delta_filepath)
    )
    assert append_delta.schema.names == ORDERED_METADATA_COLUMN_NAMES
