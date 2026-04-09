# ruff: noqa: PLR2004, D209, D205
import glob
import math
import os
from pathlib import Path
from unittest.mock import patch

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from tests.utils import generate_sample_records
from timdex_dataset_api.records import TIMDEXRecords


def _count_rows_via_duckdb_parquet(timdex_dataset) -> int:
    return timdex_dataset.conn.query(f"""
        select count(*)
        from read_parquet(
            '{timdex_dataset.records.data_root}/**/*.parquet',
            hive_partitioning=true
        )
    """).fetchone()[0]


def _count_parquet_files(timdex_dataset) -> int:
    return len(
        glob.glob(
            f"{timdex_dataset.records.data_root}/**/*.parquet",
            recursive=True,
        )
    )


def test_records_data_root_created_on_init(timdex_dataset_empty):
    expected = f"{timdex_dataset_empty.location.removesuffix('/')}/data/records"
    assert os.path.exists(expected)


def test_embeddings_data_root_created_on_init(timdex_dataset_empty):
    expected = f"{timdex_dataset_empty.location.removesuffix('/')}/data/embeddings"
    assert os.path.exists(expected)
    assert timdex_dataset_empty.embeddings.data_root == expected


def test_dataset_write_records_to_timdex_dataset_empty(
    timdex_dataset_empty, sample_records_generator
):
    written_files = timdex_dataset_empty.records.write(sample_records_generator(10_000))

    assert len(written_files) == 1
    assert os.path.exists(timdex_dataset_empty.location)
    assert _count_rows_via_duckdb_parquet(timdex_dataset_empty) == 10_000


def test_dataset_write_default_max_rows_per_file(
    timdex_dataset_empty, sample_records_generator
):
    """Default is 100k rows per file, therefore writing 200,033 records should result in
    3 files (x2 @ 100k rows, x1 @ 33 rows)."""
    default_max_rows_per_file = timdex_dataset_empty.config.max_rows_per_file
    total_records = 200_033

    timdex_dataset_empty.records.write(sample_records_generator(total_records))

    assert _count_rows_via_duckdb_parquet(timdex_dataset_empty) == total_records
    assert _count_parquet_files(timdex_dataset_empty) == math.ceil(
        total_records / default_max_rows_per_file
    )


def test_dataset_write_schema_applied_to_dataset(
    timdex_dataset_empty, sample_records_generator
):
    timdex_dataset_empty.records.write(sample_records_generator(10))

    # manually load dataset to confirm schema without TIMDEXDataset projecting schema
    # during load
    dataset = ds.dataset(
        timdex_dataset_empty.location,
        format="parquet",
        partitioning="hive",
    )

    assert set(dataset.schema.names) == set(TIMDEXRecords.SCHEMA.names)


def test_dataset_write_partition_for_single_source(
    timdex_dataset_empty, sample_records_generator
):
    written_files = timdex_dataset_empty.records.write(sample_records_generator(10))
    assert len(written_files) == 1
    assert os.path.exists(timdex_dataset_empty.location)
    assert "year=2024/month=12/day=01" in written_files[0].path


def test_dataset_write_partition_for_multiple_sources(
    timdex_dataset_empty, sample_records_generator
):
    # perform write for source="alma" and run_date="2024-12-01"
    written_files_source_a = timdex_dataset_empty.records.write(
        sample_records_generator(10)
    )

    assert os.path.exists(written_files_source_a[0].path)
    assert _count_rows_via_duckdb_parquet(timdex_dataset_empty) == 10

    # perform write for source="libguides" and run_date="2024-12-01"
    written_files_source_b = timdex_dataset_empty.records.write(
        generate_sample_records(num_records=7, source="libguides")
    )

    assert os.path.exists(written_files_source_b[0].path)
    assert os.path.exists(written_files_source_a[0].path)
    assert _count_rows_via_duckdb_parquet(timdex_dataset_empty) == 17


def test_dataset_write_partition_ignore_existing_data(
    timdex_dataset_empty, sample_records_generator
):
    # perform two (2) writes for source="alma" and run_date="2024-12-01"
    written_files_source_a0 = timdex_dataset_empty.records.write(
        sample_records_generator(10)
    )
    written_files_source_a1 = timdex_dataset_empty.records.write(
        sample_records_generator(10)
    )

    # assert that both files exist and no overwriting occurs
    assert os.path.exists(written_files_source_a0[0].path)
    assert os.path.exists(written_files_source_a1[0].path)
    assert _count_rows_via_duckdb_parquet(timdex_dataset_empty) == 20


@patch("timdex_dataset_api.data_source.uuid.uuid4")
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
    _ = timdex_dataset_empty.records.write(sample_records_generator(10))
    written_files_source_a1 = timdex_dataset_empty.records.write(
        sample_records_generator(7)
    )

    # assert that only the second file exists and overwriting occurs
    assert os.path.exists(written_files_source_a1[0].path)
    assert _count_rows_via_duckdb_parquet(timdex_dataset_empty) == 7


def test_dataset_write_single_append_delta_success(
    timdex_dataset_empty, sample_records_generator
):
    written_files = timdex_dataset_empty.records.write(sample_records_generator(1_000))
    records_deltas_path = timdex_dataset_empty.metadata.append_deltas_path_for(
        TIMDEXRecords
    )
    append_deltas = os.listdir(records_deltas_path)

    assert len(append_deltas) == len(written_files)


def test_dataset_write_multiple_append_deltas_success(
    timdex_dataset_empty, sample_records_generator
):
    """Expecting 10 ETL parquet files written, and so 10 append deltas as well."""
    timdex_dataset_empty.config.max_rows_per_file = 100
    timdex_dataset_empty.config.max_rows_per_group = 100

    written_files = timdex_dataset_empty.records.write(sample_records_generator(1_000))
    records_deltas_path = timdex_dataset_empty.metadata.append_deltas_path_for(
        TIMDEXRecords
    )
    append_deltas = os.listdir(records_deltas_path)

    assert len(written_files) == 10
    assert len(append_deltas) == len(written_files)


def test_dataset_write_append_delta_expected_metadata_columns(
    timdex_dataset_empty, sample_records_generator
):
    timdex_dataset_empty.records.write(sample_records_generator(1_000))
    records_deltas_path = timdex_dataset_empty.metadata.append_deltas_path_for(
        TIMDEXRecords
    )
    append_delta_filepath = os.listdir(records_deltas_path)[0]

    append_delta = pq.ParquetFile(Path(records_deltas_path) / append_delta_filepath)
    assert append_delta.schema.names == TIMDEXRecords.SOURCE_METADATA_COLUMNS
