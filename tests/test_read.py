# ruff: noqa: PLR2004

import pandas as pd
import pyarrow as pa
import pytest

from timdex_dataset_api.dataset import TIMDEX_DATASET_SCHEMA

DATASET_COLUMNS_SET = set(TIMDEX_DATASET_SCHEMA.names)


def test_read_batches_yields_pyarrow_record_batches(fixed_local_dataset):
    batches = fixed_local_dataset.read_batches_iter()
    batch = next(batches)
    assert isinstance(batch, pa.RecordBatch)


def test_read_batches_all_columns_by_default(fixed_local_dataset):
    batches = fixed_local_dataset.read_batches_iter()
    batch = next(batches)
    assert set(batch.column_names) == DATASET_COLUMNS_SET


def test_read_batches_filter_columns(fixed_local_dataset):
    columns_subset = ["source", "transformed_record"]
    batches = fixed_local_dataset.read_batches_iter(columns=columns_subset)
    batch = next(batches)
    assert set(batch.column_names) == set(columns_subset)


def test_read_batches_no_filters_gets_full_dataset(fixed_local_dataset):
    batches = fixed_local_dataset.read_batches_iter()
    table = pa.Table.from_batches(batches)
    assert len(table) == fixed_local_dataset.row_count


def test_read_batches_with_filters_gets_subset_of_dataset(fixed_local_dataset):
    batches = fixed_local_dataset.read_batches_iter(
        source="libguides",
        run_date="2024-12-01",
        run_type="daily",
        action="index",
    )

    table = pa.Table.from_batches(batches)
    assert len(table) == 1_000
    assert len(table) < fixed_local_dataset.row_count

    # assert loaded dataset is unchanged by filtering for a read method
    assert fixed_local_dataset.row_count == 5_000


def test_read_dataframe_batches_yields_dataframes(fixed_local_dataset):
    df_iter = fixed_local_dataset.read_dataframes_iter()
    df_batch = next(df_iter)
    assert isinstance(df_batch, pd.DataFrame)
    assert len(df_batch) == 1_000


def test_read_dataframe_reads_all_dataset_rows_after_filtering(fixed_local_dataset):
    df = fixed_local_dataset.read_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == fixed_local_dataset.row_count


def test_read_dicts_yields_dictionary_for_each_dataset_record(fixed_local_dataset):
    records = fixed_local_dataset.read_dicts_iter()
    record = next(records)
    assert isinstance(record, dict)
    assert set(record.keys()) == DATASET_COLUMNS_SET


def test_read_batches_filter_to_none_returns_empty_list(fixed_local_dataset):
    batches = fixed_local_dataset.read_batches_iter(source="not-gonna-find-me")
    assert list(batches) == []


def test_read_dicts_filter_to_none_stopiteration_immediately(fixed_local_dataset):
    batches = fixed_local_dataset.read_dicts_iter(source="not-gonna-find-me")
    with pytest.raises(StopIteration):
        next(batches)


def test_read_transformed_records_yields_parsed_dictionary(fixed_local_dataset):
    batches = fixed_local_dataset.read_transformed_records_iter()
    transformed_record = next(batches)
    assert isinstance(transformed_record, dict)
    assert transformed_record == {"title": ["Hello World."]}
