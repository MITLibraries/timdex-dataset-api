# ruff: noqa: PLR2004

import pandas as pd
import pyarrow as pa
import pytest

from timdex_dataset_api.dataset import TIMDEX_DATASET_SCHEMA

DATASET_COLUMNS_SET = set(TIMDEX_DATASET_SCHEMA.names)


def test_read_batches_yields_pyarrow_record_batches(timdex_dataset_multi_source):
    batches = timdex_dataset_multi_source.read_batches_iter()
    batch = next(batches)
    assert isinstance(batch, pa.RecordBatch)


def test_read_batches_all_columns_by_default(timdex_dataset_multi_source):
    batches = timdex_dataset_multi_source.read_batches_iter()
    batch = next(batches)
    assert set(batch.column_names) == DATASET_COLUMNS_SET


def test_read_batches_filter_columns(timdex_dataset_multi_source):
    columns_subset = ["source", "transformed_record"]
    batches = timdex_dataset_multi_source.read_batches_iter(columns=columns_subset)
    batch = next(batches)
    assert set(batch.column_names) == set(columns_subset)


def test_read_batches_no_filters_gets_full_dataset(timdex_dataset_multi_source):
    batches = timdex_dataset_multi_source.read_batches_iter()
    table = pa.Table.from_batches(batches)
    assert len(table) == timdex_dataset_multi_source.row_count


def test_read_batches_with_filters_gets_subset_of_dataset(timdex_dataset_multi_source):
    batches = timdex_dataset_multi_source.read_batches_iter(
        source="libguides",
        run_date="2024-12-01",
        run_type="daily",
        action="index",
    )

    table = pa.Table.from_batches(batches)
    assert len(table) == 1_000
    assert len(table) < timdex_dataset_multi_source.row_count

    # assert loaded dataset is unchanged by filtering for a read method
    assert timdex_dataset_multi_source.row_count == 5_000


def test_read_dataframe_batches_yields_dataframes(timdex_dataset_multi_source):
    df_iter = timdex_dataset_multi_source.read_dataframes_iter()
    df_batch = next(df_iter)
    assert isinstance(df_batch, pd.DataFrame)
    assert len(df_batch) == 1_000


def test_read_dataframe_reads_all_dataset_rows_after_filtering(
    timdex_dataset_multi_source,
):
    df = timdex_dataset_multi_source.read_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == timdex_dataset_multi_source.row_count


def test_read_dicts_yields_dictionary_for_each_dataset_record(
    timdex_dataset_multi_source,
):
    records = timdex_dataset_multi_source.read_dicts_iter()
    record = next(records)
    assert isinstance(record, dict)
    assert set(record.keys()) == DATASET_COLUMNS_SET


def test_read_batches_filter_to_none_returns_empty_list(timdex_dataset_multi_source):
    batches = timdex_dataset_multi_source.read_batches_iter(source="not-gonna-find-me")
    assert list(batches) == []


def test_read_dicts_filter_to_none_stopiteration_immediately(timdex_dataset_multi_source):
    batches = timdex_dataset_multi_source.read_dicts_iter(source="not-gonna-find-me")
    with pytest.raises(StopIteration):
        next(batches)


def test_read_transformed_records_yields_parsed_dictionary(timdex_dataset_multi_source):
    batches = timdex_dataset_multi_source.read_transformed_records_iter()
    transformed_record = next(batches)
    assert isinstance(transformed_record, dict)
    assert transformed_record == {"title": ["Hello World."]}
