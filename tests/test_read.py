# ruff: noqa: D205, D209, PLR2004

from datetime import date

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
    assert len(table) == timdex_dataset_multi_source.dataset.count_rows()


def test_read_batches_with_filters_gets_subset_of_dataset(timdex_dataset_multi_source):
    batches = timdex_dataset_multi_source.read_batches_iter(
        source="libguides",
        run_date="2024-12-01",
        run_type="daily",
        action="index",
    )

    table = pa.Table.from_batches(batches)
    assert len(table) == 1_000
    assert len(table) < timdex_dataset_multi_source.dataset.count_rows()

    # assert loaded dataset is unchanged by filtering for a read method
    assert timdex_dataset_multi_source.dataset.count_rows() == 5_000


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
    assert len(df) == timdex_dataset_multi_source.dataset.count_rows()


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


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_all_current_records_deduped(timdex_dataset_with_runs):
    timdex_dataset_with_runs.load(current_records=True)
    all_records_df = timdex_dataset_with_runs.read_dataframe()

    # assert both sources have accurate record counts for current records only
    assert all_records_df.source.value_counts().to_dict() == {"dspace": 90, "alma": 100}

    # assert only one "full" run, per source
    assert len(all_records_df[all_records_df.run_type == "full"].run_id.unique()) == 2

    # assert run_date min/max dates align with both sources min/max dates
    assert all_records_df.run_date.min() == date(2025, 1, 1)  # both
    assert all_records_df.run_date.max() == date(2025, 2, 5)  # dspace


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_source_current_records_deduped(timdex_dataset_with_runs):
    timdex_dataset_with_runs.load(current_records=True, source="alma")
    alma_records_df = timdex_dataset_with_runs.read_dataframe()

    # assert only alma records present and correct count
    assert alma_records_df.source.value_counts().to_dict() == {"alma": 100}

    # assert only one "full" run
    assert len(alma_records_df[alma_records_df.run_type == "full"].run_id.unique()) == 1

    # assert run_date min/max dates are correct for single source
    assert alma_records_df.run_date.min() == date(2025, 1, 1)
    assert alma_records_df.run_date.max() == date(2025, 1, 5)


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_all_read_methods_get_deduplication(
    timdex_dataset_with_runs,
):
    timdex_dataset_with_runs.load(current_records=True, source="alma")

    full_df = timdex_dataset_with_runs.read_dataframe()
    all_records = list(timdex_dataset_with_runs.read_dicts_iter())
    transformed_records = list(timdex_dataset_with_runs.read_transformed_records_iter())

    assert len(full_df) == len(all_records) == len(transformed_records)


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_current_records_no_additional_filtering_accurate_records_yielded(
    timdex_dataset_with_runs,
):
    timdex_dataset_with_runs.load(current_records=True, source="alma")
    df = timdex_dataset_with_runs.read_dataframe()
    assert df.action.value_counts().to_dict() == {"index": 99, "delete": 1}


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_current_records_action_filtering_accurate_records_yielded(
    timdex_dataset_with_runs,
):
    timdex_dataset_with_runs.load(current_records=True, source="alma")
    df = timdex_dataset_with_runs.read_dataframe(action="index")
    assert df.action.value_counts().to_dict() == {"index": 99}


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_current_records_index_filtering_accurate_records_yielded(
    timdex_dataset_with_runs,
):
    """This is a somewhat complex test, but demonstrates that only 'current' records
    are yielded when .load(current_records=True) is applied.

    Given these runs from the fixture:
    [
        ...
        (25, "alma", "2025-01-03", "daily", "index", "run-5"),   <---- filtered to
        (10, "alma", "2025-01-04", "daily", "delete", "run-6"),  <---- influences current
        ...
    ]

    Though we are filtering to run-5, which has 25 total records to-index, we see only 15
    records yielded.  Why?  This is because while we have filtered to only yield from
    run-5, run-6 had 10 deletes which made records alma:0|9 no longer "current" in run-5.
    As we yielded records reverse chronologically, the deletes from run-6 (alma:0-alma:9)
    "influenced" what records we would see as we continue backwards in time.
    """
    # with current_records=False, we get all 25 records from run-5
    timdex_dataset_with_runs.load(current_records=False, source="alma")
    df = timdex_dataset_with_runs.read_dataframe(run_id="run-5")
    assert len(df) == 25

    # with current_records=True, we only get 15 records from run-5
    # because newer run-6 influenced what records are current for older run-5
    timdex_dataset_with_runs.load(current_records=True, source="alma")
    df = timdex_dataset_with_runs.read_dataframe(run_id="run-5")
    assert len(df) == 15
    assert list(df.timdex_record_id) == [
        "alma:10",
        "alma:11",
        "alma:12",
        "alma:13",
        "alma:14",
        "alma:15",
        "alma:16",
        "alma:17",
        "alma:18",
        "alma:19",
        "alma:20",
        "alma:21",
        "alma:22",
        "alma:23",
        "alma:24",
    ]


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_load_current_records_gets_correct_same_day_full_run(
    timdex_dataset_same_day_runs,
):
    """Two full runs were performed on the same day, but 'run-2' was performed most
    recently.  current_records=True should discover the more recent of the two 'run-2',
    not 'run-1'."""
    timdex_dataset_same_day_runs.load(current_records=True, run_type="full")
    df = timdex_dataset_same_day_runs.read_dataframe()

    assert list(df.run_id.unique()) == ["run-2"]


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_load_current_records_gets_correct_same_day_daily_runs_ordering(
    timdex_dataset_same_day_runs,
):
    """Two runs were performed on 2025-01-02, but the most recent records should be from
    run 'run-5' which are action='delete', not 'run-4' with action='index'."""
    timdex_dataset_same_day_runs.load(current_records=True, run_type="daily")
    first_record = next(timdex_dataset_same_day_runs.read_dicts_iter())

    assert first_record["run_id"] == "run-5"
    assert first_record["action"] == "delete"
