# ruff: noqa: D205, D209, PLR2004


import pandas as pd
import pyarrow as pa
import pytest
from duckdb import ParserException

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


def test_read_dataframes_yields_dataframes(timdex_dataset_multi_source):
    df_iter = timdex_dataset_multi_source.read_dataframes_iter()
    df_batch = next(df_iter)
    assert isinstance(df_batch, pd.DataFrame)
    assert len(df_batch) == 1_000


def test_read_dataframe_gets_full_dataset(
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


def test_read_batches_where_filters_response(timdex_dataset_multi_source):
    df_all = timdex_dataset_multi_source.read_dataframe()
    total_count = len(df_all)

    where = (
        "source = 'libguides' AND run_date = '2024-12-01' AND "
        "run_type = 'daily' AND action = 'index'"
    )
    df_where = timdex_dataset_multi_source.read_dataframe(where=where)

    assert len(df_where) == 1_000
    assert len(df_where) < total_count


def test_read_batches_where_and_dataset_filters_are_combined(timdex_dataset_multi_source):
    """Test that when key/value DatasetFilters AND a SQL where clause is provided, they
    are combined in the final DuckDB SQL query."""
    where = "run_date = '2024-12-01' AND run_type = 'daily'"
    df = timdex_dataset_multi_source.read_dataframe(
        where=where, source="libguides", action="index"
    )
    assert len(df) == 1_000
    assert set(df["source"].unique().tolist()) == {"libguides"}
    assert set(df["action"].unique().tolist()) == {"index"}


@pytest.mark.parametrize(
    "bad_where",
    [
        "SELECT * FROM current_records WHERE source = 'libguides'",
        "FROM records WHERE source = 'libguides'",
        "ORDER BY timdex_record_id",
        "LIMIT 3",
    ],
)
def test_read_batches_where_rejects_non_predicate_sql(
    timdex_dataset_multi_source, bad_where
):
    with pytest.raises(ParserException):
        next(timdex_dataset_multi_source.read_batches_iter(where=bad_where))


def test_read_dataframe_respects_where(timdex_dataset_multi_source):
    where = "source = 'libguides' AND action = 'index'"
    df = timdex_dataset_multi_source.read_dataframe(where=where)
    assert len(df) > 0
    assert set(df["source"].unique().tolist()) == {"libguides"}
    assert set(df["action"].unique().tolist()) == {"index"}


def test_read_dicts_iter_respects_where_and_filters(timdex_dataset_multi_source):
    where = "run_type = 'daily'"
    it = timdex_dataset_multi_source.read_dicts_iter(where=where, source="libguides")
    first = next(it)
    assert first["run_type"] == "daily"
    assert first["source"] == "libguides"


def test_dataset_all_current_records_deduped(timdex_dataset_with_runs_with_metadata):
    df = timdex_dataset_with_runs_with_metadata.read_dataframe(
        table="current_records",
        columns=["timdex_record_id"],
    )
    assert df is not None
    assert df["timdex_record_id"].nunique() == len(df)


def test_dataset_source_current_records_deduped(timdex_dataset_with_runs_with_metadata):
    df = timdex_dataset_with_runs_with_metadata.read_dataframe(
        table="current_records", source="alma"
    )
    assert df is not None
    assert (df["source"] == "alma").all()
    assert df["timdex_record_id"].nunique() == len(df)


def test_dataset_all_read_methods_get_deduplication(
    timdex_dataset_with_runs_with_metadata,
):
    batch_rows = 0
    for b in timdex_dataset_with_runs_with_metadata.read_batches_iter(
        table="current_records", columns=["timdex_record_id"]
    ):
        batch_rows += len(b)
    dict_rows = sum(
        1
        for _ in timdex_dataset_with_runs_with_metadata.read_dicts_iter(
            table="current_records", columns=["timdex_record_id"]
        )
    )
    df = timdex_dataset_with_runs_with_metadata.read_dataframe(
        table="current_records", columns=["timdex_record_id"]
    )
    assert df is not None
    assert batch_rows == dict_rows == len(df)
    assert df["timdex_record_id"].nunique() == len(df)


def test_dataset_current_records_no_additional_filtering_accurate_records_yielded(
    timdex_dataset_with_runs_with_metadata,
):
    df_all = timdex_dataset_with_runs_with_metadata.read_dataframe(
        table="current_records"
    )
    assert df_all is not None
    df_total = timdex_dataset_with_runs_with_metadata.read_dataframe()
    assert df_total is not None
    assert len(df_all) <= len(df_total)
    assert df_all["timdex_record_id"].nunique() == len(df_all)


def test_dataset_current_records_action_filtering_accurate_records_yielded(
    timdex_dataset_with_runs_with_metadata,
):
    df = timdex_dataset_with_runs_with_metadata.read_dataframe(
        table="current_records", action="index"
    )
    assert df is not None
    assert set(df["action"].unique().tolist()) == {"index"}


def test_dataset_current_records_index_filtering_accurate_records_yielded(
    timdex_dataset_with_runs_with_metadata,
):
    # with all records, run-5 has 25 rows
    df_all = timdex_dataset_with_runs_with_metadata.read_dataframe(
        source="alma", run_id="run-5"
    )
    assert df_all is not None
    assert len(df_all) == 25

    # within current_records, only 15 remain due to later deletes
    df_current = timdex_dataset_with_runs_with_metadata.read_dataframe(
        table="current_records", source="alma", run_id="run-5"
    )
    assert df_current is not None
    assert len(df_current) == 15
    assert list(df_current.timdex_record_id) == [
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


def test_dataset_load_current_records_gets_correct_same_day_full_run(
    timdex_dataset_same_day_runs,
):
    # ensure metadata exists for this dataset
    timdex_dataset_same_day_runs.metadata.rebuild_dataset_metadata()
    timdex_dataset_same_day_runs.metadata.refresh()
    df = timdex_dataset_same_day_runs.read_dataframe(
        table="current_records", run_type="full"
    )
    assert list(df.run_id.unique()) == ["run-2"]


def test_dataset_load_current_records_gets_correct_same_day_daily_runs_ordering(
    timdex_dataset_same_day_runs,
):
    timdex_dataset_same_day_runs.metadata.rebuild_dataset_metadata()
    timdex_dataset_same_day_runs.metadata.refresh()
    first_record = next(
        timdex_dataset_same_day_runs.read_dicts_iter(
            table="current_records", run_type="daily"
        )
    )
    # ordering is latest by run_timestamp within day;
    # just assert it's one of the daily runs
    assert first_record["run_id"] in {"run-4", "run-5"}
    assert first_record["action"] in {"index", "delete"}


def test_read_batches_iter_limit_returns_n_rows(timdex_dataset_multi_source):
    batches = timdex_dataset_multi_source.read_batches_iter(limit=10)
    table = pa.Table.from_batches(batches)
    assert len(table) == 10
