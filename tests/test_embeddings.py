# ruff: noqa: PLR2004
import json
import math
import os
from datetime import UTC, date, datetime

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from tests.utils import generate_sample_embeddings_for_run
from timdex_dataset_api.embeddings import (
    METADATA_SELECT_FILTER_COLUMNS,
    TIMDEX_DATASET_EMBEDDINGS_SCHEMA,
    DatasetEmbedding,
    TIMDEXEmbeddings,
)

EMBEDDINGS_COLUMNS_SET = set(TIMDEX_DATASET_EMBEDDINGS_SCHEMA.names)
EMBEDDINGS_WITH_METADATA_COLUMNS_SET = EMBEDDINGS_COLUMNS_SET | set(
    METADATA_SELECT_FILTER_COLUMNS
)


def test_dataset_embedding_init():
    values = {
        "timdex_record_id": "alma:123",
        "run_id": "test-run-1",
        "run_record_offset": 0,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_strategy": "full_record",
        "embedding_timestamp": "2024-12-01T10:00:00+00:00",
        "embedding_vector": [0.1, 0.2, 0.3],
        "embedding_object": json.dumps(
            {"token1": 0.1, "token2": 0.2, "token3": 0.3}
        ).encode(),
    }
    embedding = DatasetEmbedding(**values)

    assert embedding
    assert embedding.timdex_record_id == "alma:123"
    assert embedding.embedding_timestamp == datetime(2024, 12, 1, 10, 0, tzinfo=UTC)
    assert embedding.embedding_object == b'{"token1": 0.1, "token2": 0.2, "token3": 0.3}'


def test_dataset_embedding_date_properties():
    embedding = DatasetEmbedding(
        timdex_record_id="alma:123",
        run_id="test-run-1",
        run_record_offset=0,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_strategy="full_record",
        embedding_timestamp="2024-12-01T10:00:00+00:00",
        embedding_vector=[0.1, 0.2, 0.3],
    )

    assert (embedding.year, embedding.month, embedding.day) == ("2024", "12", "01")


def test_dataset_embedding_to_dict():
    values = {
        "timdex_record_id": "alma:123",
        "run_id": "test-run-1",
        "run_record_offset": 0,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_strategy": "full_record",
        "embedding_timestamp": "2024-12-01T10:00:00+00:00",
        "embedding_vector": [0.1, 0.2, 0.3],
        "embedding_object": None,
    }
    embedding = DatasetEmbedding(**values)
    embedding_dict = embedding.to_dict()

    assert embedding_dict["timdex_record_id"] == "alma:123"
    assert embedding_dict["year"] == "2024"
    assert embedding_dict["month"] == "12"
    assert embedding_dict["day"] == "01"
    assert embedding_dict["embedding_vector"] == [0.1, 0.2, 0.3]


def test_embeddings_data_root_property(timdex_dataset_empty):
    timdex_embeddings = TIMDEXEmbeddings(timdex_dataset_empty)

    expected = f"{timdex_dataset_empty.location.removesuffix('/')}/data/embeddings"
    assert timdex_embeddings.data_embeddings_root == expected


def test_embeddings_write_basic(timdex_dataset_empty, sample_embeddings_generator):
    timdex_embeddings = TIMDEXEmbeddings(timdex_dataset_empty)
    written_files = timdex_embeddings.write(sample_embeddings_generator(100))

    assert len(written_files) == 1
    assert os.path.exists(written_files[0].path)

    # verify written data can be read
    dataset = ds.dataset(
        timdex_embeddings.data_embeddings_root, format="parquet", partitioning="hive"
    )
    assert dataset.count_rows() == 100


def test_embeddings_write_partitioning(timdex_dataset_empty, sample_embeddings_generator):
    timdex_embeddings = TIMDEXEmbeddings(timdex_dataset_empty)
    written_files = timdex_embeddings.write(sample_embeddings_generator(10))

    assert len(written_files) == 1
    assert "year=2024/month=12/day=01" in written_files[0].path


def test_embeddings_write_schema_applied(
    timdex_dataset_empty, sample_embeddings_generator
):
    timdex_embeddings = TIMDEXEmbeddings(timdex_dataset_empty)
    timdex_embeddings.write(sample_embeddings_generator(10))

    # manually load dataset to confirm schema
    dataset = ds.dataset(
        timdex_embeddings.data_embeddings_root,
        format="parquet",
        partitioning="hive",
    )

    assert set(dataset.schema.names) == set(TIMDEX_DATASET_EMBEDDINGS_SCHEMA.names)


def test_embeddings_create_batches(timdex_dataset_empty, sample_embeddings_generator):
    timdex_embeddings = TIMDEXEmbeddings(timdex_dataset_empty)
    total_embeddings = 101
    timdex_dataset_empty.config.write_batch_size = 50

    batches = list(
        timdex_embeddings.create_embedding_batches(
            sample_embeddings_generator(total_embeddings)
        )
    )

    assert len(batches) == math.ceil(
        total_embeddings / timdex_dataset_empty.config.write_batch_size
    )


def test_embeddings_read_batches_yields_pyarrow_record_batches(
    timdex_dataset_empty, sample_embeddings_generator, sample_records_generator
):
    # write matching records and rebuild metadata
    timdex_dataset_empty.write(
        sample_records_generator(100, source="alma", run_id="test-run"),
        write_append_deltas=False,
    )
    timdex_dataset_empty.metadata.rebuild_dataset_metadata()
    timdex_dataset_empty.refresh()

    # write embeddings and refresh to pick up new views
    timdex_dataset_empty.embeddings.write(
        sample_embeddings_generator(100, run_id="test-run")
    )
    timdex_dataset_empty.refresh()

    batches = timdex_dataset_empty.embeddings.read_batches_iter()
    batch = next(batches)
    assert isinstance(batch, pa.RecordBatch)


def test_embeddings_read_batches_all_columns_by_default(timdex_embeddings_with_runs):
    batches = timdex_embeddings_with_runs.read_batches_iter()
    batch = next(batches)
    assert set(batch.column_names) == EMBEDDINGS_WITH_METADATA_COLUMNS_SET


def test_embeddings_read_batches_filter_columns(timdex_embeddings_with_runs):
    columns_subset = ["timdex_record_id", "run_id", "embedding_strategy"]
    batches = timdex_embeddings_with_runs.read_batches_iter(columns=columns_subset)
    batch = next(batches)
    assert set(batch.column_names) == set(columns_subset)


def test_embeddings_read_batches_explicit_columns_excludes_metadata(
    timdex_embeddings_with_runs,
):
    """Explicit column selection excludes metadata columns."""
    columns_subset = ["timdex_record_id", "embedding_vector"]
    batches = timdex_embeddings_with_runs.read_batches_iter(columns=columns_subset)
    batch = next(batches)
    # only requested columns returned, no metadata columns
    assert set(batch.column_names) == set(columns_subset)
    assert "source" not in batch.column_names
    assert "run_date" not in batch.column_names


def test_embeddings_read_batches_mixed_columns(timdex_embeddings_with_runs):
    """Select both embeddings and metadata columns explicitly."""
    columns = ["timdex_record_id", "source", "embedding_vector", "run_date"]
    batches = timdex_embeddings_with_runs.read_batches_iter(columns=columns)
    batch = next(batches)
    assert set(batch.column_names) == set(columns)


def test_embeddings_read_batches_metadata_only_columns(timdex_embeddings_with_runs):
    """Select only metadata columns explicitly."""
    columns = ["source", "run_date", "run_type"]
    batches = timdex_embeddings_with_runs.read_batches_iter(columns=columns)
    batch = next(batches)
    assert set(batch.column_names) == set(columns)


def test_embeddings_read_batches_invalid_columns_raises_error(
    timdex_embeddings_with_runs,
):
    """Invalid column names raise ValueError with helpful message."""
    columns = ["timdex_record_id", "invalid_column", "source"]
    with pytest.raises(ValueError, match=r"Invalid column.*invalid_column"):
        list(timdex_embeddings_with_runs.read_batches_iter(columns=columns))


def test_embeddings_read_batches_gets_full_dataset(timdex_embeddings_with_runs):
    batches = timdex_embeddings_with_runs.read_batches_iter()
    table = pa.Table.from_batches(batches)
    dataset = ds.dataset(
        timdex_embeddings_with_runs.data_embeddings_root,
        format="parquet",
        partitioning="hive",
    )
    assert len(table) == dataset.count_rows()


def test_embeddings_read_batches_with_filters_gets_subset_of_dataset(
    timdex_embeddings_with_runs,
):
    batches = timdex_embeddings_with_runs.read_batches_iter(
        run_id="abc123", embedding_strategy="full_record"
    )
    table = pa.Table.from_batches(batches)
    dataset = ds.dataset(
        timdex_embeddings_with_runs.data_embeddings_root,
        format="parquet",
        partitioning="hive",
    )
    assert len(table) == 100
    assert len(table) < dataset.count_rows()


def test_embeddings_read_batches_with_metadata_filter_source(
    timdex_embeddings_with_runs,
):
    """Filter embeddings by 'source' column from metadata.records via join."""
    batches = timdex_embeddings_with_runs.read_batches_iter(source="alma")
    table = pa.Table.from_batches(batches)
    assert len(table) == 150


def test_embeddings_read_batches_with_metadata_filter_run_date(
    timdex_embeddings_with_runs,
):
    """Filter embeddings by 'run_date' column from metadata.records via join."""
    batches = timdex_embeddings_with_runs.read_batches_iter(run_date="2024-12-01")
    table = pa.Table.from_batches(batches)
    assert len(table) == 150


def test_embeddings_read_batches_with_combined_filters(
    timdex_embeddings_with_runs,
):
    """Combine embeddings filter and metadata filter."""
    batches = timdex_embeddings_with_runs.read_batches_iter(
        source="alma", run_id="abc123"
    )
    table = pa.Table.from_batches(batches)
    assert len(table) == 100


def test_embeddings_read_dataframes_yields_dataframes(timdex_embeddings_with_runs):
    df_iter = timdex_embeddings_with_runs.read_dataframes_iter()
    df_batch = next(df_iter)
    assert isinstance(df_batch, pd.DataFrame)
    assert len(df_batch) == 150


def test_embeddings_read_dataframe_gets_full_dataset(timdex_embeddings_with_runs):
    df = timdex_embeddings_with_runs.read_dataframe()
    dataset = ds.dataset(
        timdex_embeddings_with_runs.data_embeddings_root,
        format="parquet",
        partitioning="hive",
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == dataset.count_rows()


def test_embeddings_read_dicts_yields_dictionary_for_each_embeddings_record(
    timdex_embeddings_with_runs,
):
    dict_iter = timdex_embeddings_with_runs.read_dicts_iter()
    record = next(dict_iter)
    assert isinstance(record, dict)
    assert set(record.keys()) == EMBEDDINGS_WITH_METADATA_COLUMNS_SET


def test_current_embeddings_view_single_run(timdex_dataset_for_embeddings_views):
    td = timdex_dataset_for_embeddings_views

    # write embeddings for run "apple-1"
    td.embeddings.write(generate_sample_embeddings_for_run(td, run_id="apple-1"))
    td.refresh()

    # query current_embeddings for apple source using read_dataframe
    result = td.embeddings.read_dataframe(table="current_embeddings", source="apple")

    assert len(result) == 10
    assert (result["run_id"] == "apple-1").all()
    assert (result["run_date"] == date(2025, 6, 1)).all()


def test_current_embeddings_view_multiple_runs(timdex_dataset_for_embeddings_views):
    td = timdex_dataset_for_embeddings_views

    # write embeddings for runs "orange-1" and "orange-2"
    td.embeddings.write(generate_sample_embeddings_for_run(td, run_id="orange-1"))
    td.embeddings.write(generate_sample_embeddings_for_run(td, run_id="orange-2"))
    td.refresh()

    # query current_embeddings for orange source using read_dataframe
    result = td.embeddings.read_dataframe(table="current_embeddings", source="orange")

    # 10 total current embeddings:
    # 5 from orange-1 (offsets 5-9), 5 from orange-2 (offsets 0-4)
    assert len(result) == 10

    # verify 5 from orange-1 (records not in orange-2, run_date 2025-07-01)
    orange_1_records = result[result["run_id"] == "orange-1"]
    assert len(orange_1_records) == 5
    assert (orange_1_records["run_date"] == date(2025, 7, 1)).all()

    # verify 5 from orange-2 (newer records, run_date 2025-07-02)
    orange_2_records = result[result["run_id"] == "orange-2"]
    assert len(orange_2_records) == 5
    assert (orange_2_records["run_date"] == date(2025, 7, 2)).all()


def test_current_embeddings_view_handles_duplicate_run_embeddings(
    timdex_dataset_for_embeddings_views,
):
    td = timdex_dataset_for_embeddings_views

    # write embeddings for run "lemon-1"
    td.embeddings.write(generate_sample_embeddings_for_run(td, run_id="lemon-1"))

    # first embeddings run for run "lemon-2"
    td.embeddings.write(
        generate_sample_embeddings_for_run(
            td, run_id="lemon-2", embedding_timestamp="2025-08-02T00:00:00+00:00"
        )
    )

    # second embeddings run for run "lemon-2" with a later timestamp
    td.embeddings.write(
        generate_sample_embeddings_for_run(
            td, run_id="lemon-2", embedding_timestamp="2025-08-03T00:00:00+00:00"
        )
    )
    td.refresh()

    # check all embeddings for lemon-2 to verify both writes exist
    all_lemon_2 = td.embeddings.read_dataframe(table="embeddings", run_id="lemon-2")
    # should have 10 rows (5 from first write, 5 from second write)
    assert len(all_lemon_2) == 10

    # verify both timestamps exist
    unique_timestamps = all_lemon_2["embedding_timestamp"].unique()
    assert len(unique_timestamps) == 2

    # query current_embeddings for lemon source
    result = td.embeddings.read_dataframe(table="current_embeddings", source="lemon")

    # 10 current embeddings: 5 from lemon-1, 5 from lemon-2 (latest timestamp)
    assert len(result) == 10

    # verify lemon-1 embeddings (run_date 2025-08-01)
    lemon_1_result = result[result["run_id"] == "lemon-1"]
    assert len(lemon_1_result) == 5
    assert (lemon_1_result["run_date"] == date(2025, 8, 1)).all()

    # verify lemon-2 embeddings have the later embedding timestamp (run_date 2025-08-02)
    lemon_2_result = result[result["run_id"] == "lemon-2"]
    assert len(lemon_2_result) == 5
    assert (lemon_2_result["run_date"] == date(2025, 8, 2)).all()

    # all lemon-2 current embeddings should have the later embedding timestamp
    max_timestamp = all_lemon_2["embedding_timestamp"].max()
    assert (lemon_2_result["embedding_timestamp"] == max_timestamp).all()


def test_embeddings_view_includes_all_embeddings(timdex_dataset_for_embeddings_views):
    td = timdex_dataset_for_embeddings_views

    # write embeddings for lemon-1
    td.embeddings.write(generate_sample_embeddings_for_run(td, run_id="lemon-1"))

    # write embeddings for lemon-2 (first time) with explicit timestamp
    td.embeddings.write(
        generate_sample_embeddings_for_run(
            td, run_id="lemon-2", embedding_timestamp="2025-08-02T00:00:00+00:00"
        )
    )

    # write embeddings for lemon-2 again with later timestamp
    td.embeddings.write(
        generate_sample_embeddings_for_run(
            td, run_id="lemon-2", embedding_timestamp="2025-08-03T00:00:00+00:00"
        )
    )
    td.refresh()

    # query all embeddings for lemon source
    result = td.embeddings.read_dataframe(table="embeddings", source="lemon")

    # 20 total embeddings: 10 from lemon-1, 5 from lemon-2 first write,
    # 5 from lemon-2 second write
    assert len(result) == 20

    # verify run_date distribution
    lemon_1_embeddings = result[result["run_id"] == "lemon-1"]
    assert len(lemon_1_embeddings) == 10
    assert (lemon_1_embeddings["run_date"] == date(2025, 8, 1)).all()

    lemon_2_embeddings = result[result["run_id"] == "lemon-2"]
    assert len(lemon_2_embeddings) == 10  # 5 from each write
    assert (lemon_2_embeddings["run_date"] == date(2025, 8, 2)).all()


def test_embeddings_read_batches_iter_returns_empty_when_embeddings_missing(
    timdex_dataset_empty, caplog
):
    result = list(timdex_dataset_empty.embeddings.read_batches_iter())
    assert result == []
    assert (
        "Table 'embeddings' not found in DuckDB context.  Embeddings may not yet exist "
        "or TIMDEXDataset.refresh() may be required." in caplog.text
    )


def test_embeddings_read_batches_iter_returns_empty_for_invalid_table(
    timdex_embeddings_with_runs, caplog
):
    """read_batches_iter returns empty iterator for nonexistent table name."""
    with pytest.raises(
        ValueError,
        match="Invalid table: 'nonexistent'",
    ):
        list(timdex_embeddings_with_runs.read_batches_iter(table="nonexistent"))
