# ruff: noqa: PLR2004

import hashlib
import os
from datetime import UTC, datetime

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from tests.utils import generate_sample_fulltexts_for_run, generate_sample_records
from timdex_dataset_api import TIMDEXDataset
from timdex_dataset_api.data_types import TIMDEXFulltexts
from timdex_dataset_api.data_types.fulltexts import DatasetFulltext

FULLTEXTS_AVAILABLE_COLUMNS_SET = set(TIMDEXFulltexts.AVAILABLE_READ_COLUMNS)


def test_dataset_fulltext_init():
    values = {
        "timdex_record_id": "alma:123",
        "run_id": "test-run-1",
        "run_record_offset": 0,
        "fulltext_timestamp": "2024-12-01T10:00:00+00:00",
        "fulltext": b"Sample fulltext content.",
    }
    fulltext = DatasetFulltext(**values)

    assert fulltext
    assert fulltext.timdex_record_id == "alma:123"
    assert fulltext.fulltext_timestamp == datetime(2024, 12, 1, 10, 0, tzinfo=UTC)
    assert fulltext.fulltext_md5 is None
    assert fulltext.fulltext == b"Sample fulltext content."


def test_dataset_fulltext_date_properties():
    fulltext = DatasetFulltext(
        timdex_record_id="alma:123",
        run_id="test-run-1",
        run_record_offset=0,
        fulltext_timestamp="2024-12-01T10:00:00+00:00",
        fulltext=b"Sample fulltext content.",
    )

    assert (fulltext.year, fulltext.month, fulltext.day) == ("2024", "12", "01")


def test_dataset_fulltext_to_dict():
    values = {
        "timdex_record_id": "alma:123",
        "run_id": "test-run-1",
        "run_record_offset": 0,
        "fulltext_timestamp": "2024-12-01T10:00:00+00:00",
        "fulltext": b"Sample fulltext content.",
    }
    fulltext = DatasetFulltext(**values)
    fulltext_dict = fulltext.to_dict()

    target_md5 = hashlib.md5(
        b"Sample fulltext content.", usedforsecurity=False
    ).hexdigest()

    assert fulltext_dict["timdex_record_id"] == "alma:123"
    assert fulltext_dict["year"] == "2024"
    assert fulltext_dict["month"] == "12"
    assert fulltext_dict["day"] == "01"
    assert fulltext_dict["fulltext"] == b"Sample fulltext content."
    assert fulltext_dict["fulltext_md5"] == target_md5


def test_dataset_fulltext_to_dict_preserves_provided_md5():
    fulltext = DatasetFulltext(
        timdex_record_id="alma:123",
        run_id="test-run-1",
        run_record_offset=0,
        fulltext_timestamp="2024-12-01T10:00:00+00:00",
        fulltext_md5="provided-md5",
        fulltext=b"Sample fulltext content.",
    )

    assert fulltext.to_dict()["fulltext_md5"] == "provided-md5"


def test_dataset_fulltext_to_dict_no_md5_without_fulltext():
    fulltext = DatasetFulltext(
        timdex_record_id="alma:123",
        run_id="test-run-1",
        run_record_offset=0,
        fulltext_timestamp="2024-12-01T10:00:00+00:00",
    )

    assert fulltext.to_dict()["fulltext_md5"] is None


def test_fulltexts_data_root_property(timdex_dataset_empty):
    timdex_fulltexts = TIMDEXFulltexts(timdex_dataset_empty)

    expected = f"{timdex_dataset_empty.location.removesuffix('/')}/data/fulltexts"
    assert timdex_fulltexts.data_root == expected
    assert os.path.exists(expected)


def test_fulltexts_write_basic(timdex_dataset_empty, sample_fulltexts_generator):
    timdex_fulltexts = TIMDEXFulltexts(timdex_dataset_empty)
    written_files = timdex_fulltexts.write(sample_fulltexts_generator(100))

    assert len(written_files) == 1
    assert os.path.exists(written_files[0].path)

    # verify written data can be read
    dataset = ds.dataset(
        timdex_fulltexts.data_root, format="parquet", partitioning="hive"
    )
    assert dataset.count_rows() == 100


def test_fulltexts_write_partitioning(timdex_dataset_empty, sample_fulltexts_generator):
    timdex_fulltexts = TIMDEXFulltexts(timdex_dataset_empty)
    written_files = timdex_fulltexts.write(sample_fulltexts_generator(10))

    assert len(written_files) == 1
    assert "year=2024/month=12/day=01" in written_files[0].path


def test_fulltexts_write_schema_applied(timdex_dataset_empty, sample_fulltexts_generator):
    timdex_fulltexts = TIMDEXFulltexts(timdex_dataset_empty)
    timdex_fulltexts.write(sample_fulltexts_generator(10))

    # manually load dataset to confirm schema
    dataset = ds.dataset(
        timdex_fulltexts.data_root,
        format="parquet",
        partitioning="hive",
    )

    assert set(dataset.schema.names) == set(TIMDEXFulltexts.SCHEMA.names)


def test_fulltexts_read_batches_yields_pyarrow_record_batches(
    timdex_dataset_empty, sample_fulltexts_generator, sample_records_generator
):
    # write matching records and rebuild metadata
    timdex_dataset_empty.records.write(
        sample_records_generator(100, source="alma", run_id="test-run"),
        write_append_deltas=False,
    )
    timdex_dataset_empty.metadata.rebuild_dataset_metadata()
    timdex_dataset_empty.refresh()

    # write fulltexts
    timdex_dataset_empty.fulltexts.write(
        sample_fulltexts_generator(100, run_id="test-run")
    )

    # rebuild metadata to include fulltexts, then refresh
    timdex_dataset_empty.metadata.rebuild_dataset_metadata()
    timdex_dataset_empty.refresh()

    batches = timdex_dataset_empty.fulltexts.read_batches_iter()
    batch = next(batches)
    assert isinstance(batch, pa.RecordBatch)


def test_fulltexts_read_batches_all_columns_by_default(timdex_fulltexts_with_runs):
    batches = timdex_fulltexts_with_runs.read_batches_iter()
    batch = next(batches)
    assert set(batch.column_names) == FULLTEXTS_AVAILABLE_COLUMNS_SET


def test_fulltexts_read_batches_filter_columns(timdex_fulltexts_with_runs):
    columns_subset = [
        "timdex_record_id",
        "run_id",
        "fulltext_timestamp",
        "fulltext_md5",
    ]
    batches = timdex_fulltexts_with_runs.read_batches_iter(columns=columns_subset)
    batch = next(batches)
    assert set(batch.column_names) == set(columns_subset)


def test_fulltexts_read_batches_explicit_columns_excludes_metadata(
    timdex_fulltexts_with_runs,
):
    """Explicit column selection excludes metadata columns."""
    columns_subset = ["timdex_record_id", "fulltext"]
    batches = timdex_fulltexts_with_runs.read_batches_iter(columns=columns_subset)
    batch = next(batches)
    # only requested columns returned, no metadata columns
    assert set(batch.column_names) == set(columns_subset)
    assert "source" not in batch.column_names
    assert "run_date" not in batch.column_names


def test_fulltexts_read_batches_mixed_columns(timdex_fulltexts_with_runs):
    """Select both fulltexts and metadata columns explicitly."""
    columns = ["timdex_record_id", "source", "fulltext", "run_date"]
    batches = timdex_fulltexts_with_runs.read_batches_iter(columns=columns)
    batch = next(batches)
    assert set(batch.column_names) == set(columns)


def test_fulltexts_read_batches_metadata_only_columns(timdex_fulltexts_with_runs):
    """Select only metadata columns explicitly."""
    columns = ["source", "run_date", "run_type"]
    batches = timdex_fulltexts_with_runs.read_batches_iter(columns=columns)
    batch = next(batches)
    assert set(batch.column_names) == set(columns)


def test_fulltexts_read_batches_invalid_columns_raises_error(
    timdex_fulltexts_with_runs,
):
    """Invalid column names raise ValueError with helpful message."""
    columns = ["timdex_record_id", "invalid_column", "source"]
    with pytest.raises(ValueError, match=r"Invalid column.*invalid_column"):
        list(timdex_fulltexts_with_runs.read_batches_iter(columns=columns))


def test_fulltexts_read_batches_gets_full_dataset(timdex_fulltexts_with_runs):
    batches = timdex_fulltexts_with_runs.read_batches_iter()
    table = pa.Table.from_batches(batches)
    dataset = ds.dataset(
        timdex_fulltexts_with_runs.data_root,
        format="parquet",
        partitioning="hive",
    )
    assert len(table) == dataset.count_rows()


def test_fulltexts_read_batches_with_filters_gets_subset_of_dataset(
    timdex_fulltexts_with_runs,
):
    batches = timdex_fulltexts_with_runs.read_batches_iter(run_id="abc123")
    table = pa.Table.from_batches(batches)
    dataset = ds.dataset(
        timdex_fulltexts_with_runs.data_root,
        format="parquet",
        partitioning="hive",
    )
    assert len(table) == 100
    assert len(table) < dataset.count_rows()


def test_fulltexts_read_batches_with_metadata_filter_source(
    timdex_fulltexts_with_runs,
):
    """Filter fulltexts by 'source' column from metadata.records via join."""
    batches = timdex_fulltexts_with_runs.read_batches_iter(source="alma")
    table = pa.Table.from_batches(batches)
    assert len(table) == 150


def test_fulltexts_read_batches_with_metadata_filter_run_date(
    timdex_fulltexts_with_runs,
):
    """Filter fulltexts by 'run_date' column from metadata.records via join."""
    batches = timdex_fulltexts_with_runs.read_batches_iter(run_date="2024-12-01")
    table = pa.Table.from_batches(batches)
    assert len(table) == 150


def test_fulltexts_read_batches_with_combined_filters(
    timdex_fulltexts_with_runs,
):
    """Combine fulltexts filter and metadata filter."""
    batches = timdex_fulltexts_with_runs.read_batches_iter(source="alma", run_id="abc123")
    table = pa.Table.from_batches(batches)
    assert len(table) == 100


def test_fulltexts_read_batches_with_fulltext_md5_filter(timdex_fulltexts_with_runs):
    """Filter fulltexts by generated fulltext_md5 metadata."""
    expected_md5 = hashlib.md5(
        b"Sample fulltext content for alma:0.  Content " + (b"x" * 1024),
        usedforsecurity=False,
    ).hexdigest()

    batches = timdex_fulltexts_with_runs.read_batches_iter(fulltext_md5=expected_md5)
    table = pa.Table.from_batches(batches)

    assert len(table) == 2
    assert set(table.column("timdex_record_id").to_pylist()) == {"alma:0"}
    assert set(table.column("fulltext_md5").to_pylist()) == {expected_md5}


def test_fulltexts_read_dataframes_yields_dataframes(timdex_fulltexts_with_runs):
    df_iter = timdex_fulltexts_with_runs.read_dataframes_iter()
    df_batch = next(df_iter)
    assert isinstance(df_batch, pd.DataFrame)
    assert len(df_batch) == 150


def test_fulltexts_read_dataframe_gets_full_dataset(timdex_fulltexts_with_runs):
    df = timdex_fulltexts_with_runs.read_dataframe()
    dataset = ds.dataset(
        timdex_fulltexts_with_runs.data_root,
        format="parquet",
        partitioning="hive",
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == dataset.count_rows()


def test_fulltexts_read_dicts_yields_dictionary_for_each_fulltexts_record(
    timdex_fulltexts_with_runs,
):
    dict_iter = timdex_fulltexts_with_runs.read_dicts_iter()
    record = next(dict_iter)
    assert isinstance(record, dict)
    assert set(record.keys()) == FULLTEXTS_AVAILABLE_COLUMNS_SET


def test_current_fulltexts_view_single_run(timdex_dataset_for_fulltexts_views):
    td = timdex_dataset_for_fulltexts_views

    # write fulltexts for run "apple-1"
    td.fulltexts.write(generate_sample_fulltexts_for_run(td, run_id="apple-1"))

    # rebuild metadata to include fulltexts, then refresh
    td.metadata.rebuild_dataset_metadata()
    td.refresh()

    # query current_fulltexts for apple source using read_dataframe
    result = td.fulltexts.read_dataframe(table="current_fulltexts", source="apple")

    assert len(result) == 10
    assert (result["run_id"] == "apple-1").all()
    assert (result["run_date"] == pd.Timestamp("2025-06-01")).all()


def test_current_fulltexts_view_multiple_runs(timdex_dataset_for_fulltexts_views):
    td = timdex_dataset_for_fulltexts_views

    # write fulltexts for runs "orange-1" and "orange-2"
    td.fulltexts.write(generate_sample_fulltexts_for_run(td, run_id="orange-1"))
    td.fulltexts.write(generate_sample_fulltexts_for_run(td, run_id="orange-2"))

    # rebuild metadata to include fulltexts, then refresh
    td.metadata.rebuild_dataset_metadata()
    td.refresh()

    # query current_fulltexts for orange source using read_dataframe
    result = td.fulltexts.read_dataframe(table="current_fulltexts", source="orange")

    # 10 total current fulltexts:
    # 5 from orange-1 (offsets 5-9), 5 from orange-2 (offsets 0-4)
    assert len(result) == 10

    # verify 5 from orange-1 (records not in orange-2, run_date 2025-07-01)
    orange_1_records = result[result["run_id"] == "orange-1"]
    assert len(orange_1_records) == 5
    assert (orange_1_records["run_date"] == pd.Timestamp("2025-07-01")).all()

    # verify 5 from orange-2 (newer records, run_date 2025-07-02)
    orange_2_records = result[result["run_id"] == "orange-2"]
    assert len(orange_2_records) == 5
    assert (orange_2_records["run_date"] == pd.Timestamp("2025-07-02")).all()


def test_current_fulltexts_prefers_record_recency_over_fulltext_recency(
    tmp_path,
):
    td = TIMDEXDataset(str(tmp_path / "record_recency_wins_dataset/"))

    td.records.write(
        generate_sample_records(
            num_records=10,
            source="pear",
            run_date="2025-09-01",
            run_type="full",
            run_id="pear-1",
        ),
        write_append_deltas=False,
    )
    td.records.write(
        generate_sample_records(
            num_records=5,
            source="pear",
            run_date="2025-09-02",
            run_type="daily",
            run_id="pear-2",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()
    td.refresh()

    # older record version gets a later fulltext event
    td.fulltexts.write(
        generate_sample_fulltexts_for_run(
            td,
            run_id="pear-1",
            fulltext_timestamp="2025-09-04T00:00:00+00:00",
        ),
        write_append_deltas=False,
    )
    td.fulltexts.write(
        generate_sample_fulltexts_for_run(
            td,
            run_id="pear-2",
            fulltext_timestamp="2025-09-03T00:00:00+00:00",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()
    td.refresh()

    result = td.fulltexts.read_dataframe(table="current_fulltexts", source="pear")

    assert len(result) == 10

    overlap = result[result["timdex_record_id"].isin([f"pear:{i}" for i in range(5)])]
    non_overlap = result[
        result["timdex_record_id"].isin([f"pear:{i}" for i in range(5, 10)])
    ]

    assert (overlap["run_id"] == "pear-2").all()
    assert (non_overlap["run_id"] == "pear-1").all()
    assert (
        overlap["fulltext_timestamp"] == pd.Timestamp("2025-09-03T00:00:00+00:00")
    ).all()


def test_current_fulltexts_excludes_superseded_record_versions(tmp_path):
    td = TIMDEXDataset(str(tmp_path / "current_fulltexts_current_records_only/"))

    td.records.write(
        generate_sample_records(
            num_records=10,
            source="grape",
            run_date="2025-09-01",
            run_type="full",
            run_id="grape-1",
        ),
        write_append_deltas=False,
    )
    td.records.write(
        generate_sample_records(
            num_records=5,
            source="grape",
            run_date="2025-09-02",
            run_type="daily",
            run_id="grape-2",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()

    td.fulltexts.write(
        generate_sample_fulltexts_for_run(td, run_id="grape-1"),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()
    td.refresh()

    result = td.fulltexts.read_dataframe(table="current_fulltexts", source="grape")

    assert len(result) == 5
    assert set(result["timdex_record_id"]) == {f"grape:{i}" for i in range(5, 10)}
    assert (result["run_id"] == "grape-1").all()


def test_current_fulltexts_view_handles_duplicate_run_fulltexts(
    tmp_path,
):
    """Test that duplicate fulltexts for the same run are handled correctly."""
    td = TIMDEXDataset(str(tmp_path / "dup_run_dataset/"))

    # scenario: lemon - full run + daily run (daily will have fulltexts written twice)
    td.records.write(
        generate_sample_records(
            num_records=10,
            source="lemon",
            run_date="2025-08-01",
            run_type="full",
            run_id="lemon-1",
        ),
        write_append_deltas=False,
    )
    td.records.write(
        generate_sample_records(
            num_records=5,
            source="lemon",
            run_date="2025-08-02",
            run_type="daily",
            run_id="lemon-2",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()
    td = TIMDEXDataset(td.location)

    # write fulltexts for run "lemon-1"
    td.fulltexts.write(generate_sample_fulltexts_for_run(td, run_id="lemon-1"))

    # first fulltexts write for run "lemon-2"
    td.fulltexts.write(
        generate_sample_fulltexts_for_run(
            td, run_id="lemon-2", fulltext_timestamp="2025-08-02T00:00:00+00:00"
        )
    )

    # second fulltexts write for run "lemon-2" with a later timestamp
    td.fulltexts.write(
        generate_sample_fulltexts_for_run(
            td, run_id="lemon-2", fulltext_timestamp="2025-08-03T00:00:00+00:00"
        )
    )

    # rebuild metadata to include fulltexts, then refresh
    td.metadata.rebuild_dataset_metadata()
    td.refresh()

    # check all fulltexts for lemon-2 to verify both writes exist
    all_lemon_2 = td.fulltexts.read_dataframe(table="fulltexts", run_id="lemon-2")
    # should have 10 rows (5 from first write, 5 from second write)
    assert len(all_lemon_2) == 10

    # verify both timestamps exist
    unique_timestamps = all_lemon_2["fulltext_timestamp"].unique()
    assert len(unique_timestamps) == 2

    # query current_fulltexts for lemon source
    result = td.fulltexts.read_dataframe(table="current_fulltexts", source="lemon")

    # 10 current fulltexts: 5 from lemon-1, 5 from lemon-2 (latest timestamp)
    assert len(result) == 10

    # verify lemon-1 fulltexts (run_date 2025-08-01)
    lemon_1_result = result[result["run_id"] == "lemon-1"]
    assert len(lemon_1_result) == 5
    assert (lemon_1_result["run_date"] == pd.Timestamp("2025-08-01")).all()

    # verify lemon-2 fulltexts have the later fulltext timestamp (run_date 2025-08-02)
    lemon_2_result = result[result["run_id"] == "lemon-2"]
    assert len(lemon_2_result) == 5
    assert (lemon_2_result["run_date"] == pd.Timestamp("2025-08-02")).all()

    # all lemon-2 current fulltexts should have the later fulltext timestamp
    max_timestamp = all_lemon_2["fulltext_timestamp"].max()
    assert (lemon_2_result["fulltext_timestamp"] == max_timestamp).all()


def test_fulltexts_view_includes_all_fulltexts(tmp_path):
    """Test that the fulltexts view includes all fulltexts from multiple writes."""
    td = TIMDEXDataset(str(tmp_path / "all_fulltexts_dataset/"))

    # scenario: lemon - full run + daily run (daily will have fulltexts written twice)
    td.records.write(
        generate_sample_records(
            num_records=10,
            source="lemon",
            run_date="2025-08-01",
            run_type="full",
            run_id="lemon-1",
        ),
        write_append_deltas=False,
    )
    td.records.write(
        generate_sample_records(
            num_records=5,
            source="lemon",
            run_date="2025-08-02",
            run_type="daily",
            run_id="lemon-2",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()
    td = TIMDEXDataset(td.location)

    # write fulltexts for lemon-1
    td.fulltexts.write(generate_sample_fulltexts_for_run(td, run_id="lemon-1"))

    # write fulltexts for lemon-2 (first time) with explicit timestamp
    td.fulltexts.write(
        generate_sample_fulltexts_for_run(
            td, run_id="lemon-2", fulltext_timestamp="2025-08-02T00:00:00+00:00"
        )
    )

    # write fulltexts for lemon-2 again with later timestamp
    td.fulltexts.write(
        generate_sample_fulltexts_for_run(
            td, run_id="lemon-2", fulltext_timestamp="2025-08-03T00:00:00+00:00"
        )
    )

    # rebuild metadata to include fulltexts, then refresh
    td.metadata.rebuild_dataset_metadata()
    td.refresh()

    # query all fulltexts for lemon source
    result = td.fulltexts.read_dataframe(table="fulltexts", source="lemon")

    # 20 total fulltexts: 10 from lemon-1, 5 from lemon-2 first write,
    # 5 from lemon-2 second write
    assert len(result) == 20

    # verify run_date distribution
    lemon_1_fulltexts = result[result["run_id"] == "lemon-1"]
    assert len(lemon_1_fulltexts) == 10
    assert (lemon_1_fulltexts["run_date"] == pd.Timestamp("2025-08-01")).all()

    lemon_2_fulltexts = result[result["run_id"] == "lemon-2"]
    assert len(lemon_2_fulltexts) == 10  # 5 from each write
    assert (lemon_2_fulltexts["run_date"] == pd.Timestamp("2025-08-02")).all()


def test_fulltexts_read_batches_iter_returns_empty_when_fulltexts_missing(
    timdex_dataset_empty,
):
    with pytest.raises(
        ValueError,
        match=r"Table 'fulltexts' not found in DuckDB context.*rebuild_dataset_metadata",
    ):
        list(timdex_dataset_empty.fulltexts.read_batches_iter())


def test_fulltexts_read_batches_iter_returns_empty_for_invalid_table(
    timdex_fulltexts_with_runs,
):
    """read_batches_iter returns empty iterator for nonexistent table name."""
    with pytest.raises(
        ValueError,
        match="Invalid table: 'nonexistent'",
    ):
        list(timdex_fulltexts_with_runs.read_batches_iter(table="nonexistent"))
