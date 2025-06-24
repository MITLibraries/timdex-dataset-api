# ruff: noqa: PLR2004

import duckdb

from timdex_dataset_api import TIMDEXDataset, TIMDEXDatasetMetadata


def test_tdm_init_from_timdex_dataset_instance_success(dataset_with_same_day_runs):
    tdm = TIMDEXDatasetMetadata(timdex_dataset=dataset_with_same_day_runs)
    assert isinstance(tdm.timdex_dataset, TIMDEXDataset)


def test_tdm_init_from_timdex_dataset_path_success(dataset_with_runs_location):
    tdm = TIMDEXDatasetMetadata.from_dataset_location(dataset_with_runs_location)
    assert isinstance(tdm.timdex_dataset, TIMDEXDataset)


def test_tdm_default_database_location_in_memory(timdex_dataset_metadata):
    assert timdex_dataset_metadata.db_path == ":memory:"
    result = timdex_dataset_metadata.conn.query("PRAGMA database_list;").fetchone()
    assert result[1] == "memory"  # name of database
    assert result[2] is None  # file associated with database, where None is memory


def test_tdm_explicit_database_in_file(tmp_path, dataset_with_runs_location):
    db_path = str(tmp_path / "tda.duckdb")
    tdm = TIMDEXDatasetMetadata.from_dataset_location(
        dataset_with_runs_location,
        db_path=db_path,
    )
    assert tdm.db_path == db_path
    result = tdm.conn.query("PRAGMA database_list;").fetchone()
    assert result[1] == "tda"  # name of database
    assert result[2] == db_path  # filepath passed during init


def test_tdm_get_duckdb_connection(timdex_dataset_metadata):
    conn = timdex_dataset_metadata.get_connection()
    assert isinstance(conn, duckdb.DuckDBPyConnection)


def test_tdm_set_threads(timdex_dataset_metadata):
    # set to 64
    timdex_dataset_metadata.set_database_thread_usage(64)
    sixty_four_thread_count = timdex_dataset_metadata.conn.query(
        """SELECT current_setting('threads');"""
    ).fetchone()[0]
    assert sixty_four_thread_count == 64

    # set to 12
    timdex_dataset_metadata.set_database_thread_usage(12)
    sixty_four_thread_count = timdex_dataset_metadata.conn.query(
        """SELECT current_setting('threads');"""
    ).fetchone()[0]
    assert sixty_four_thread_count == 12


def test_tdm_init_sets_up_database(timdex_dataset_metadata):
    df = timdex_dataset_metadata.conn.query("show tables;").to_df()
    assert set(df.name) == {"current_records", "records"}


def test_tdm_get_current_parquet_files(timdex_dataset_metadata):
    parquet_files = timdex_dataset_metadata.get_current_parquet_files()
    # assert 5 total parquet files in dataset
    # but only 3 contain current records
    assert len(timdex_dataset_metadata.timdex_dataset.dataset.files) == 5
    assert len(parquet_files) == 3


def test_tdm_get_record_to_run_mapping(timdex_dataset_metadata):
    record_map = timdex_dataset_metadata.get_current_record_to_run_map()

    assert len(record_map) == 75
    assert record_map["alma:0"] == "run-5"
    assert record_map["alma:5"] == "run-4"
    assert record_map["alma:19"] == "run-4"
    assert "run-3" not in record_map.values()
    assert record_map["alma:20"] == "run-2"


def test_tdm_current_records_subset_of_all_records(timdex_dataset_metadata):
    records_df = timdex_dataset_metadata.conn.query("select * from records;").to_df()
    current_records_df = timdex_dataset_metadata.conn.query(
        "select * from current_records;"
    ).to_df()
    assert set(current_records_df.timdex_record_id).issubset(
        set(records_df.timdex_record_id)
    )
