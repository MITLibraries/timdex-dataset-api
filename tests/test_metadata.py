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


def test_tdm_connection_has_static_database_attached(timdex_dataset_metadata):
    assert set(
        timdex_dataset_metadata.conn.query("""show databases;""").to_df().database_name
    ) == {"memory", "static_db"}


def test_tdm_connection_static_database_records_table_exists(timdex_dataset_metadata):
    records_df = timdex_dataset_metadata.conn.query(
        """select * from static_db.records;"""
    ).to_df()
    assert len(records_df) > 0
