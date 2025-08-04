from duckdb import DuckDBPyConnection

from timdex_dataset_api import TIMDEXDatasetMetadata


def test_tdm_init_no_metadata_file_warning_success(caplog, dataset_with_runs_location):
    tdm = TIMDEXDatasetMetadata(dataset_with_runs_location)

    assert tdm.conn is None
    assert "Static metadata database not found" in caplog.text


def test_tdm_local_dataset_structure_properties():
    local_root = "/path/to/nothing"
    tdm_local = TIMDEXDatasetMetadata(local_root)
    assert tdm_local.location == local_root
    assert tdm_local.location_scheme == "file"


def test_tdm_s3_dataset_structure_properties(mocked_timdex_bucket):
    s3_root = "s3://timdex/dataset"
    tdm_s3 = TIMDEXDatasetMetadata(s3_root)
    assert tdm_s3.location == s3_root
    assert tdm_s3.location_scheme == "s3"


def test_tdm_create_metadata_database_file_success(caplog, timdex_dataset_metadata_empty):
    caplog.set_level("DEBUG")
    timdex_dataset_metadata_empty.recreate_static_database_file()


def test_tdm_init_metadata_file_found_success(timdex_dataset_metadata):
    assert isinstance(timdex_dataset_metadata.conn, DuckDBPyConnection)


def test_tdm_connection_has_static_database_attached(timdex_dataset_metadata):
    assert set(
        timdex_dataset_metadata.conn.query("""show databases;""").to_df().database_name
    ) == {"memory", "static_db"}


def test_tdm_connection_static_database_records_table_exists(timdex_dataset_metadata):
    records_df = timdex_dataset_metadata.conn.query(
        """select * from static_db.records;"""
    ).to_df()
    assert len(records_df) > 0
