import glob
import os
from pathlib import Path

from duckdb import DuckDBPyConnection

from timdex_dataset_api import TIMDEXDatasetMetadata


def test_tdm_init_no_metadata_file_warning_success(caplog, timdex_dataset_with_runs):
    TIMDEXDatasetMetadata(timdex_dataset_with_runs.location)

    assert "Static metadata database not found" in caplog.text


def test_tdm_local_dataset_structure_properties(tmp_path):
    local_root = str(Path(tmp_path) / "path/to/nothing")
    tdm_local = TIMDEXDatasetMetadata(local_root)
    assert tdm_local.location == local_root
    assert tdm_local.location_scheme == "file"


def test_tdm_s3_dataset_structure_properties(s3_bucket_mocked):
    s3_root = "s3://timdex/dataset"
    tdm_s3 = TIMDEXDatasetMetadata(s3_root)
    assert tdm_s3.location == s3_root
    assert tdm_s3.location_scheme == "s3"


def test_tdm_create_metadata_database_file_success(caplog, timdex_metadata_empty):
    caplog.set_level("DEBUG")
    timdex_metadata_empty.recreate_static_database_file()


def test_tdm_init_metadata_file_found_success(timdex_metadata):
    assert isinstance(timdex_metadata.conn, DuckDBPyConnection)


def test_tdm_connection_has_static_database_attached(timdex_metadata):
    assert set(
        timdex_metadata.conn.query("""show databases;""").to_df().database_name
    ) == {"memory", "static_db"}


def test_tdm_connection_static_database_records_table_exists(timdex_metadata):
    records_df = timdex_metadata.conn.query(
        """select * from static_db.records;"""
    ).to_df()
    assert len(records_df) > 0


def test_dataset_metadata_structure_is_idempotent(timdex_metadata):
    assert os.path.exists(timdex_metadata.metadata_root)
    start_file_count = glob.glob(f"{timdex_metadata.metadata_root}/**/*")

    timdex_metadata.create_metadata_structure()

    assert os.path.exists(timdex_metadata.metadata_root)
    end_file_count = glob.glob(f"{timdex_metadata.metadata_root}/**/*")
    assert start_file_count == end_file_count


def test_tdm_views_created_on_init(timdex_metadata):
    views = timdex_metadata.conn.query(
        """select table_name from information_schema.tables where table_type = 'VIEW';"""
    ).to_df()

    expected_views = {"append_deltas", "records", "current_records"}
    actual_views = set(views.table_name)
    assert expected_views <= actual_views


def test_tdm_records_view_structure(timdex_metadata):
    records_df = timdex_metadata.conn.query("""select * from records limit 1;""").to_df()
    expected_columns = {
        "timdex_record_id",
        "source",
        "run_date",
        "run_type",
        "action",
        "run_id",
        "run_record_offset",
        "run_timestamp",
        "filename",
    }
    assert set(records_df.columns) == expected_columns


def test_tdm_current_records_view_structure(timdex_metadata):
    current_records_df = timdex_metadata.conn.query(
        """select * from current_records limit 1;"""
    ).to_df()
    expected_columns = {
        "timdex_record_id",
        "source",
        "run_date",
        "run_type",
        "action",
        "run_id",
        "run_record_offset",
        "run_timestamp",
        "filename",
    }
    assert set(current_records_df.columns) == expected_columns


def test_tdm_append_deltas_view_empty_structure(timdex_metadata):
    append_deltas_df = timdex_metadata.conn.query(
        """select * from append_deltas;"""
    ).to_df()
    expected_columns = {
        "timdex_record_id",
        "source",
        "run_date",
        "run_type",
        "action",
        "run_id",
        "run_record_offset",
        "run_timestamp",
        "filename",
    }
    assert set(append_deltas_df.columns) == expected_columns
    assert len(append_deltas_df) == 0


def test_tdm_records_count_property(timdex_metadata):
    assert timdex_metadata.records_count > 0

    manual_count = timdex_metadata.conn.query(
        """select count(*) from records;"""
    ).fetchone()[0]
    assert timdex_metadata.records_count == manual_count


def test_tdm_current_records_count_property(timdex_metadata):
    assert timdex_metadata.current_records_count > 0

    manual_count = timdex_metadata.conn.query(
        """select count(*) from current_records;"""
    ).fetchone()[0]
    assert timdex_metadata.current_records_count == manual_count


def test_tdm_append_deltas_count_property_empty(timdex_metadata):
    assert timdex_metadata.append_deltas_count == 0


def test_tdm_records_equals_static_without_deltas(timdex_metadata):
    static_count = timdex_metadata.conn.query(
        """select count(*) from static_db.records;"""
    ).fetchone()[0]
    records_count = timdex_metadata.conn.query(
        """select count(*) from records;"""
    ).fetchone()[0]
    assert static_count == records_count


def test_tdm_current_records_filtering_logic(timdex_metadata):
    current_count = timdex_metadata.current_records_count
    total_count = timdex_metadata.records_count

    assert current_count <= total_count
    assert current_count > 0


def test_tdm_views_with_append_deltas(timdex_metadata_with_deltas):
    views = timdex_metadata_with_deltas.conn.query(
        """select table_name from information_schema.tables where table_type = 'VIEW';"""
    ).to_df()

    expected_views = {"append_deltas", "records", "current_records"}
    actual_views = set(views.table_name)
    assert expected_views.issubset(actual_views)


def test_tdm_append_deltas_view_has_data(timdex_metadata_with_deltas):
    append_deltas_count = timdex_metadata_with_deltas.append_deltas_count
    assert append_deltas_count > 0


def test_tdm_records_includes_deltas(timdex_metadata_with_deltas):
    static_count = timdex_metadata_with_deltas.conn.query(
        """select count(*) from static_db.records;"""
    ).fetchone()[0]
    deltas_count = timdex_metadata_with_deltas.append_deltas_count
    records_count = timdex_metadata_with_deltas.records_count

    assert records_count == static_count + deltas_count
    assert records_count > static_count


def test_tdm_current_records_with_deltas_logic(timdex_metadata_with_deltas):
    current_count = timdex_metadata_with_deltas.current_records_count
    total_count = timdex_metadata_with_deltas.records_count

    assert current_count <= total_count
    assert current_count > 0

    # verify current records view returns unique timdex_record_id values
    current_records_df = timdex_metadata_with_deltas.conn.query(
        """select timdex_record_id from current_records;"""
    ).to_df()

    unique_count = len(current_records_df.timdex_record_id.unique())
    assert unique_count == current_count


def test_tdm_current_records_most_recent_version(timdex_metadata_with_deltas):
    # check that for records with multiple versions, only the most recent is returned
    multi_version_records = timdex_metadata_with_deltas.conn.query(
        """
        select timdex_record_id, count(*) as version_count
        from records
        group by timdex_record_id
        having count(*) > 1
        limit 1;
        """
    ).to_df()

    if len(multi_version_records) > 0:
        record_id = multi_version_records.iloc[0]["timdex_record_id"]

        # get most recent timestamp for this record
        most_recent = timdex_metadata_with_deltas.conn.query(
            f"""
            select run_timestamp, run_id
            from records
            where timdex_record_id = '{record_id}'
            order by run_timestamp desc
            limit 1;
            """
        ).to_df()

        # verify current_records contains this version
        current_version = timdex_metadata_with_deltas.conn.query(
            f"""
            select run_timestamp, run_id
            from current_records
            where timdex_record_id = '{record_id}';
            """
        ).to_df()

        assert len(current_version) == 1
        assert (
            current_version.iloc[0]["run_timestamp"]
            == most_recent.iloc[0]["run_timestamp"]
        )
        assert current_version.iloc[0]["run_id"] == most_recent.iloc[0]["run_id"]
