# ruff: noqa: S105, S108

import glob
import os
from pathlib import Path

from duckdb import DuckDBPyConnection

from timdex_dataset_api import TIMDEXDataset, TIMDEXDatasetMetadata

ORDERED_METADATA_COLUMN_NAMES = [
    "timdex_record_id",
    "source",
    "run_date",
    "run_type",
    "action",
    "run_id",
    "run_record_offset",
    "run_timestamp",
    "filename",
]


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
    timdex_metadata_empty.rebuild_dataset_metadata()


def test_tdm_init_metadata_file_found_success(timdex_metadata):
    assert isinstance(timdex_metadata.conn, DuckDBPyConnection)


def test_tdm_duckdb_context_creates_metadata_schema(timdex_metadata):
    assert (
        timdex_metadata.conn.query(
            """
            select count(*)
            from information_schema.schemata
            where catalog_name = 'memory'
            and schema_name = 'metadata';
            """
        ).fetchone()[0]
        == 1
    )


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
    records_df = timdex_metadata.conn.query(
        """select * from metadata.records limit 1;"""
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
    assert set(records_df.columns) == expected_columns


def test_tdm_current_records_view_structure(timdex_metadata):
    current_records_df = timdex_metadata.conn.query(
        """select * from metadata.current_records limit 1;"""
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
        """select * from metadata.append_deltas;"""
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
        """select count(*) from metadata.records;"""
    ).fetchone()[0]
    assert timdex_metadata.records_count == manual_count


def test_tdm_current_records_count_property(timdex_metadata):
    assert timdex_metadata.current_records_count > 0

    manual_count = timdex_metadata.conn.query(
        """select count(*) from metadata.current_records;"""
    ).fetchone()[0]
    assert timdex_metadata.current_records_count == manual_count


def test_tdm_append_deltas_count_property_empty(timdex_metadata):
    assert timdex_metadata.append_deltas_count == 0


def test_tdm_records_equals_static_without_deltas(timdex_metadata):
    static_count = timdex_metadata.conn.query(
        """select count(*) from static_db.records;"""
    ).fetchone()[0]
    records_count = timdex_metadata.conn.query(
        """select count(*) from metadata.records;"""
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
        """select timdex_record_id from metadata.current_records;"""
    ).to_df()

    unique_count = len(current_records_df.timdex_record_id.unique())
    assert unique_count == current_count


def test_tdm_current_records_most_recent_version(timdex_metadata_with_deltas):
    # check that for records with multiple versions, only the most recent is returned
    multi_version_records = timdex_metadata_with_deltas.conn.query(
        """
        select timdex_record_id, count(*) as version_count
        from metadata.records
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
            from metadata.records
            where timdex_record_id = '{record_id}'
            order by run_timestamp desc
            limit 1;
            """
        ).to_df()

        # verify current_records contains this version
        current_version = timdex_metadata_with_deltas.conn.query(
            f"""
            select run_timestamp, run_id
            from metadata.current_records
            where timdex_record_id = '{record_id}';
            """
        ).to_df()

        assert len(current_version) == 1
        assert (
            current_version.iloc[0]["run_timestamp"]
            == most_recent.iloc[0]["run_timestamp"]
        )
        assert current_version.iloc[0]["run_id"] == most_recent.iloc[0]["run_id"]


def test_tdm_merge_append_deltas_static_counts_match_records_count_before_merge(
    timdex_metadata_with_deltas, timdex_metadata_merged_deltas
):
    static_count_merged_deltas = timdex_metadata_merged_deltas.conn.query(
        """select count(*) as count from static_db.records;"""
    ).fetchone()[0]
    assert static_count_merged_deltas == timdex_metadata_with_deltas.records_count


def test_tdm_merge_append_deltas_adds_records_to_static_db(
    timdex_metadata_with_deltas, timdex_metadata_merged_deltas
):
    append_deltas = timdex_metadata_with_deltas.conn.query(
        f"""
            select
            {','.join(ORDERED_METADATA_COLUMN_NAMES)}
            from metadata.append_deltas
        """
    ).to_df()

    merged_static_db = timdex_metadata_merged_deltas.conn.query(
        f"""
            select
            {','.join(ORDERED_METADATA_COLUMN_NAMES)}
            from static_db.records
        """
    ).to_df()

    assert set(map(tuple, append_deltas.to_numpy())).issubset(
        set(map(tuple, merged_static_db.to_numpy()))
    )


def test_tdm_merge_append_deltas_deletes_append_deltas(
    timdex_metadata_with_deltas, timdex_metadata_merged_deltas
):
    assert timdex_metadata_with_deltas.append_deltas_count != 0
    assert os.listdir(timdex_metadata_with_deltas.append_deltas_path)

    assert timdex_metadata_merged_deltas.append_deltas_count == 0
    assert not os.listdir(timdex_metadata_merged_deltas.append_deltas_path)


def test_tdm_prepare_duckdb_secret_and_extensions_home_env_var_set_and_valid(
    monkeypatch, tmp_path_factory, timdex_dataset_with_runs
):
    preset_home = tmp_path_factory.mktemp("my-account")
    monkeypatch.setenv("HOME", str(preset_home))

    tdm = TIMDEXDatasetMetadata(timdex_dataset_with_runs.location)
    df = (
        tdm.conn.query(
            """
        select
            current_setting('secret_directory') as secret_directory,
            current_setting('extension_directory') as extension_directory
        ;
        """
        )
        .to_df()
        .iloc[0]
    )
    assert "my-account" in df.secret_directory
    assert df.extension_directory == ""  # expected and okay when HOME set


def test_tdm_prepare_duckdb_secret_and_extensions_home_env_var_unset(
    monkeypatch, timdex_dataset_with_runs
):
    monkeypatch.delenv("HOME", raising=False)

    tdm = TIMDEXDatasetMetadata(timdex_dataset_with_runs.location)

    df = (
        tdm.conn.query(
            """
        select
            current_setting('secret_directory') as secret_directory,
            current_setting('extension_directory') as extension_directory
        ;
        """
        )
        .to_df()
        .iloc[0]
    )
    assert df.secret_directory == "/tmp/.duckdb/secrets"
    assert df.extension_directory == "/tmp/.duckdb/extensions"


def test_tdm_prepare_duckdb_secret_and_extensions_home_env_var_set_but_empty(
    monkeypatch, timdex_dataset_with_runs
):
    monkeypatch.setenv("HOME", "")  # simulate AWS Lambda environment

    tdm = TIMDEXDatasetMetadata(timdex_dataset_with_runs.location)

    df = (
        tdm.conn.query(
            """
        select
            current_setting('secret_directory') as secret_directory,
            current_setting('extension_directory') as extension_directory
        ;
        """
        )
        .to_df()
        .iloc[0]
    )
    assert df.secret_directory == "/tmp/.duckdb/secrets"
    assert df.extension_directory == "/tmp/.duckdb/extensions"


def test_tdm_preload_current_records_default_false(tmp_path):
    tdm = TIMDEXDatasetMetadata(str(tmp_path))
    assert tdm.preload_current_records is False


def test_tdm_preload_current_records_flag_true(tmp_path):
    tdm = TIMDEXDatasetMetadata(str(tmp_path), preload_current_records=True)
    assert tdm.preload_current_records is True


def test_tdm_preload_false_no_temp_table(timdex_dataset_with_runs):
    # instantiate TIMDEXDataset without preloading current records (default)
    td = TIMDEXDataset(timdex_dataset_with_runs.location)

    # assert that materialized, temporary table "temp.current_records" does not exist
    temp_table_count = td.metadata.conn.query(
        """
        select count(*)
        from information_schema.tables
        where table_catalog = 'temp'
        and table_name = 'current_records'
        and table_type = 'LOCAL TEMPORARY'
        ;
        """
    ).fetchone()[0]

    assert temp_table_count == 0


def test_tdm_preload_true_has_temp_table(timdex_dataset_with_runs):
    # instantiate TIMDEXDataset with preloading current records
    td = TIMDEXDataset(timdex_dataset_with_runs.location, preload_current_records=True)

    # assert that materialized, temporary table "temp.current_records" does exist
    temp_table_count = td.metadata.conn.query(
        """
            select count(*)
            from information_schema.tables
            where table_catalog = 'temp'
            and table_name = 'current_records'
            and table_type = 'LOCAL TEMPORARY'
            ;
            """
    ).fetchone()[0]

    assert temp_table_count == 1
