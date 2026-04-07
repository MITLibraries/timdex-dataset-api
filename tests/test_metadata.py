# ruff: noqa: S105, S108

import glob
import os
from pathlib import Path

from duckdb import DuckDBPyConnection

from tests.utils import generate_sample_embeddings_for_run, generate_sample_records
from timdex_dataset_api import TIMDEXDataset
from timdex_dataset_api.embeddings import TIMDEXEmbeddings
from timdex_dataset_api.metadata import DataTypeMetadataConfig, TIMDEXDatasetMetadata
from timdex_dataset_api.records import TIMDEXRecords


def test_tdm_init_no_metadata_file_warning_success(caplog, tmp_path):
    # creating a new TIMDEXDataset will log warning if no metadata file
    caplog.set_level("WARNING")
    TIMDEXDataset(str(tmp_path / "new_empty_dataset"))
    assert "Static metadata database not found" in caplog.text


def test_tdm_local_dataset_structure_properties(tmp_path):
    local_root = str(Path(tmp_path) / "path/to/nothing")
    td_local = TIMDEXDataset(local_root)
    assert td_local.location == local_root
    assert td_local.location_scheme == "file"


def test_tdm_s3_dataset_structure_properties(timdex_dataset_empty):
    # test that location_scheme property works correctly for local paths
    # S3 tests require full mocking and are covered in other tests
    assert timdex_dataset_empty.location_scheme == "file"


def test_data_type_metadata_config_prejoin_records_default_true():
    config = DataTypeMetadataConfig(
        name="example",
        metadata_columns=["timdex_record_id"],
        data_path="data/example",
    )
    assert config.prejoin_records is True


def test_data_source_metadata_configs_are_derived_from_base_class():
    assert TIMDEXRecords.METADATA_CONFIG.name == TIMDEXRecords.NAME
    assert TIMDEXRecords.METADATA_CONFIG.data_path == TIMDEXRecords.DATA_PATH
    assert TIMDEXRecords.METADATA_CONFIG.prejoin_records is False
    assert (
        TIMDEXRecords.METADATA_CONFIG.metadata_columns
        == TIMDEXDatasetMetadata.BASE_METADATA_COLUMNS
    )

    assert TIMDEXEmbeddings.METADATA_CONFIG.name == TIMDEXEmbeddings.NAME
    assert TIMDEXEmbeddings.METADATA_CONFIG.data_path == TIMDEXEmbeddings.DATA_PATH
    assert TIMDEXEmbeddings.METADATA_CONFIG.prejoin_records is True
    assert TIMDEXEmbeddings.METADATA_CONFIG.metadata_columns == [
        "timdex_record_id",
        "run_id",
        "run_record_offset",
        *TIMDEXEmbeddings.ADDITIONAL_METADATA_COLUMNS,
        "filename",
    ]


def test_dataset_registers_current_view_specs_from_data_sources(tmp_path):
    td = TIMDEXDataset(str(tmp_path / "register_current_view_specs"))

    expected_view_names = [
        spec.name
        for spec in (
            TIMDEXRecords.CURRENT_VIEW_SPECS + TIMDEXEmbeddings.CURRENT_VIEW_SPECS
        )
    ]
    assert [spec.name for spec in td.current_metadata_view_specs] == expected_view_names


def test_tdm_create_metadata_database_file_success(
    caplog, timdex_dataset_with_runs, timdex_metadata_empty
):
    caplog.set_level("DEBUG")
    # use a fresh dataset from timdex_dataset_with_runs location
    td = TIMDEXDataset(timdex_dataset_with_runs.location)
    td.metadata.rebuild_dataset_metadata()


def test_tdm_init_metadata_file_found_success(timdex_metadata):
    assert isinstance(timdex_metadata.timdex_dataset.conn, DuckDBPyConnection)


def test_tdm_duckdb_context_creates_metadata_schema(timdex_metadata):
    assert (
        timdex_metadata.timdex_dataset.conn.query("""
            select count(*)
            from information_schema.schemata
            where catalog_name = 'memory'
            and schema_name = 'metadata';
            """).fetchone()[0]
        == 1
    )


def test_tdm_connection_has_static_database_attached(timdex_metadata):
    assert set(
        timdex_metadata.timdex_dataset.conn.query("""show databases;""")
        .to_df()
        .database_name
    ) == {"memory", "static_db"}


def test_tdm_connection_static_database_records_table_exists(timdex_metadata):
    records_df = timdex_metadata.timdex_dataset.conn.query(
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
    views = timdex_metadata.timdex_dataset.conn.query(
        """select table_name from information_schema.tables where table_type = 'VIEW';"""
    ).to_df()

    expected_views = {"records_append_deltas", "records", "current_records"}
    actual_views = set(views.table_name)
    assert expected_views <= actual_views


def test_tdm_current_view_specs_missing_dependencies_are_skipped_generically(
    caplog, tmp_path
):
    dataset_path = str(tmp_path / "current_view_missing_dependencies")

    td = TIMDEXDataset(dataset_path)
    td.records.write(
        generate_sample_records(
            num_records=10,
            source="alma",
            run_date="2025-03-01",
            run_type="full",
            run_id="missing-deps-run",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()

    caplog.set_level("WARNING")
    caplog.clear()

    td_with_metadata = TIMDEXDataset(dataset_path)

    metadata_objects = td_with_metadata.conn.query("""
        select table_name
        from information_schema.tables
        where table_schema = 'metadata'
    """).to_df()
    metadata_names = set(metadata_objects.table_name)

    missing_specs = []
    for spec in td_with_metadata.current_metadata_view_specs:
        missing_required_tables = [
            table_name
            for table_name in spec.required_metadata_tables
            if table_name not in metadata_names
        ]
        if not missing_required_tables:
            continue

        missing_specs.append(spec.name)
        assert spec.name not in metadata_names
        assert (
            "Skipping metadata."
            f"{spec.name} view creation because missing dependencies: "
            f"{', '.join(missing_required_tables)}"
        ) in caplog.text

    assert missing_specs


def test_tdm_records_view_structure(timdex_metadata):
    records_df = timdex_metadata.timdex_dataset.conn.query(
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
    current_records_df = timdex_metadata.timdex_dataset.conn.query(
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
    append_deltas_df = timdex_metadata.timdex_dataset.conn.query(
        """select * from metadata.records_append_deltas;"""
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
        "append_delta_filename",
    }
    assert set(append_deltas_df.columns) == expected_columns
    assert len(append_deltas_df) == 0


def test_tdm_records_count_property(timdex_metadata):
    assert timdex_metadata.records_count > 0

    manual_count = timdex_metadata.timdex_dataset.conn.query(
        """select count(*) from metadata.records;"""
    ).fetchone()[0]
    assert timdex_metadata.records_count == manual_count


def test_tdm_current_records_count_property(timdex_metadata):
    assert timdex_metadata.current_records_count > 0

    manual_count = timdex_metadata.timdex_dataset.conn.query(
        """select count(*) from metadata.current_records;"""
    ).fetchone()[0]
    assert timdex_metadata.current_records_count == manual_count


def test_tdm_append_deltas_count_property_empty(timdex_metadata):
    assert timdex_metadata.append_deltas_count == 0


def test_tdm_records_equals_static_without_deltas(timdex_metadata):
    static_count = timdex_metadata.timdex_dataset.conn.query(
        """select count(*) from static_db.records;"""
    ).fetchone()[0]
    records_count = timdex_metadata.timdex_dataset.conn.query(
        """select count(*) from metadata.records;"""
    ).fetchone()[0]
    assert static_count == records_count


def test_tdm_current_records_filtering_logic(timdex_metadata):
    current_count = timdex_metadata.current_records_count
    total_count = timdex_metadata.records_count

    assert current_count <= total_count
    assert current_count > 0


def test_tdm_views_with_append_deltas(timdex_metadata_with_deltas):
    views = timdex_metadata_with_deltas.timdex_dataset.conn.query(
        """select table_name from information_schema.tables where table_type = 'VIEW';"""
    ).to_df()

    expected_views = {"records_append_deltas", "records", "current_records"}
    actual_views = set(views.table_name)
    assert expected_views.issubset(actual_views)


def test_tdm_append_deltas_view_has_data(timdex_metadata_with_deltas):
    append_deltas_count = timdex_metadata_with_deltas.append_deltas_count
    assert append_deltas_count > 0


def test_tdm_records_includes_deltas(timdex_metadata_with_deltas):
    static_count = timdex_metadata_with_deltas.timdex_dataset.conn.query(
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
    current_records_df = timdex_metadata_with_deltas.timdex_dataset.conn.query(
        """select timdex_record_id from metadata.current_records;"""
    ).to_df()

    unique_count = len(current_records_df.timdex_record_id.unique())
    assert unique_count == current_count


def test_tdm_current_records_most_recent_version(timdex_metadata_with_deltas):
    # check that for records with multiple versions, only the most recent is returned
    multi_version_records = timdex_metadata_with_deltas.timdex_dataset.conn.query("""
        select timdex_record_id, count(*) as version_count
        from metadata.records
        group by timdex_record_id
        having count(*) > 1
        limit 1;
        """).to_df()

    if len(multi_version_records) > 0:
        record_id = multi_version_records.iloc[0]["timdex_record_id"]

        # get most recent timestamp for this record
        most_recent = timdex_metadata_with_deltas.timdex_dataset.conn.query(f"""
            select run_timestamp, run_id
            from metadata.records
            where timdex_record_id = '{record_id}'
            order by run_timestamp desc
            limit 1;
            """).to_df()

        # verify current_records contains this version
        current_version = timdex_metadata_with_deltas.timdex_dataset.conn.query(f"""
            select run_timestamp, run_id
            from metadata.current_records
            where timdex_record_id = '{record_id}';
            """).to_df()

        assert len(current_version) == 1
        assert (
            current_version.iloc[0]["run_timestamp"]
            == most_recent.iloc[0]["run_timestamp"]
        )
        assert current_version.iloc[0]["run_id"] == most_recent.iloc[0]["run_id"]


def test_tdm_merge_append_deltas_static_counts_match_records_count_before_merge(
    timdex_metadata_with_deltas, timdex_metadata_merged_deltas
):
    static_count_merged_deltas = timdex_metadata_merged_deltas.timdex_dataset.conn.query(
        """select count(*) as count from static_db.records;"""
    ).fetchone()[0]
    assert static_count_merged_deltas == timdex_metadata_with_deltas.records_count


def test_tdm_merge_append_deltas_adds_records_to_static_db(
    timdex_metadata_with_deltas, timdex_metadata_merged_deltas
):
    columns = ",".join(TIMDEXRecords.METADATA_CONFIG.metadata_columns)
    append_deltas = timdex_metadata_with_deltas.timdex_dataset.conn.query(f"""
            select
            {columns}
            from metadata.records_append_deltas
        """).to_df()

    merged_static_db = timdex_metadata_merged_deltas.timdex_dataset.conn.query(f"""
            select
            {columns}
            from static_db.records
        """).to_df()

    assert set(map(tuple, append_deltas.to_numpy())).issubset(
        set(map(tuple, merged_static_db.to_numpy()))
    )


def test_tdm_merge_append_deltas_deletes_append_deltas(
    timdex_metadata_with_deltas, timdex_metadata_merged_deltas
):
    records_deltas_path_before = timdex_metadata_with_deltas.append_deltas_path_for(
        TIMDEXRecords.METADATA_CONFIG
    )
    records_deltas_path_after = timdex_metadata_merged_deltas.append_deltas_path_for(
        TIMDEXRecords.METADATA_CONFIG
    )

    assert timdex_metadata_with_deltas.append_deltas_count != 0
    assert os.listdir(records_deltas_path_before)

    assert timdex_metadata_merged_deltas.append_deltas_count == 0
    assert not os.listdir(records_deltas_path_after)


def test_tdm_embeddings_metadata_view_structure(tmp_path):
    td = TIMDEXDataset(str(tmp_path / "embeddings_metadata_structure"))

    td.records.write(
        generate_sample_records(
            num_records=25,
            source="alma",
            run_date="2025-03-01",
            run_type="full",
            run_id="emb-structure-run",
        ),
        write_append_deltas=False,
    )

    td.metadata.rebuild_dataset_metadata()

    td.embeddings.write(
        generate_sample_embeddings_for_run(td, run_id="emb-structure-run"),
        write_append_deltas=False,
    )

    td.metadata.rebuild_dataset_metadata()

    embeddings_df = td.conn.query(
        """select * from metadata.embeddings limit 1;"""
    ).to_df()
    assert len(embeddings_df) == 1
    # pre-joined view includes native embeddings columns + records columns
    expected_columns = set(TIMDEXEmbeddings.METADATA_CONFIG.metadata_columns) | {
        "source",
        "run_date",
        "run_type",
        "action",
        "run_timestamp",
    }
    assert set(embeddings_df.columns) == expected_columns


def test_tdm_current_embeddings_view_structure(tmp_path):
    td = TIMDEXDataset(str(tmp_path / "current_embeddings_structure"))

    td.records.write(
        generate_sample_records(
            num_records=25,
            source="alma",
            run_date="2025-03-01",
            run_type="full",
            run_id="emb-current-structure-run",
        ),
        write_append_deltas=False,
    )

    td.metadata.rebuild_dataset_metadata()

    td.embeddings.write(
        generate_sample_embeddings_for_run(td, run_id="emb-current-structure-run"),
        write_append_deltas=False,
    )

    td.metadata.rebuild_dataset_metadata()

    current_embeddings_df = td.conn.query(
        """select * from metadata.current_embeddings limit 1;"""
    ).to_df()

    assert len(current_embeddings_df) == 1
    # pre-joined view includes native embeddings columns + records columns
    expected_columns = set(TIMDEXEmbeddings.METADATA_CONFIG.metadata_columns) | {
        "source",
        "run_date",
        "run_type",
        "action",
        "run_timestamp",
    }
    assert set(current_embeddings_df.columns) == expected_columns


def test_tdm_current_embeddings_latest_per_record_strategy(tmp_path):
    td = TIMDEXDataset(str(tmp_path / "current_embeddings_latest"))

    td.records.write(
        generate_sample_records(
            num_records=10,
            source="alma",
            run_date="2025-03-01",
            run_type="full",
            run_id="emb-current-latest-run-1",
        ),
        write_append_deltas=False,
    )
    td.records.write(
        generate_sample_records(
            num_records=5,
            source="alma",
            run_date="2025-03-02",
            run_type="daily",
            run_id="emb-current-latest-run-2",
        ),
        write_append_deltas=False,
    )

    td.metadata.rebuild_dataset_metadata()

    td.embeddings.write(
        generate_sample_embeddings_for_run(
            td,
            run_id="emb-current-latest-run-1",
            embedding_timestamp="2025-03-10T00:00:00+00:00",
        ),
        write_append_deltas=False,
    )
    td.embeddings.write(
        generate_sample_embeddings_for_run(
            td,
            run_id="emb-current-latest-run-2",
            embedding_timestamp="2025-03-11T00:00:00+00:00",
        ),
        write_append_deltas=False,
    )

    td.metadata.rebuild_dataset_metadata()

    current_embeddings_df = td.conn.query("""
        select
            timdex_record_id,
            run_id,
            embedding_strategy
        from metadata.current_embeddings
    """).to_df()

    expected_total_rows = 10
    expected_run_1_rows = 5
    expected_run_2_rows = 5

    assert len(current_embeddings_df) == expected_total_rows
    assert (
        len(
            current_embeddings_df[
                current_embeddings_df.run_id == "emb-current-latest-run-1"
            ]
        )
        == expected_run_1_rows
    )
    assert (
        len(
            current_embeddings_df[
                current_embeddings_df.run_id == "emb-current-latest-run-2"
            ]
        )
        == expected_run_2_rows
    )


def test_tdm_current_run_embeddings_view_structure(tmp_path):
    td = TIMDEXDataset(str(tmp_path / "current_run_embeddings_structure"))

    td.records.write(
        generate_sample_records(
            num_records=25,
            source="alma",
            run_date="2025-03-01",
            run_type="full",
            run_id="emb-current-run-structure-run",
        ),
        write_append_deltas=False,
    )

    td.metadata.rebuild_dataset_metadata()

    td.embeddings.write(
        generate_sample_embeddings_for_run(td, run_id="emb-current-run-structure-run"),
        write_append_deltas=False,
    )

    td.metadata.rebuild_dataset_metadata()

    current_run_embeddings_df = td.conn.query(
        """select * from metadata.current_run_embeddings limit 1;"""
    ).to_df()

    assert len(current_run_embeddings_df) == 1
    # pre-joined view includes native embeddings columns + records columns
    expected_columns = set(TIMDEXEmbeddings.METADATA_CONFIG.metadata_columns) | {
        "source",
        "run_date",
        "run_type",
        "action",
        "run_timestamp",
    }
    assert set(current_run_embeddings_df.columns) == expected_columns


def test_tdm_prejoined_embeddings_view_has_correct_source_values(tmp_path):
    """Verify pre-joined source column matches the underlying records."""
    td = TIMDEXDataset(str(tmp_path / "prejoin_source_values"))

    td.records.write(
        generate_sample_records(
            num_records=10,
            source="alma",
            run_date="2025-03-01",
            run_type="full",
            run_id="prejoin-run-1",
        ),
        write_append_deltas=False,
    )
    td.records.write(
        generate_sample_records(
            num_records=10,
            source="dspace",
            run_date="2025-03-02",
            run_type="full",
            run_id="prejoin-run-2",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()

    td.embeddings.write(
        generate_sample_embeddings_for_run(td, run_id="prejoin-run-1"),
        write_append_deltas=False,
    )
    td.embeddings.write(
        generate_sample_embeddings_for_run(td, run_id="prejoin-run-2"),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()

    # all embeddings from run-1 should have source='alma'
    alma_embeddings = td.conn.query("""
        select count(*) from metadata.embeddings
        where run_id = 'prejoin-run-1' and source = 'alma'
    """).fetchone()[0]
    assert alma_embeddings == 10  # noqa: PLR2004

    # all embeddings from run-2 should have source='dspace'
    dspace_embeddings = td.conn.query("""
        select count(*) from metadata.embeddings
        where run_id = 'prejoin-run-2' and source = 'dspace'
    """).fetchone()[0]
    assert dspace_embeddings == 10  # noqa: PLR2004

    # verify current_embeddings also has pre-joined source column
    alma_current = td.conn.query("""
        select count(*) from metadata.current_embeddings
        where source = 'alma'
    """).fetchone()[0]
    dspace_current = td.conn.query("""
        select count(*) from metadata.current_embeddings
        where source = 'dspace'
    """).fetchone()[0]
    assert alma_current == 10  # noqa: PLR2004
    assert dspace_current == 10  # noqa: PLR2004


def test_tdm_prejoined_embeddings_filterable_by_run_date(tmp_path):
    """Verify pre-joined run_date column is usable for filtering."""
    td = TIMDEXDataset(str(tmp_path / "prejoin_run_date_filter"))

    td.records.write(
        generate_sample_records(
            num_records=10,
            source="alma",
            run_date="2025-03-01",
            run_type="full",
            run_id="filter-run-1",
        ),
        write_append_deltas=False,
    )
    td.records.write(
        generate_sample_records(
            num_records=10,
            source="alma",
            run_date="2025-04-01",
            run_type="full",
            run_id="filter-run-2",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()

    td.embeddings.write(
        generate_sample_embeddings_for_run(td, run_id="filter-run-1"),
        write_append_deltas=False,
    )
    td.embeddings.write(
        generate_sample_embeddings_for_run(td, run_id="filter-run-2"),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()

    # filter embeddings by run_date
    march_embeddings = td.conn.query("""
        select count(*) from metadata.embeddings
        where run_date = cast('2025-03-01' as date)
    """).fetchone()[0]
    assert march_embeddings == 10  # noqa: PLR2004

    april_embeddings = td.conn.query("""
        select count(*) from metadata.embeddings
        where run_date = cast('2025-04-01' as date)
    """).fetchone()[0]
    assert april_embeddings == 10  # noqa: PLR2004


def test_tdm_keyset_paginated_query_on_prejoined_embeddings_view(tmp_path):
    """Verify build_keyset_paginated_metadata_query works on pre-joined embeddings."""
    td = TIMDEXDataset(str(tmp_path / "keyset_prejoin_embeddings"))

    td.records.write(
        generate_sample_records(
            num_records=25,
            source="alma",
            run_date="2025-03-01",
            run_type="full",
            run_id="keyset-prejoin-run",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()

    td.embeddings.write(
        generate_sample_embeddings_for_run(td, run_id="keyset-prejoin-run"),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()
    td.reflect_sa_tables()

    # build a keyset pagination query against the pre-joined embeddings view
    query = td.metadata.build_keyset_paginated_metadata_query(
        "embeddings",
        limit=10,
        keyset_value=(0, 0, 0),
    )

    # execute and verify results
    result_df = td.conn.query(query).to_df()
    assert len(result_df) == 10  # noqa: PLR2004
    expected_cols = set(
        TIMDEXDatasetMetadata.BASE_METADATA_COLUMNS
        + TIMDEXEmbeddings.ADDITIONAL_METADATA_COLUMNS
        + ["run_id_hash", "filename_hash"]
    )
    assert set(result_df.columns) == expected_cols


def test_tdm_embeddings_write_append_deltas_without_static_embeddings_table(tmp_path):
    record_count = 20
    td = TIMDEXDataset(str(tmp_path / "embeddings_append_deltas_only"))

    # build records metadata only
    td.records.write(
        generate_sample_records(
            num_records=record_count,
            source="alma",
            run_date="2025-03-02",
            run_type="full",
            run_id="emb-delta-run",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()

    # write embeddings with append deltas (without rebuilding static metadata first)
    td.embeddings.write(generate_sample_embeddings_for_run(td, run_id="emb-delta-run"))

    # embeddings metadata views should still exist and include append deltas
    embeddings_count = td.conn.query(
        """select count(*) from metadata.embeddings;"""
    ).fetchone()[0]
    embeddings_deltas_count = td.conn.query(
        """select count(*) from metadata.embeddings_append_deltas;"""
    ).fetchone()[0]

    embeddings_deltas_path = td.metadata.append_deltas_path_for(
        TIMDEXEmbeddings.METADATA_CONFIG
    )
    assert embeddings_count == record_count
    assert embeddings_deltas_count == record_count
    assert os.listdir(embeddings_deltas_path)


def test_tdm_merge_append_deltas_merges_embeddings(tmp_path):
    run_1_count = 30
    run_2_count = 10
    td = TIMDEXDataset(str(tmp_path / "embeddings_merge"))

    # write records + initial embeddings and rebuild so static_db.embeddings exists
    td.records.write(
        generate_sample_records(
            num_records=run_1_count,
            source="alma",
            run_date="2025-03-03",
            run_type="full",
            run_id="emb-merge-run-1",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()

    td.embeddings.write(
        generate_sample_embeddings_for_run(td, run_id="emb-merge-run-1"),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()

    # write second embeddings run with append deltas
    td.records.write(
        generate_sample_records(
            num_records=run_2_count,
            source="alma",
            run_date="2025-03-04",
            run_type="daily",
            run_id="emb-merge-run-2",
        ),
        write_append_deltas=False,
    )
    td.metadata.rebuild_dataset_metadata()

    td.embeddings.write(generate_sample_embeddings_for_run(td, run_id="emb-merge-run-2"))

    embeddings_count_before_merge = td.conn.query(
        """select count(*) from metadata.embeddings;"""
    ).fetchone()[0]
    assert (
        td.conn.query(
            """select count(*) from metadata.embeddings_append_deltas;"""
        ).fetchone()[0]
        == run_2_count
    )

    td.metadata.merge_append_deltas()
    td.refresh()

    embeddings_static_after_merge = td.conn.query(
        """select count(*) from static_db.embeddings;"""
    ).fetchone()[0]
    embeddings_deltas_after_merge = td.conn.query(
        """select count(*) from metadata.embeddings_append_deltas;"""
    ).fetchone()[0]

    assert embeddings_static_after_merge == embeddings_count_before_merge
    assert embeddings_deltas_after_merge == 0


def test_td_prepare_duckdb_secret_and_extensions_home_env_var_set_and_valid(
    monkeypatch, tmp_path_factory, timdex_dataset_with_runs
):
    preset_home = tmp_path_factory.mktemp("my-account")
    monkeypatch.setenv("HOME", str(preset_home))

    td = TIMDEXDataset(timdex_dataset_with_runs.location)
    df = (
        td.conn.query("""
        select
            current_setting('secret_directory') as secret_directory,
            current_setting('extension_directory') as extension_directory
        ;
        """)
        .to_df()
        .iloc[0]
    )
    assert "my-account" in df.secret_directory
    assert df.extension_directory == ""  # expected and okay when HOME set


def test_td_prepare_duckdb_secret_and_extensions_home_env_var_unset(
    monkeypatch, timdex_dataset_with_runs
):
    monkeypatch.delenv("HOME", raising=False)

    td = TIMDEXDataset(timdex_dataset_with_runs.location)

    df = (
        td.conn.query("""
        select
            current_setting('secret_directory') as secret_directory,
            current_setting('extension_directory') as extension_directory
        ;
        """)
        .to_df()
        .iloc[0]
    )
    assert df.secret_directory == "/tmp/.duckdb/secrets"
    assert df.extension_directory == "/tmp/.duckdb/extensions"


def test_td_prepare_duckdb_secret_and_extensions_home_env_var_set_but_empty(
    monkeypatch, timdex_dataset_with_runs
):
    monkeypatch.setenv("HOME", "")  # simulate AWS Lambda environment

    td = TIMDEXDataset(timdex_dataset_with_runs.location)

    df = (
        td.conn.query("""
        select
            current_setting('secret_directory') as secret_directory,
            current_setting('extension_directory') as extension_directory
        ;
        """)
        .to_df()
        .iloc[0]
    )
    assert df.secret_directory == "/tmp/.duckdb/secrets"
    assert df.extension_directory == "/tmp/.duckdb/extensions"


def test_td_preload_current_records_default_false(tmp_path):
    td = TIMDEXDataset(str(tmp_path))
    assert td.preload_current_records is False
    assert td.preload_current_records is False


def test_td_preload_current_records_flag_true(tmp_path):
    td = TIMDEXDataset(str(tmp_path), preload_current_records=True)
    assert td.preload_current_records is True
    assert td.preload_current_records is True


def test_tdm_preload_false_no_temp_table(timdex_dataset_with_runs):
    # instantiate TIMDEXDataset without preloading current records (default)
    td = TIMDEXDataset(timdex_dataset_with_runs.location)

    # assert that materialized, temporary table "temp.current_records" does not exist
    temp_table_count = td.conn.query("""
        select count(*)
        from information_schema.tables
        where table_catalog = 'temp'
        and table_name = 'current_records'
        and table_type = 'LOCAL TEMPORARY'
        ;
        """).fetchone()[0]

    assert temp_table_count == 0


def test_tdm_preload_true_has_temp_table(timdex_dataset_with_runs):
    # instantiate TIMDEXDataset with preloading current records
    td = TIMDEXDataset(timdex_dataset_with_runs.location, preload_current_records=True)

    # assert that materialized, temporary table "temp.current_records" does exist
    temp_table_count = td.conn.query("""
            select count(*)
            from information_schema.tables
            where table_catalog = 'temp'
            and table_name = 'current_records'
            and table_type = 'LOCAL TEMPORARY'
            ;
            """).fetchone()[0]

    assert temp_table_count == 1
