# ruff: noqa: PLR2004
import json
import math
import os
from datetime import UTC, datetime

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from timdex_dataset_api.embeddings import (
    TIMDEX_DATASET_EMBEDDINGS_SCHEMA,
    DatasetEmbedding,
    TIMDEXEmbeddings,
)

EMBEDDINGS_COLUMNS_SET = set(TIMDEX_DATASET_EMBEDDINGS_SCHEMA.names)


def test_dataset_embedding_init():
    values = {
        "timdex_record_id": "alma:123",
        "run_id": "test-run-1",
        "run_record_offset": 0,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_strategy": "full_record",
        "timestamp": "2024-12-01T10:00:00+00:00",
        "embedding_vector": [0.1, 0.2, 0.3],
        "embedding_object": json.dumps(
            {"token1": 0.1, "token2": 0.2, "token3": 0.3}
        ).encode(),
    }
    embedding = DatasetEmbedding(**values)

    assert embedding
    assert embedding.timdex_record_id == "alma:123"
    assert embedding.timestamp == datetime(2024, 12, 1, 10, 0, tzinfo=UTC)
    assert embedding.embedding_object == b'{"token1": 0.1, "token2": 0.2, "token3": 0.3}'


def test_dataset_embedding_date_properties():
    embedding = DatasetEmbedding(
        timdex_record_id="alma:123",
        run_id="test-run-1",
        run_record_offset=0,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_strategy="full_record",
        timestamp="2024-12-01T10:00:00+00:00",
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
        "timestamp": "2024-12-01T10:00:00+00:00",
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
    timdex_dataset_empty, sample_embeddings_generator
):
    timdex_embeddings = TIMDEXEmbeddings(timdex_dataset_empty)
    timdex_embeddings.write(sample_embeddings_generator(100))
    timdex_embeddings = TIMDEXEmbeddings(timdex_dataset_empty)

    batches = timdex_embeddings.read_batches_iter()
    batch = next(batches)
    assert isinstance(batch, pa.RecordBatch)


def test_embeddings_read_batches_all_columns_by_default(timdex_embeddings_with_runs):
    batches = timdex_embeddings_with_runs.read_batches_iter()
    batch = next(batches)
    assert set(batch.column_names) == EMBEDDINGS_COLUMNS_SET


def test_embeddings_read_batches_filter_columns(timdex_embeddings_with_runs):
    columns_subset = ["timdex_record_id", "run_id", "embedding_strategy"]
    batches = timdex_embeddings_with_runs.read_batches_iter(columns=columns_subset)
    batch = next(batches)
    assert set(batch.column_names) == set(columns_subset)


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
    assert set(record.keys()) == EMBEDDINGS_COLUMNS_SET
