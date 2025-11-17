# ruff: noqa: PLR2004
import json
import math
import os
from datetime import UTC, datetime

import pyarrow.dataset as ds

from timdex_dataset_api.embeddings import (
    TIMDEX_DATASET_EMBEDDINGS_SCHEMA,
    DatasetEmbedding,
    TIMDEXEmbeddings,
)


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
