# ruff: noqa: PLR2004

import pyarrow as pa
import pytest

from timdex_dataset_api.utils import (
    get_batch_fragment_index,
    get_first_column_value_from_batch,
)


def test_get_first_column_value_from_batch():
    data = pa.array(["value1", "value2", "value3"])
    batch = pa.RecordBatch.from_arrays([data], ["test_column"])

    result = get_first_column_value_from_batch(batch, "test_column")
    assert result == "value1"


def test_get_first_column_value_from_batch_empty_raises_error():
    data = pa.array([])
    batch = pa.RecordBatch.from_arrays([data], ["test_column"])

    with pytest.raises(
        ValueError, match="No values found for column 'test_column' in batch."
    ):
        get_first_column_value_from_batch(batch, "test_column")


def test_get_batch_fragment_index():
    fragment_index = pa.array([42])
    batch = pa.RecordBatch.from_arrays([fragment_index], ["__fragment_index"])

    result = get_batch_fragment_index(batch)
    assert result == 42
