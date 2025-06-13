"""timdex_dataset_api/utils.py"""

from typing import Any

import pyarrow as pa


def get_first_column_value_from_batch(
    batch: pa.RecordBatch,
    column: str,
) -> Any:  # noqa: ANN401
    """Given a pyarrow RecordBatch, return the value of the first row from a column."""
    values = batch.column(column).to_pylist()
    if not values:
        raise ValueError(f"No values found for column '{column}' in batch.")
    return values[0]


def get_batch_fragment_index(batch: pa.RecordBatch) -> int:
    """Return the fragment index from a RecordBatch, by using the first row value."""
    return get_first_column_value_from_batch(batch, "__fragment_index")
