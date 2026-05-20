"""timdex_dataset_api/__init__.py"""

from importlib.metadata import version

from timdex_dataset_api.data_types import (
    DatasetEmbedding,
    DatasetFulltext,
    DatasetRecord,
    TIMDEXEmbeddings,
    TIMDEXFulltexts,
    TIMDEXRecords,
)
from timdex_dataset_api.dataset import TIMDEXDataset
from timdex_dataset_api.metadata import TIMDEXDatasetMetadata

__version__ = version("timdex_dataset_api")

# NOTE: proposed that we remove all imports from 'data_types' on next major release
#   to avoid over-populating the root namespace.   Leaving now for backwards
#   compatibility.
__all__ = [
    "DatasetEmbedding",
    "DatasetFulltext",
    "DatasetRecord",
    "TIMDEXDataset",
    "TIMDEXDatasetMetadata",
    "TIMDEXEmbeddings",
    "TIMDEXFulltexts",
    "TIMDEXRecords",
]
