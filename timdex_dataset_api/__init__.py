"""timdex_dataset_api/__init__.py"""

from timdex_dataset_api.dataset import TIMDEXDataset
from timdex_dataset_api.embeddings import DatasetEmbedding, TIMDEXEmbeddings
from timdex_dataset_api.metadata import TIMDEXDatasetMetadata
from timdex_dataset_api.record import DatasetRecord

__version__ = "3.6.1"

__all__ = [
    "DatasetEmbedding",
    "DatasetRecord",
    "TIMDEXDataset",
    "TIMDEXDatasetMetadata",
    "TIMDEXEmbeddings",
]
