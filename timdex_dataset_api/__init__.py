"""timdex_dataset_api/__init__.py"""

from importlib.metadata import version

from timdex_dataset_api.dataset import TIMDEXDataset
from timdex_dataset_api.embeddings import DatasetEmbedding, TIMDEXEmbeddings
from timdex_dataset_api.metadata import TIMDEXDatasetMetadata
from timdex_dataset_api.record import DatasetRecord

__version__ = version("timdex_dataset_api")

__all__ = [
    "DatasetEmbedding",
    "DatasetRecord",
    "TIMDEXDataset",
    "TIMDEXDatasetMetadata",
    "TIMDEXEmbeddings",
]
