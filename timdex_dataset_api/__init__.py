"""timdex_dataset_api/__init__.py"""

from importlib.metadata import version

from timdex_dataset_api.data_type import DataTypeTableConfig, TIMDEXDataType
from timdex_dataset_api.dataset import TIMDEXDataset
from timdex_dataset_api.embeddings import DatasetEmbedding, TIMDEXEmbeddings
from timdex_dataset_api.metadata import TIMDEXDatasetMetadata
from timdex_dataset_api.records import DatasetRecord, TIMDEXRecords

__version__ = version("timdex_dataset_api")

__all__ = [
    "DataTypeTableConfig",
    "DatasetEmbedding",
    "DatasetRecord",
    "TIMDEXDataType",
    "TIMDEXDataset",
    "TIMDEXDatasetMetadata",
    "TIMDEXEmbeddings",
    "TIMDEXRecords",
]
