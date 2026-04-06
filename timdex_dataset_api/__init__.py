"""timdex_dataset_api/__init__.py"""

from importlib.metadata import version

from timdex_dataset_api.data_source import TIMDEXDataSource, ValidTable
from timdex_dataset_api.dataset import TIMDEXDataset
from timdex_dataset_api.embeddings import DatasetEmbedding, TIMDEXEmbeddings
from timdex_dataset_api.metadata import (
    CurrentMetadataViewSpec,
    DataTypeMetadataConfig,
    TIMDEXDatasetMetadata,
)
from timdex_dataset_api.records import DatasetRecord, TIMDEXRecords

__version__ = version("timdex_dataset_api")

__all__ = [
    "CurrentMetadataViewSpec",
    "DataTypeMetadataConfig",
    "DatasetEmbedding",
    "DatasetRecord",
    "TIMDEXDataSource",
    "TIMDEXDataset",
    "TIMDEXDatasetMetadata",
    "TIMDEXEmbeddings",
    "TIMDEXRecords",
    "ValidTable",
]
