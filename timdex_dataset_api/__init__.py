"""timdex_dataset_api/__init__.py"""

from importlib.metadata import version

from timdex_dataset_api.dataset import TIMDEXDataset
from timdex_dataset_api.metadata import TIMDEXDatasetMetadata

__version__ = version("timdex_dataset_api")

__all__ = [
    "TIMDEXDataset",
    "TIMDEXDatasetMetadata",
]
