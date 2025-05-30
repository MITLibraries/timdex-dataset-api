"""timdex_dataset_api/__init__.py"""

from timdex_dataset_api.dataset import TIMDEXDataset
from timdex_dataset_api.record import DatasetRecord

__version__ = "2.0.0"

__all__ = [
    "DatasetRecord",
    "TIMDEXDataset",
]
