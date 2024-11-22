"""timdex_dataset_api/__init__.py"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("timdex_dataset_api")
except PackageNotFoundError:
    __version__ = "unknown"
