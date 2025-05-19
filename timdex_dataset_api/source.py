"""timdex_dataset_api/source.py"""

import time
from collections.abc import Iterator
from typing import Unpack

import pyarrow as pa

from timdex_dataset_api.config import configure_logger
from timdex_dataset_api.dataset import DatasetFilters, TIMDEXDataset, TIMDEXDatasetConfig
from timdex_dataset_api.exceptions import DatasetNotLoadedError
from timdex_dataset_api.run import TIMDEXRunManager

logger = configure_logger(__name__)


class TIMDEXSource(TIMDEXDataset):
    """Class to represent the current state of a single source in the TIMDEX dataset.

    This class extends TIMDEXDataset with some key opinionations:
        - the load() method requires a source name and only parquet files associated
        with current records in Opensearch are included
        - all read operations are deduped so only the first time a record is encountered
        is it yielded; ensuring all records yielded are current in Opensearch (Note: this
        can include action "index" or "delete")
    """

    def __init__(
        self,
        location: str,
        source: str,
        config: TIMDEXDatasetConfig | None = None,
    ):
        super().__init__(location=location, config=config)
        self.source = source
        self._seen_records: set = set()

    def load(
        self,
        *,
        reverse_chronological: bool = True,
        **filters: Unpack[DatasetFilters],
    ) -> None:
        """Extends TIMDEXDataset.load to limit to current, ordered runs for a source.

        This extended load() method loads the parquet dataset but limits to only
        parquet files associated with current runs for a source (i.e. all runs since the
        last "full" run).  Because these parquet files are also reverse chronologically
        ordered, yielding in order produces the "current" version of each record.

        Args:
            - reverse_chronological: bool | default True, boolean to keep the parquet
            files returned from TIMDEXRunManager reverse chronologically ordered (most
            recent first)
        """
        start_time = time.perf_counter()

        # perform normal load which discovers all parquet files in dataset
        self._reload_dataset()

        # get list of ordered parquet files for current source runs
        timdex_run_manager = TIMDEXRunManager(timdex_dataset=self)
        current_ordered_parquet_files = (
            timdex_run_manager.get_current_source_parquet_files(self.source)
        )

        if not reverse_chronological:
            current_ordered_parquet_files = list(reversed(current_ordered_parquet_files))

        # update dataset paths and reload the dataset
        self.paths = current_ordered_parquet_files
        self._reload_dataset()

        if list(self.paths) != list(self.dataset.files):  # type: ignore[attr-defined]
            raise ValueError(
                "Mismatch between current parquet files found for source "
                "and parquet files loaded for dataset."
            )

        self.dataset = self._get_filtered_dataset(**filters)

        logger.info(
            f"Dataset limited to current records for source: '{self.source}', "
            f"parquet files count: {len(self.paths)}, "
            f"elapsed: {round(time.perf_counter() - start_time, 2)}s",
        )

    def read_batches_iter(
        self,
        columns: list[str] | None = None,
        **filters: Unpack[DatasetFilters],
    ) -> Iterator[pa.RecordBatch]:
        """Extends TIMDEXDataset.read_batches_iter to dedupe records during yielding."""
        if not self.dataset:
            raise DatasetNotLoadedError(
                "Dataset is not loaded. Please call the `load` method first."
            )
        dataset = self._get_filtered_dataset(**filters)

        # init set for processed records
        self._seen_records.clear()

        for batch in dataset.to_batches(
            columns=columns,
            batch_size=self.config.read_batch_size,
            batch_readahead=self.config.batch_read_ahead,
            fragment_readahead=self.config.fragment_read_ahead,
        ):
            if len(batch) > 0:
                # get list of timdex ids from batch
                timdex_ids = batch.column("timdex_record_id").to_pylist()

                # init list of batch indices for records unseen
                unseen_batch_indices = []

                # check each record id and track unseen ones
                for i, record_id in enumerate(timdex_ids):
                    if record_id not in self._seen_records:
                        unseen_batch_indices.append(i)
                        self._seen_records.add(record_id)

                # if all records from batch were seen, return empty batch with same schema
                if not unseen_batch_indices:
                    continue

                # yield unseen records from batch
                deduped_batch = batch.take(pa.array(unseen_batch_indices))  # type: ignore[arg-type]
                if len(deduped_batch) > 0:
                    yield deduped_batch
