import itertools
import logging
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import attrs
import pyarrow as pa
import pyarrow.dataset as ds
from attrs import asdict, define, field

from timdex_dataset_api import TIMDEXDataset
from timdex_dataset_api.record import datetime_iso_parse

logger = logging.getLogger(__name__)

TIMDEX_DATASET_EMBEDDINGS_SCHEMA = pa.schema(
    (
        pa.field("timdex_record_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("run_record_offset", pa.int32()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("embedding_model", pa.string()),
        pa.field("embedding_strategy", pa.string()),
        pa.field("embedding_vector", pa.list_(pa.float32())),
        pa.field("embedding_object", pa.binary()),
        pa.field("year", pa.string()),
        pa.field("month", pa.string()),
        pa.field("day", pa.string()),
    )
)


@define
class DatasetEmbedding:
    """Container for single record embedding.

    Fields:
        timdex_record_id: Fields (timdex_record_id, run_id, run_record_offset) combine to
            form a composite key that points to a single, distinct record version in the
            records data.
        run_id: ...
        run_record_offset: ...
        embedding_model: Embedding model name, e.g. HuggingFace URI
        embedding_strategy: Strategy used to create embedding
            - this correlates to a transformation strategy in the timdex-embeddings CLI
            application, e.g. "full_record"
        timestamp: Timestamp when embedding was created
        embedding_vector: Numerical vector representation of embedding
            - preferred form for storing embedding as a numerical array
        embedding_object: Object representation of the embedding
            - example: {token:weight, ...} representation for sparse vector
            - flexible enough to hold other representations
    """

    timdex_record_id: str = field()
    run_id: str = field()
    run_record_offset: int = field()
    embedding_model: str = field()
    embedding_strategy: str = field()
    timestamp: datetime = field(  # type: ignore[assignment]
        converter=datetime_iso_parse,
        default=attrs.Factory(lambda: datetime.now(tz=UTC).isoformat()),
    )
    embedding_vector: list[float] | None = field(default=None)
    embedding_object: bytes | None = field(default=None)

    @property
    def year(self) -> str:
        return self.timestamp.strftime("%Y")

    @property
    def month(self) -> str:
        return self.timestamp.strftime("%m")

    @property
    def day(self) -> str:
        return self.timestamp.strftime("%d")

    def to_dict(
        self,
    ) -> dict:
        """Serialize instance as dictionary."""
        return {
            **asdict(self),
            "year": self.year,
            "month": self.month,
            "day": self.day,
        }


class TIMDEXEmbeddings:

    def __init__(self, timdex_dataset: TIMDEXDataset):
        """Init TIMDEXEmbeddings.

        Class to handle the writing and readings of embeddings associated with TIMDEX
        records.

        Args:
            - timdex_dataset: instance of TIMDEXDataset
        """
        self.timdex_dataset = timdex_dataset

        self.schema = TIMDEX_DATASET_EMBEDDINGS_SCHEMA
        self.partition_columns = ["year", "month", "day"]

    @property
    def data_embeddings_root(self) -> str:
        return f"{self.timdex_dataset.location.removesuffix('/')}/data/embeddings"

    def write(
        self,
        embeddings_iter: Iterator[DatasetEmbedding],
        *,
        use_threads: bool = True,
    ) -> list[ds.WrittenFile]:
        """Write embeddings as parquet files to /data/embeddings.

        Approach is similar to TIMDEXDataset.write() for Records:
            - use self.data_embeddings_root for location of embeddings parquet files
            - use pyarrow Dataset to write rows
        """
        start_time = time.perf_counter()
        written_files: list[ds.WrittenFile] = []

        filesystem, path = self.timdex_dataset.parse_location(self.data_embeddings_root)

        embedding_batches_iter = self.create_embedding_batches(embeddings_iter)
        ds.write_dataset(
            embedding_batches_iter,
            base_dir=path,
            basename_template="%s-{i}.parquet" % (str(uuid.uuid4())),  # noqa: UP031
            existing_data_behavior="overwrite_or_ignore",
            filesystem=filesystem,
            file_visitor=lambda written_file: written_files.append(written_file),  # type: ignore[arg-type]
            format="parquet",
            max_open_files=500,
            max_rows_per_file=self.timdex_dataset.config.max_rows_per_file,
            max_rows_per_group=self.timdex_dataset.config.max_rows_per_group,
            partitioning=self.partition_columns,
            partitioning_flavor="hive",
            schema=self.schema,
            use_threads=use_threads,
        )

        self.log_write_statistics(start_time, written_files)

        return written_files

    def create_embedding_batches(
        self, embeddings_iter: Iterator["DatasetEmbedding"]
    ) -> Iterator[pa.RecordBatch]:
        for i, embedding_batch in enumerate(
            itertools.batched(
                embeddings_iter, self.timdex_dataset.config.write_batch_size
            )
        ):
            embedding_dicts = [embedding.to_dict() for embedding in embedding_batch]
            batch = pa.RecordBatch.from_pylist(embedding_dicts)
            logger.debug(f"Yielding batch {i + 1} for dataset writing.")
            yield batch

    def log_write_statistics(
        self,
        start_time: float,
        written_files: list[ds.WrittenFile],
    ) -> None:
        """Parse written files from write and log statistics."""
        total_time = round(time.perf_counter() - start_time, 2)
        total_files = len(written_files)
        total_rows = sum(
            [wf.metadata.num_rows for wf in written_files]  # type: ignore[attr-defined]
        )
        total_size = sum([wf.size for wf in written_files])  # type: ignore[attr-defined]
        logger.info(
            f"Dataset write complete - elapsed: "
            f"{total_time}s, "
            f"total files: {total_files}, "
            f"total rows: {total_rows}, "
            f"total size: {total_size}"
        )
