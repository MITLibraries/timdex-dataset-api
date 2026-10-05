# ruff: noqa: D212

"""
Date: 2026-10-02

Description:

This "migration" is a full overlay (replacement) of the TIMDEX dataset from production
to stage.  Migration is scare quoted here because it's not a typical adjustment of data
within a single environment, but a complete refresh.

Unlike other migrations in this folder, this migration can be run multiple times, each
time fully resetting stage data with prod data.  We may eventually explore how to make
this a first-class operation in the TIMDEX dataset environments. Until then, we can use
this "migration" to tease out the moving parts and perform the sync on an ad-hoc basis.

This utilizes a combination of readwrite + readonly permissions to protect the source
data.

Overall process (steps, by number or name):
1. delete: in stage, fully delete `s3://<timdex-stage>/dataset` folder
2. sync: copy `s3://<timdex-prod>/dataset` --> `s3://<timdex-stage>/dataset`
3. metadata: use TDA to rebuild metadata in stage
4. drop: invoke TIM as ECS task in stage to drop current OpenSearch indexes
5. reindex: invoke TIM as ECS task in stage to reindex all sources (without embeddings)
6. enrichments:
    - invoke TIM as ECS task in stage to update all docs with EMBEDDINGS, all sources
    - invoke TIM as ECS task in stage to update all docs with FULLTEXTS, all sources

Requirements:
- TimdexManager@stage AWS credentials set in terminal
- Accessible AWS profile for readonly production TIMDEX access called "timmy-prod"
- Env vars:
    - TIMDEX_PROD_BUCKET
    - TIMDEX_STAGE_BUCKET
    - TIM_ECS_SUBNETS: single string, comma delimited
    - TIM_ECS_SECURITY_GROUP
    - TIMDEX_STAGE_OPENSEARCH_ENDPOINT

Usage:

Steps run in order from --start (default: 2, sync) through the last step.

# run from sync onward, without deleting first (rclone sync still removes stage-only
# objects)
uv run python migrations/004_2026_10_02_sync_prod_to_stage.py

# first fully delete s3://<stage>/dataset/ (prompts to confirm), then run all steps
uv run python migrations/004_2026_10_02_sync_prod_to_stage.py --start 1

# resume from reindexing, by name or number
uv run python migrations/004_2026_10_02_sync_prod_to_stage.py --start reindex
uv run python migrations/004_2026_10_02_sync_prod_to_stage.py --start 5
"""

import argparse
import itertools
import logging
import os
import pathlib
import subprocess
import time
from typing import Any

import boto3
import duckdb

from timdex_dataset_api import TIMDEXDataset
from timdex_dataset_api.config import configure_dev_logger
from timdex_dataset_api.utils import S3Client

configure_dev_logger()

PROD_BUCKET = os.environ["TIMDEX_PROD_BUCKET"]
STAGE_BUCKET = os.environ["TIMDEX_STAGE_BUCKET"]
DATASET_PREFIX = "dataset/"
PROD_REMOTE = "prod"
STAGE_REMOTE = "stage"

TIM_ECS_CLUSTER = "timdex-stage"
TIM_ECS_TASK_DEFINITION = "timdex-tim-stage"
TIM_ECS_SUBNETS = [
    subnet.strip()
    for subnet in os.environ["TIM_ECS_SUBNETS"].split(",")
    if subnet.strip()
]
TIM_ECS_SECURITY_GROUP = os.environ["TIM_ECS_SECURITY_GROUP"]
TIM_ECS_CONTAINER_NAME = "timdex-tim"
TIM_ECS_ENVIRONMENT = {
    "WARNING_ONLY_LOGGERS": "boto3,botocore,opensearch,urllib3",
    "TIMDEX_OPENSEARCH_ENDPOINT": os.environ["TIMDEX_STAGE_OPENSEARCH_ENDPOINT"],
    "AUTH_SERVICE_TYPE": "aoss",
}
TIM_ECS_POLL_SECONDS = 10
OPENSEARCH_PRIMARY_ALIAS = "all-current"

# alias -> sources, copied from timdex-pipeline-lambdas Config.INDEX_ALIASES
# https://github.com/MITLibraries/timdex-pipeline-lambdas/blob/main/lambdas/config.py
OPENSEARCH_INDEX_ALIASES = {
    "geo": ["gismit", "gisogm"],
    "rdi": ["jpal", "whoas", "zenodo"],
    "timdex": [
        "alma",
        "aspace",
        "digitalcollections",
        "dspace",
        "libguides",
        "mitlibwebsite",
        "researchdatabases",
    ],
    "use": [
        "aspace",
        "digitalcollections",
        "dspace",
        "gismit",
        "gisogm",
        "libguides",
        "mitlibwebsite",
        "researchdatabases",
    ],
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if "prod" not in PROD_BUCKET:
    raise ValueError(f"Prod bucket doesn't look right: {PROD_BUCKET}")
if "stage" not in STAGE_BUCKET:
    raise ValueError(f"Stage bucket doesn't look right: {STAGE_BUCKET}")
if "stage" not in TIM_ECS_CLUSTER or "stage" not in TIM_ECS_TASK_DEFINITION:
    raise ValueError(
        "ECS cluster or task definition doesn't look right: "
        f"{TIM_ECS_CLUSTER}, {TIM_ECS_TASK_DEFINITION}"
    )


def _get_timdex_source_list(table: str = "current_records") -> list[str]:
    """Return sorted distinct sources from a stage dataset metadata table or view.

    Returns an empty list if the table does not exist, e.g. 'current_fulltexts' when
    the dataset has no fulltexts.
    """
    td = TIMDEXDataset(f"s3://{STAGE_BUCKET}/{DATASET_PREFIX}")
    try:
        sources = td.conn.query(f"select distinct source from metadata.{table};")
    except duckdb.CatalogException:
        logger.warning("Metadata table 'metadata.%s' not found", table)
        return []
    return sorted(sources.to_df()["source"])


def tim_ecs_runner(command: list[str]) -> list[str]:
    """Run TIM as an ECS task in stage and block until it stops.

    The task's CloudWatch logs are streamed to the local logger while it runs, and all
    log messages are returned so callers can parse TIM output.  Raises if the task
    fails to start or the TIM container exits non-zero.

    Args:
        command: TIM CLI arguments, e.g. ["delete", "--index=alma-...", "--force"]
    """
    ecs_client = boto3.client("ecs")
    logs_client = boto3.client("logs")

    response = ecs_client.run_task(
        cluster=TIM_ECS_CLUSTER,
        taskDefinition=TIM_ECS_TASK_DEFINITION,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "securityGroups": [TIM_ECS_SECURITY_GROUP],
                "subnets": TIM_ECS_SUBNETS,
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": TIM_ECS_CONTAINER_NAME,
                    "command": command,
                    "environment": [
                        {"name": name, "value": value}
                        for name, value in TIM_ECS_ENVIRONMENT.items()
                    ],
                }
            ]
        },
    )
    if response["failures"]:
        raise RuntimeError(f"Failed to start TIM ECS task: {response['failures']}")
    task_arn = response["tasks"][0]["taskArn"]
    task_id = task_arn.rsplit("/", maxsplit=1)[-1]
    region = ecs_client.meta.region_name
    logger.info(
        "Started TIM ECS task %s: tim %s, console: https://%s.console.aws.amazon.com"
        "/ecs/v2/clusters/%s/tasks/%s",
        task_id,
        " ".join(command),
        region,
        TIM_ECS_CLUSTER,
        task_id,
    )

    log_group, log_stream = _get_tim_task_log_location(ecs_client, task_id)
    messages: list[str] = []
    next_token = None
    last_status = None
    while True:
        task = ecs_client.describe_tasks(cluster=TIM_ECS_CLUSTER, tasks=[task_arn])[
            "tasks"
        ][0]
        if task["lastStatus"] != last_status:
            last_status = task["lastStatus"]
            logger.info("TIM ECS task %s status: %s", task_id, last_status)

        if last_status == "STOPPED":
            # allow trailing log events to land in CloudWatch before a final read
            time.sleep(TIM_ECS_POLL_SECONDS)

        new_messages, next_token = _read_log_messages(
            logs_client, log_group, log_stream, next_token
        )
        for message in new_messages:
            logger.info("[tim %s] %s", task_id, message)
        messages.extend(new_messages)

        if last_status == "STOPPED":
            break
        time.sleep(TIM_ECS_POLL_SECONDS)

    container = next(c for c in task["containers"] if c["name"] == TIM_ECS_CONTAINER_NAME)
    exit_code = container.get("exitCode")
    if exit_code != 0:
        raise RuntimeError(
            f"TIM ECS task {task_id} failed, exit code: {exit_code}, "
            f"reason: {task.get('stoppedReason')}"
        )
    logger.info("TIM ECS task %s complete", task_id)
    return messages


def _get_tim_task_log_location(ecs_client: Any, task_id: str) -> tuple[str, str]:  # noqa: ANN401
    """Return the CloudWatch (log group, log stream) for a TIM ECS task."""
    task_definition = ecs_client.describe_task_definition(
        taskDefinition=TIM_ECS_TASK_DEFINITION
    )["taskDefinition"]
    container = next(
        c
        for c in task_definition["containerDefinitions"]
        if c["name"] == TIM_ECS_CONTAINER_NAME
    )
    options = container["logConfiguration"]["options"]
    log_stream = f"{options['awslogs-stream-prefix']}/{TIM_ECS_CONTAINER_NAME}/{task_id}"
    return options["awslogs-group"], log_stream


def _read_log_messages(
    logs_client: Any,  # noqa: ANN401
    log_group: str,
    log_stream: str,
    next_token: str | None,
) -> tuple[list[str], str | None]:
    """Read all log messages available after next_token, returning the new token."""
    messages = []
    while True:
        kwargs = {
            "logGroupName": log_group,
            "logStreamName": log_stream,
            "startFromHead": True,
        }
        if next_token:
            kwargs["nextToken"] = next_token
        try:
            response = logs_client.get_log_events(**kwargs)
        except logs_client.exceptions.ResourceNotFoundException:
            # log stream is not created until the container starts
            return messages, next_token
        messages.extend(event["message"] for event in response["events"])
        if response["nextForwardToken"] == next_token:
            return messages, next_token
        next_token = response["nextForwardToken"]


def _get_opensearch_aliases() -> dict[str, list[str]]:
    """Return {alias: [indexes]} for all stage OpenSearch aliases, via `tim aliases`.

    Parses the 'Alias: <alias>' / 'Indexes: <index>, <index>' line pairs that TIM
    prints.  Returns an empty dict if there are no aliases.
    """
    messages = [message.strip() for message in tim_ecs_runner(["aliases"])]
    messages = [message for message in messages if message]
    aliases = {}
    for message, next_message in itertools.pairwise(messages):
        if message.startswith("Alias: ") and next_message.startswith("Indexes: "):
            alias = message.removeprefix("Alias: ")
            aliases[alias] = next_message.removeprefix("Indexes: ").split(", ")
    return aliases


def _get_indexes_by_source(indexes: list[str]) -> dict[str, list[str]]:
    """Group index names, e.g. 'alma-2026-01-01t00-00-00', by source."""
    indexes_by_source: dict[str, list[str]] = {}
    for index in indexes:
        indexes_by_source.setdefault(index.split("-", maxsplit=1)[0], []).append(index)
    return indexes_by_source


def delete_stage_dataset_data() -> None:
    """Delete all object keys under DATASET_PREFIX in the stage bucket."""
    s3_client = S3Client()
    stage_dataset_uri = f"s3://{STAGE_BUCKET}/{DATASET_PREFIX}"

    deleted = s3_client.delete_folder(stage_dataset_uri)

    leftover = s3_client.list_objects(stage_dataset_uri)
    if leftover:
        raise RuntimeError(
            "Stage dataset deletion incomplete, "
            f"{len(leftover)} object(s) remain at s3://{STAGE_BUCKET}/{DATASET_PREFIX}"
        )
    logger.info(
        "Deleted %d objects from s3://%s/%s", len(deleted), STAGE_BUCKET, DATASET_PREFIX
    )


def rclone_sync_prod_data_to_stage() -> None:
    """Mirror s3://<prod>/dataset/ to s3://<stage>/dataset/ via rclone.

    Uses migrations/assets/004_rclone.conf, in which the 'stage' remote resolves
    ambient credentials and the 'prod' remote assumes the readonly timmy prod
    role.  Streaming relay: objects flow prod -> this machine -> stage.
    """
    conf = pathlib.Path(__file__).parent / "assets" / "004_rclone.conf"
    command = [
        "rclone",
        f"--config={conf}",
        "sync",
        "--progress",
        "--transfers",
        "16",
        "--checkers",
        "32",
        f"{PROD_REMOTE}:{PROD_BUCKET}/{DATASET_PREFIX}",
        f"{STAGE_REMOTE}:{STAGE_BUCKET}/{DATASET_PREFIX}",
    ]
    subprocess.run(command, check=True)  # noqa: S603


def rebuild_dataset_metadata() -> None:
    """Rebuild stage dataset metadata from the freshly synced parquet files."""
    td = TIMDEXDataset(f"s3://{STAGE_BUCKET}/{DATASET_PREFIX}")
    td.metadata.rebuild_dataset_metadata()


def drop_opensearch_indexes() -> None:
    """Invoke TIM as ECS task to drop current indexes for all sources.

    Each source will have a current index in Opensearch.  This is known by looking to see
    what index is aliased to alias `all-current`.  It should also be the most recent date.

    By dropping the current index for each source, we effectively remove them from all
    aliases (e.g. all-current, timdex, use, etc.).

    Monitor until all ECS tasks are complete.
    """
    sources = _get_timdex_source_list()
    aliases = _get_opensearch_aliases()
    current_indexes = _get_indexes_by_source(aliases.get(OPENSEARCH_PRIMARY_ALIAS, []))
    logger.info(
        "Current indexes in alias '%s': %s", OPENSEARCH_PRIMARY_ALIAS, current_indexes
    )

    for source in sources:
        if source not in current_indexes:
            logger.warning("No current index for source '%s', skipping", source)
            continue
        for index in current_indexes[source]:
            logger.info("Dropping current index '%s' for source '%s'", index, source)
            tim_ecs_runner(["delete", f"--index={index}", "--force"])

    if untouched := set(current_indexes) - set(sources):
        logger.warning(
            "Sources in alias '%s' but not in dataset were left alone: %s",
            OPENSEARCH_PRIMARY_ALIAS,
            sorted(untouched),
        )


def reindex_all_sources() -> None:
    """Invoke TIM as ECS task to fully index all sources.

    Loop through sources and perform a full re-indexing of all current records via TIM.

    TIM `reindex-source` creates a new index for the source and promotes it to the
    primary alias and the source alias.  Any other aliases for the source, per
    OPENSEARCH_INDEX_ALIASES, are passed as --alias options.

    Embeddings are skipped here and added in the enrichments step.

    Monitor until all ECS tasks are complete.
    """
    sources = _get_timdex_source_list()

    for source in sources:
        command = [
            "reindex-source",
            f"--source={source}",
            *[
                f"--alias={alias}"
                for alias, alias_sources in OPENSEARCH_INDEX_ALIASES.items()
                if source in alias_sources
            ],
            "--skip-embeddings",
            f"s3://{STAGE_BUCKET}/{DATASET_PREFIX}",
        ]
        logger.info("Reindexing source '%s'", source)
        tim_ecs_runner(command)


def index_all_enrichments() -> None:
    """Invoke TIM as ECS tasks to update all docs with embeddings, then fulltexts.

    Each source is updated in its primary-aliased index (the index created by the
    reindex step) with all of its current enrichments.  Only sources that have current
    enrichments of a given type in the dataset get a task.

    Monitor until all ECS tasks are complete.
    """
    _index_enrichments("embeddings", "bulk-update-embeddings")
    _index_enrichments("fulltexts", "bulk-update-fulltexts")


def _index_enrichments(enrichment: str, tim_command: str) -> None:
    sources = _get_timdex_source_list(table=f"current_{enrichment}")
    logger.info("Sources with current %s: %s", enrichment, sources)
    for source in sources:
        logger.info("Updating source '%s' with %s", source, enrichment)
        tim_ecs_runner(
            [tim_command, f"--source={source}", f"s3://{STAGE_BUCKET}/{DATASET_PREFIX}"]
        )


# (name, description, function), numbered from 1 in this order
STEPS = [
    ("delete", "delete stage dataset data (prompts first)", delete_stage_dataset_data),
    ("sync", "rclone sync prod -> stage dataset", rclone_sync_prod_data_to_stage),
    ("metadata", "rebuild stage dataset metadata", rebuild_dataset_metadata),
    ("drop", "drop current stage OpenSearch indexes", drop_opensearch_indexes),
    ("reindex", "reindex all sources in stage OpenSearch", reindex_all_sources),
    ("enrichments", "index embeddings + fulltexts enrichments", index_all_enrichments),
]
STEP_NAMES = [name for name, _, _ in STEPS]


def main(*, start: int = 2) -> None:
    if start == 1:
        while True:
            answer = input(
                f"Delete ALL data and re-sync prod -> stage against '{STAGE_BUCKET}'? "
                "[yes/no]: "
            )
            if answer == "yes":
                break
            if answer == "no":
                raise SystemExit("Aborted by user.")

    for number, (name, description, step) in enumerate(STEPS, start=1):
        if number < start:
            logger.info("Step %d (%s): skipped", number, name)
            continue
        logger.info("Step %d (%s): %s", number, name, description)
        step()


def _parse_step(value: str) -> int:
    """Parse a step number or name into a step number."""
    if value.isdigit() and 1 <= int(value) <= len(STEPS):
        return int(value)
    if value in STEP_NAMES:
        return STEP_NAMES.index(value) + 1
    raise argparse.ArgumentTypeError(
        f"invalid step '{value}', use 1-{len(STEPS)} or one of: {', '.join(STEP_NAMES)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync TIMDEX dataset prod -> stage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="steps:\n"
        + "\n".join(
            f"  {number}  {name:<12} {description}"
            for number, (name, description, _) in enumerate(STEPS, start=1)
        ),
    )
    parser.add_argument(
        "--start",
        type=_parse_step,
        default=2,
        help="step to start from, by number or name (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(start=args.start)
