import re
from datetime import UTC, date, datetime

import pytest

from timdex_dataset_api.record import DatasetRecord


def test_dataset_record_init_with_valid_run_date_parses_year_month_day():
    values = {
        "timdex_record_id": "alma:123",
        "source_record": b"<record><title>Hello World.</title></record>",
        "transformed_record": b"""{"title":["Hello World."]}""",
        "source": "libguides",
        "run_date": "2024-12-01",
        "run_type": "full",
        "action": "index",
        "run_id": "000-111-aaa-bbb",
        "run_record_offset": 0,
    }
    record = DatasetRecord(**values)

    assert record
    assert (record.year, record.month, record.day) == (
        "2024",
        "12",
        "01",
    )


def test_dataset_record_init_with_invalid_run_date_raise_error():
    values = {
        "timdex_record_id": "alma:123",
        "source_record": b"<record><title>Hello World.</title></record>",
        "transformed_record": b"""{"title":["Hello World."]}""",
        "source": "libguides",
        "run_date": "-12-01",
        "run_type": "full",
        "action": "index",
        "run_id": "000-111-aaa-bbb",
        "run_record_offset": 0,
    }

    with pytest.raises(
        ValueError, match=re.escape("time data '-12-01' does not match format '%Y-%m-%d'")
    ):
        DatasetRecord(**values)


def test_dataset_record_serialization():
    values = {
        "timdex_record_id": "alma:123",
        "source_record": b"<record><title>Hello World.</title></record>",
        "transformed_record": b"""{"title":["Hello World."]}""",
        "source": "libguides",
        "run_date": "2024-12-01",
        "run_type": "full",
        "action": "index",
        "run_id": "abc123",
        "run_record_offset": 0,
    }
    dataset_record = DatasetRecord(**values)

    assert dataset_record.to_dict() == {
        "timdex_record_id": "alma:123",
        "source_record": b"<record><title>Hello World.</title></record>",
        "transformed_record": b"""{"title":["Hello World."]}""",
        "source": "libguides",
        "run_date": date(2024, 12, 1),
        "run_type": "full",
        "action": "index",
        "run_id": "abc123",
        "run_record_offset": 0,
        "run_timestamp": datetime(2024, 12, 1, 0, 0, tzinfo=UTC),
        "year": "2024",
        "month": "12",
        "day": "01",
    }


@pytest.mark.parametrize(
    ("run_timestamp_input", "expected_run_timestamp", "expected_exception"),
    [
        (
            None,
            None,
            TypeError,  # expecting string, not None
        ),
        (
            date(2025, 1, 1),
            None,
            TypeError,  # expecting string, not datetime object
        ),
        (
            "2024-12-01T10:00:00Z",
            datetime(2024, 12, 1, 10, 0, tzinfo=UTC),
            None,
        ),
        (
            "2024-12-01T23:59:59.999999+00:00",
            datetime(2024, 12, 1, 23, 59, 59, 999999, tzinfo=UTC),
            None,
        ),
    ],
)
def test_dataset_record_run_timestamp_parsing(
    run_timestamp_input, expected_run_timestamp, expected_exception
):
    values = {
        "timdex_record_id": "alma:123",
        "source_record": b"<record><title>Hello World.</title></record>",
        "transformed_record": b"""{"title":["Hello World."]}""",
        "source": "libguides",
        "run_date": "2024-12-01",
        "run_type": "full",
        "action": "index",
        "run_id": "abc123",
        "run_record_offset": 0,
        "run_timestamp": run_timestamp_input,
    }
    if not expected_exception:
        dataset_record = DatasetRecord(**values)
        assert dataset_record.to_dict() == {
            "timdex_record_id": "alma:123",
            "source_record": b"<record><title>Hello World.</title></record>",
            "transformed_record": b"""{"title":["Hello World."]}""",
            "source": "libguides",
            "run_date": date(2024, 12, 1),
            "run_type": "full",
            "action": "index",
            "run_id": "abc123",
            "run_record_offset": 0,
            "run_timestamp": expected_run_timestamp,
            "year": "2024",
            "month": "12",
            "day": "01",
        }
    else:
        with pytest.raises(expected_exception):
            DatasetRecord(**values)
