import re
from datetime import date

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
        "year": "2024",
        "month": "12",
        "day": "01",
    }
