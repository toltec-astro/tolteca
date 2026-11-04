"""Reader policies select classes without opening or binding resources."""

from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError
from tollan.accessor.pandas.model_table import Contains

from tolteca_datamodels.core.source import SourceReaderRules


class _Reader:
    def __init__(self, source):
        pytest.fail("Selecting a reader constructed an accessor")

    @property
    def meta(self) -> object:
        pytest.fail("Selecting a reader read metadata")


def test_record_table_and_content_identification_use_the_same_rules():
    record = SimpleNamespace(interface="tel", data_models=["lmt", "toltec"])
    source = object()
    inspected = []

    def prepare(data):
        inspected.append(data)
        return record

    policy = SourceReaderRules(
        rules=(
            ({"interface": "tel", "data_models": Contains(value="toltec")}, _Reader),
        ),
        prepare_record=prepare,
    )
    table = pd.DataFrame(
        {
            "interface": ["tel", "tel"],
            "data_models": [["lmt", "toltec"], ["lmt"]],
        },
        index=[5, 5],
    )
    assert policy(record) is _Reader
    assert policy.map_table(table) == [_Reader, None]
    assert inspected == []
    assert policy.identify(source) is _Reader
    assert inspected == [source]


@pytest.mark.parametrize("record", [None, {"interface": "unknown"}])
def test_unsupported_content_has_no_reader(record):
    policy = SourceReaderRules(
        rules=(({"interface": "tel"}, _Reader),),
        prepare_record=lambda source: record,
    )
    assert policy.identify(object()) is None


def test_record_only_policy_does_not_attempt_content_identification():
    policy = SourceReaderRules(rules=(({"interface": "tel"}, _Reader),))
    assert policy({"interface": "tel"}) is _Reader
    assert policy.identify(object()) is None


def test_content_inspection_failure_is_not_an_identification_miss():
    error = OSError("Unreadable identification header")

    def prepare(source):
        raise error

    policy = SourceReaderRules(rules=(({}, _Reader),), prepare_record=prepare)
    with pytest.raises(OSError) as caught:
        policy.identify(object())
    assert caught.value is error


def test_reader_rules_require_accessor_classes():
    with pytest.raises(ValidationError):
        SourceReaderRules(rules=(({}, "not an accessor class"),))
