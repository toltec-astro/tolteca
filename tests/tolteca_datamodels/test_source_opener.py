"""Rule-selected opening with explicit sources and caller-owned resources."""

from io import StringIO
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from tolteca_datamodels.core.source import SourceOpener


def test_open_separates_matching_record_from_unchanged_source():
    source = object()
    resource = StringIO("measurement")
    calls = []

    def acquire(value, **kwargs):
        calls.append((value, kwargs))
        return resource

    policy = SourceOpener(rules=(({"format": "text"}, acquire),))
    record = SimpleNamespace(format="text", source="not the actual resource")
    assert policy(record) is acquire
    assert calls == []
    assert policy.open(record, source=source, encoding="utf-8") is resource
    assert calls == [(source, {"encoding": "utf-8"})]
    assert not resource.closed
    resource.close()


def test_composition_preserves_original_policy_and_first_match():
    def first(source):
        return source

    def override(source):
        return source

    original = SourceOpener(rules=(({"format": "text"}, first),))
    combined = SourceOpener(
        rules=(({"format": "text", "extra": ...}, override), *original.rules)
    )
    assert original({"format": "text"}) is first
    assert combined({"format": "text"}) is override
    assert combined.map_table(pd.DataFrame({"format": ["text", "other"]})) == [
        override,
        None,
    ]


@pytest.mark.parametrize("rules", [(({"format": "a"}, 1),), (({}, None),)])
def test_invalid_rules_are_rejected_at_construction(rules):
    with pytest.raises(ValidationError):
        SourceOpener(rules=rules)


def test_unknown_record_does_not_infer_format_from_source():
    def unexpected(source):
        pytest.fail("Unmatched record opened a source")

    policy = SourceOpener(rules=(({"ext": "txt"}, unexpected),))
    assert policy({}) is None
    with pytest.raises(ValueError, match="No source opener matches record"):
        policy.open({}, source="looks-like-a-file.txt")


def test_selected_opener_failure_does_not_fall_through():
    error = OSError("Cannot open source")

    def fail(source):
        raise error

    def unexpected(source):
        pytest.fail("Another opener was tried")

    policy = SourceOpener(rules=(({}, fail), ({}, unexpected)))
    with pytest.raises(OSError) as exc:
        policy.open({}, source=object())
    assert exc.value is error


def test_batch_opens_lazily_and_preserves_row_order():
    calls = []

    def acquire(source, **kwargs):
        calls.append((source, kwargs))
        return StringIO(source)

    policy = SourceOpener(rules=(({"format": "text"}, acquire),))
    table = pd.DataFrame({"kind": ["text"] * 2, "payload": ["a", "b"]}, index=[5, 5])
    before = table.copy(deep=True)

    results = policy.iter_open_table(
        table, source_col="payload", column_names={"format": "kind"}, marker=True
    )
    assert calls == []
    pd.testing.assert_frame_equal(table, before)
    with next(results) as first:
        assert first.read() == "a"
        assert calls == [("a", {"marker": True})]
    with next(results) as second:
        assert second.read() == "b"
    assert list(results) == []
    assert calls == [("a", {"marker": True}), ("b", {"marker": True})]


@pytest.mark.parametrize("failure", ["unmatched", "backend"])
def test_batch_failure_preserves_ownership_of_previously_yielded_objects(failure):
    opened = []

    def acquire(source):
        if source == "bad":
            msg = "Cannot open source"
            raise OSError(msg)
        obj = StringIO(source)
        opened.append(obj)
        return obj

    policy = SourceOpener(rules=(({"format": "text"}, acquire),))
    table = pd.DataFrame(
        {
            "format": ["text", "other" if failure == "unmatched" else "text", "text"],
            "source": ["first", "bad", "last"],
        }
    )
    results = policy.iter_open_table(table, source_col="source")
    first = next(results)
    try:
        with pytest.raises(ValueError if failure == "unmatched" else OSError):
            next(results)
        assert opened == [first]
        assert not first.closed
        assert list(results) == []
    finally:
        first.close()


def test_early_termination_does_not_close_transferred_objects():
    policy = SourceOpener(rules=(({}, StringIO),))
    results = policy.iter_open_table(
        pd.DataFrame({"source": ["a", "b"]}), source_col="source"
    )
    first = next(results)
    del results
    assert not first.closed
    first.close()


def test_batch_requires_source_column_even_when_empty():
    policy = SourceOpener(rules=())
    assert (
        list(policy.iter_open_table(pd.DataFrame({"source": []}), source_col="source"))
        == []
    )
    with pytest.raises(KeyError, match="source"):
        list(policy.iter_open_table(pd.DataFrame(), source_col="source"))
