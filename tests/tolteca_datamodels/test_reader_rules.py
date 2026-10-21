"""Current-value rules, shared readers and explicit content fallback."""

from enum import Flag
from itertools import permutations
from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest
from pydantic import ValidationError
from tollan.accessor.pandas.model_table import Contains, ModelColumn

from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.core.file._resolution import _SourceReaderResolver
from tolteca_datamodels.core.file.trace import _SourceTraceCollector
from tolteca_datamodels.core.source import SourceReaderRules
from tolteca_datamodels.core.types import DataModelInfo
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.tel import LmtTelAccessor
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_dummy_instru.file import DummyInstruFileAccessor


class _FirstDataModel(DataModelInfo):
    name = "first"


class _SecondDataModel(DataModelInfo):
    name = "second"


class _Reader:
    def __init__(self, source):
        self.source = source

    @property
    def meta(self):
        return self.source.meta


def test_reader_declarations_validate_callables_and_rule_targets():
    with pytest.raises(ValidationError):
        SourceReaderRules(rules=(({}, lambda obj: obj),))
    with pytest.raises(ValidationError):
        SourceReaderRules(rules=(), prepare_record=42)
    with pytest.raises(ValidationError):
        SourceReaderRules(rules=(), default=_Reader)
    policy = SourceReaderRules(rules=(({}, _Reader),))
    assert policy.accessor_classes == frozenset({_Reader})
    assert policy.identify(object()) is None


def test_resolver_validates_declarations_and_choices_before_binding():
    def unexpected(obj):
        pytest.fail("Invalid reader choices reached a payload")

    class Reader:
        __init__ = unexpected

    foreign = _Reader
    with pytest.raises(ValidationError) as exc:
        _SourceReaderResolver.model_validate({"handlers": {"one": 42}})
    assert exc.value.errors()[0]["loc"] == ("handlers", "one")
    resolver = _SourceReaderResolver.model_validate(
        {"handlers": {"one": {"rules": (({}, Reader),)}}}
    )
    assert resolver.handlers["one"].accessor_classes == frozenset({Reader})
    with pytest.raises(ValidationError):
        resolver(object(), choices={"one": 42})  # ty: ignore[invalid-argument-type] -- Test runtime rejection.
    with pytest.raises(ValueError, match="Unknown reader handler"):
        resolver(object(), choices={"other": Reader})
    with pytest.raises(ValueError, match="not declared"):
        resolver(object(), choices={"one": foreign})


@pytest.mark.parametrize("order", list(permutations(("lmt", "toltec"))))
def test_shared_rule_result_binds_and_reads_once_without_identification(
    order, monkeypatch
):
    calls = []
    batch_matches = []
    match_table = SourceReaderRules.map_table

    def tracked(policy, table, **kwargs):
        batch_matches.append(policy)
        return match_table(policy, table, **kwargs)

    monkeypatch.setattr(SourceReaderRules, "map_table", tracked)

    def identify(obj):
        pytest.fail("A determined reader triggered content inspection")

    class Reader(_Reader):
        def __init__(self, obj):
            calls.append("bind")
            super().__init__(obj)

        @property
        def meta(self):
            calls.append("read")
            return super().meta

    rules = SourceReaderRules(
        prepare_record=identify, rules=(({"interface": "tel"}, Reader),)
    )
    resolver = _SourceReaderResolver(handlers=dict.fromkeys(order, rules))
    matched = resolver.match_table(pd.DataFrame({"interface": ["tel"]}))
    assert len(batch_matches) == 1
    assert tuple(matched) == order
    payload = SimpleNamespace(meta=object())
    result = resolver(payload, choices={key: matched[key][0] for key in order})
    assert result is not None
    assert result.reader.source is payload
    assert set(result.handlers) == set(order)
    assert result.reader.meta is payload.meta
    assert calls == ["bind", "read"]
    changed = resolver.match_table(pd.DataFrame({"interface": ["other"]}))
    assert len(batch_matches) == 2
    assert all(values == [None] for values in changed.values())


def test_content_fallback_shares_physical_reader_without_scope_precedence():
    calls = []

    class Reader(_Reader):
        def __init__(self, obj):
            calls.append(obj)
            super().__init__(obj)

    resolver = _SourceReaderResolver(
        handlers={
            name: SourceReaderRules(
                rules=(({"data_models": Contains(value=name)}, Reader),),
                prepare_record=lambda obj: obj,
            )
            for name in ("lmt", "toltec", "dummy")
        }
    )
    for affiliation in ("toltec", "dummy", "other"):
        payload = SimpleNamespace(data_models=["lmt", affiliation])
        result = resolver(payload)
        assert result is not None
        assert result.reader.source is payload
        assert result.handlers == (
            ("lmt", affiliation) if affiliation != "other" else ("lmt",)
        )
    assert len(calls) == 3
    assert resolver(object(), choices={}) is None
    assert not _SourceReaderResolver(handlers={})


def test_ambiguous_content_is_not_resolved_by_registration_order():
    class Other(_Reader):
        pass

    first, second = _Reader, Other
    for order in permutations((first, second)):
        handlers = {
            i: SourceReaderRules(rules=(({}, cls),), prepare_record=lambda obj: obj)
            for i, cls in enumerate(order)
        }
        with pytest.raises(RuntimeError, match="Ambiguous source readers"):
            _SourceReaderResolver(handlers=handlers)(object())


def test_distinct_rule_choices_require_content_disambiguation():
    payload = SimpleNamespace()

    class Other(_Reader):
        pass

    first, second = _Reader, Other
    resolver = _SourceReaderResolver(
        handlers={
            "one": SourceReaderRules(
                rules=(({}, first),), prepare_record=lambda obj: None
            ),
            "two": SourceReaderRules(
                rules=(({}, second),), prepare_record=lambda obj: obj
            ),
        }
    )
    result = resolver(payload, choices={"one": first, "two": second})
    assert result is not None
    assert type(result.reader) is Other
    assert result.reader.source is payload
    assert result.handlers == ("two",)


@pytest.mark.parametrize("error_type", [ValueError, KeyError, RuntimeError])
@pytest.mark.parametrize("stage", ["bind", "identify"])
def test_selected_binding_and_identifier_errors_are_not_nonmatches(error_type, stage):
    def fail(obj):
        message = "Corrupt data"
        raise error_type(message)

    def unexpected(obj):
        pytest.fail("Failure triggered another interpretation")

    class Reader:
        def __init__(self, obj):
            fail(obj)

    first = Reader
    resolver = _SourceReaderResolver(
        handlers={
            "one": SourceReaderRules(
                rules=(({}, first),),
                prepare_record=fail if stage == "identify" else unexpected,
            )
        }
    )
    choices = {"one": first} if stage == "bind" else None
    with pytest.raises(error_type, match="Corrupt data"):
        resolver(object(), choices=choices)


def test_current_membership_and_identity_route_without_filename_fallback():
    handlers = (LmtFileAccessor, ToltecFileAccessor, DummyInstruFileAccessor)
    original = FilepathInfo.from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    # A rule-input table, not a mutation of a previously bound file table.
    table = pd.DataFrame(
        {
            "source": ["record.nc"] * 5,
            "source_loc": ["record.nc"] * 5,
            "data_models": [["dummy_instru"], ["toltec"], [], None, []],
            "instru": ["dummy_instru", None, "lmt", None, "lmt"],
            "interface": ["chan01", None, "tel", None, "tel"],
            "interface_kind": ["counter", None, "tel", None, "tel"],
            "filepath_info": [original, None, original, original, None],
            "file_data_models": [["lmt", "toltec"]] * 5,
            "file_interface": ["tel"] * 5,
        }
    )
    accessor = FileAccessor(table, accessor_types=handlers)
    choices = accessor._get_reader_choices()
    assert set(choices[0]) == {DummyInstruFileAccessor}
    assert choices[0][DummyInstruFileAccessor] is not None
    assert choices[1] == {ToltecFileAccessor: None}
    assert choices[2] == {}
    assert choices[3] == dict.fromkeys(handlers)
    assert choices[4] == {}
    trace = _SourceTraceCollector("raise")
    accessor._record_scope(trace)
    trace.update_table(table)
    assert [events[-1].outcome for events in table.source_trace] == [
        "in_scope",
        "in_scope",
        "out_of_scope",
        "unrecognized",
        "out_of_scope",
    ]
    assert table.file_data_models.tolist() == [["lmt", "toltec"]] * 5
    assert table.file_interface.tolist() == ["tel"] * 5
    assert table.filepath_info.iloc[0] is original
    accessor._guess_data_kinds()
    assert table.data_kind_toltec.tolist() == [ToltecDataKind.unknown.value] * 5


def test_current_extension_null_does_not_restore_filename_extension():
    table = pd.DataFrame(
        {
            "source": ["record.nc"],
            "source_loc": ["record.nc"],
            "file_ext": ["nc"],
            "ext": [None],
        }
    )
    assert table.file_ext.tolist() == ["nc"]
    assert all(
        value is None for value in FileAccessor(table)._get_opener_choices()[0].values()
    )


def test_membership_constraints_apply_without_file_accessor_scope_filtering():
    policy = ToltecFileAccessor._reader_rules
    records = [
        {"data_models": ["lmt", "toltec"], "interface": "tel"},
        {"data_models": ["lmt", "dummy_instru"], "interface": "tel"},
        {"data_models": [], "interface": "tel"},
        {"data_models": None, "interface": "tel"},
    ]
    expected = [LmtTelAccessor, None, None, None]
    assert [policy(record) for record in records] == expected
    assert [policy(SimpleNamespace(**record)) for record in records] == expected
    assert policy.map_table(pd.DataFrame(records)) == expected


def test_rule_lookup_preserves_order_and_null_semantics():
    def unexpected(obj):
        pytest.fail("Rule matching accessed a payload")

    class Other(_Reader):
        pass

    first, second = _Reader, Other
    policy = SourceReaderRules(
        prepare_record=unexpected,
        rules=(({"kind": None}, first), ({"ext": "nc"}, second)),
    )
    records = [
        {"kind": None, "ext": "nc"},
        {"kind": "other", "ext": "nc"},
        {"kind": "other", "ext": "unknown"},
    ]
    expected = [first, second, None]
    for record, spec in zip(records, expected, strict=True):
        assert policy(record) is spec
        assert policy(SimpleNamespace(**record)) is spec
    assert policy({}) is None
    assert SourceReaderRules(rules=())(records[0]) is None
    table = pd.DataFrame(records).rename(columns={"ext": "file_ext"})
    matched = policy.map_table(table, column_names={"ext": "file_ext"})
    assert all(a is b for a, b in zip(matched, expected, strict=True))
    resolver = _SourceReaderResolver(handlers={"test": policy})
    assert resolver.match_table(table, column_names={"ext": "file_ext"}) == {
        "test": expected
    }


@pytest.mark.parametrize("fail_classification", [False, True])
def test_shared_read_has_independent_classifiers_and_one_complete_row_update(
    fail_classification,
):
    reads = []

    class Reader(_Reader):
        @property
        def meta(self):
            reads.append(self.source)
            return super().meta

    FirstKind = Flag("FirstKind", {"counts": 1, "unknown": 2})
    SecondKind = Flag("SecondKind", {"quality": 4, "unknown": 8})
    policy = SourceReaderRules(rules=(({}, Reader),))

    class First(FileAccessor[_FirstDataModel]):
        _reader_rules = policy
        _data_kind_type = FirstKind
        _data_kind_column = "data_kind_first"
        _metadata_columns: ClassVar = {"value": ModelColumn("value", dtype="Int64")}

        @staticmethod
        def _classify_metadata(reader, metadata):
            return FirstKind.counts

    class Second(FileAccessor[_SecondDataModel]):
        _reader_rules = policy
        _data_kind_type = SecondKind
        _data_kind_column = "data_kind_second"

        @staticmethod
        def _classify_metadata(reader, metadata):
            if fail_classification:
                message = "Classification failed"
                raise ValueError(message)
            return SecondKind.quality

    payload = SimpleNamespace(
        meta=SimpleNamespace(value=7, data_models=["first", "second"])
    )
    handlers = (First, Second)
    table = FileAccessor.from_sources(
        [{"source": payload, "source_loc": "record.nc"}], accessor_types=handlers
    )
    table["value"] = pd.array([99], dtype="Int64")
    result = FileAccessor(table, accessor_types=handlers).update_from_source_metadata()
    assert reads == [payload]
    if fail_classification:
        pd.testing.assert_frame_equal(
            result.drop(columns="source_trace"), table.drop(columns="source_trace")
        )
        assert (
            result.source_trace.map(source_trace_issues).iloc[0][-1].message
            == "Classification failed"
        )
    else:
        assert result.value.tolist() == [7]
        assert result.data_kind_first.tolist() == [FirstKind.counts.value]
        assert result.data_kind_second.tolist() == [SecondKind.quality.value]
