"""Columnar classification and single-source/catalog equivalence."""

from enum import Enum

import pandas as pd
import pyarrow as pa
import pytest
from tollan.utils.fileloc import fileloc

import tolteca_datamodels.file as file_module
from tolteca_datamodels.file import (
    FileIdentifier,
    FilepathInfo,
    FilepathInfoMatch,
    SourceInfoModel,
    ToltecFileAccessor,
    guess_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.types import ToltecDataKind


class ExampleKind(Enum):
    """Test-only classifications, not acquisition metadata."""

    first = "first"
    absent = "absent"
    empty = "empty"
    fallback = "fallback"


@pytest.mark.parametrize("backend", ["pandas", "arrow", "arrow_chunked"])
def test_first_match_null_empty_and_ignored_fields(monkeypatch, backend):
    # Deliberately overlapping rules test the classifier, not filename formats.
    monkeypatch.setattr(
        FileIdentifier,
        "filepath_info_data_kind",
        {
            FilepathInfoMatch(ext="nc"): ExampleKind.first,
            FilepathInfoMatch(prefix="match", suffix=None): ExampleKind.absent,
            FilepathInfoMatch(prefix="match", suffix=""): ExampleKind.empty,
        },
    )
    fields = pd.DataFrame(
        {
            "prefix": ["match", "match", "match", "unknown", "match"],
            "ext": ["nc", "txt", "txt", "txt", None],
            "suffix": [None, None, "", None, "unused"],
            "unrelated": [1, 2, 3, 4, 5],
        },
        # Output follows row order, independent of duplicate/nondefault labels.
        index=[9, 9, 2, 5, 0],
    )
    if backend.startswith("arrow"):
        fields = pa.Table.from_pandas(fields, preserve_index=False)
        if backend == "arrow_chunked":
            fields = pa.concat_tables([fields.slice(0, 2), fields.slice(2)])
    result = FileIdentifier.guess_data_kinds(fields)
    assert result == [
        ExampleKind.first,
        ExampleKind.absent,
        ExampleKind.empty,
        ToltecDataKind.Unknown,
        ToltecDataKind.Unknown,
    ]


def test_catchall_follows_declaration_order_not_specificity(monkeypatch):
    monkeypatch.setattr(
        FileIdentifier,
        "filepath_info_data_kind",
        {
            FilepathInfoMatch(ext="nc"): ExampleKind.first,
            FilepathInfoMatch(): ExampleKind.fallback,
            FilepathInfoMatch(ext="txt", suffix="specific"): ExampleKind.empty,
        },
    )
    assert FileIdentifier.guess_data_kinds(pa.table({"ext": ["nc", "txt", None]})) == [
        ExampleKind.first,
        ExampleKind.fallback,
        ExampleKind.fallback,
    ]
    assert FileIdentifier.guess_data_kinds(
        pa.table({"ext": ["txt"], "suffix": ["specific"]})
    ) == [ExampleKind.fallback]


def test_missing_columns_and_explicit_nulls(monkeypatch):
    monkeypatch.setattr(
        FileIdentifier,
        "filepath_info_data_kind",
        {
            FilepathInfoMatch(suffix=None): ExampleKind.absent,
        },
    )
    assert (
        FileIdentifier.guess_data_kinds(pd.DataFrame(index=[1, 2]))
        == [ExampleKind.absent] * 2
    )
    assert (
        FileIdentifier.guess_data_kinds(pa.table({"unrelated": [1, 2]}))
        == [ExampleKind.absent] * 2
    )
    assert FileIdentifier.guess_data_kinds(pd.DataFrame({"suffix": [pd.NA, ""]})) == [
        ExampleKind.absent,
        ToltecDataKind.Unknown,
    ]


def test_empty_input_and_empty_registry(monkeypatch):
    assert FileIdentifier.guess_data_kinds(pd.DataFrame()) == []
    assert FileIdentifier.guess_data_kinds(pa.table({})) == []
    monkeypatch.setattr(FileIdentifier, "filepath_info_data_kind", {})
    assert FileIdentifier.guess_data_kinds(pa.table({"ext": ["nc", "txt"]})) == [
        ToltecDataKind.Unknown,
        ToltecDataKind.Unknown,
    ]


def test_enum_identity_is_not_lost_to_equal_values(monkeypatch):
    class OtherKind(Enum):
        first = "first"

    monkeypatch.setattr(
        FileIdentifier,
        "filepath_info_data_kind",
        {
            FilepathInfoMatch(ext="nc"): ExampleKind.first,
            FilepathInfoMatch(ext="txt"): OtherKind.first,
        },
    )
    kinds = FileIdentifier.guess_data_kinds(pa.table({"ext": ["nc", "txt"]}))
    assert kinds[0] is ExampleKind.first
    assert kinds[1] is OtherKind.first


def test_comparisons_are_reused_only_within_a_batch(monkeypatch):
    monkeypatch.setattr(
        FileIdentifier,
        "filepath_info_data_kind",
        {
            FilepathInfoMatch(prefix="example", ext="nc"): ExampleKind.first,
            FilepathInfoMatch(prefix="example", ext="txt"): ExampleKind.empty,
        },
    )
    calls = []
    equal = file_module.pc.equal  # ty: ignore[unresolved-attribute]

    def trace(column, value):
        calls.append(value)
        return equal(column, value)

    monkeypatch.setattr(file_module.pc, "equal", trace)
    for _ in range(2):
        calls.clear()
        assert FileIdentifier.guess_data_kinds(
            pa.table({"prefix": ["example"], "ext": ["nc"]})
        ) == [
            ExampleKind.first,
        ]
        assert calls == ["example", "nc", "txt"]


def test_registry_changes_are_visible_on_next_call(monkeypatch):
    rules = {FilepathInfoMatch(ext="nc"): ExampleKind.first}
    monkeypatch.setattr(FileIdentifier, "filepath_info_data_kind", rules)
    fields = pa.table({"ext": ["nc"]})
    assert FileIdentifier.guess_data_kinds(fields) == [ExampleKind.first]
    rules[FilepathInfoMatch(ext="nc")] = ExampleKind.empty
    assert FileIdentifier.guess_data_kinds(fields) == [ExampleKind.empty]


def test_recorded_paths_batch_matches_single_source(lmtmc_records, toltecdb_records):
    paths = [row["FileName"] for row in [*lmtmc_records, *toltecdb_records]]
    single = []
    for path in paths:
        source = guess_info_from_source(path)
        assert source is not None
        single.append(source)
    pd.testing.assert_frame_equal(
        guess_info_from_sources(iter(paths)), ToltecFileAccessor.from_info_list(single)
    )


@pytest.mark.parametrize("size", [0, 1, 1000])
@pytest.mark.parametrize("entrypoint", ["function", "accessor"])
def test_catalog_classifies_once_per_batch(monkeypatch, size, entrypoint):
    path = "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
    paths = ([path, "unrecognized.nc"] * size)[:size]
    calls = []
    classify = FileIdentifier.guess_data_kinds

    def trace(fields):
        calls.append(len(fields))
        return classify(fields)

    def unexpected(*args, **kwargs):
        pytest.fail("Catalog construction used the one-record classifier")

    monkeypatch.setattr(FileIdentifier, "guess_data_kinds", trace)
    monkeypatch.setattr(file_module, "guess_info_from_source", unexpected)
    monkeypatch.setattr(FileIdentifier, "guess_data_kind", unexpected)
    build = (
        guess_info_from_sources
        if entrypoint == "function"
        else ToltecFileAccessor.from_sources
    )
    catalog = build(iter(paths))
    assert calls == [size]
    assert len(catalog) == size
    assert catalog["data_kind"].tolist() == [
        "VnaSweep" if i % 2 == 0 else "Unknown" for i in range(size)
    ]


def test_single_record_uses_the_batch_classifier(monkeypatch):
    calls = []
    classify = FileIdentifier.guess_data_kinds
    info = FilepathInfo.from_path("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert info is not None

    def trace(fields):
        calls.append(len(fields))
        assert set(fields.column_names) == set(FilepathInfoMatch._fields)
        return classify(fields)

    monkeypatch.setattr(FileIdentifier, "guess_data_kinds", trace)
    assert FileIdentifier.guess_data_kind(info) is ToltecDataKind.LmtTel
    source = guess_info_from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert source is not None
    assert source.data_kind is ToltecDataKind.LmtTel
    assert calls == [1, 1]


def test_plural_entrypoint_delegates_to_accessor_factory(monkeypatch):
    sources = iter(["tel_toltec_2025-10-31_145609_00_0002.nc"])
    expected = ToltecFileAccessor.from_sources([])
    calls = []

    def build(passed):
        calls.append(passed)
        return expected

    monkeypatch.setattr(ToltecFileAccessor, "from_sources", build)
    assert guess_info_from_sources(sources) is expected
    assert calls == [sources]
    assert next(sources) == "tel_toltec_2025-10-31_145609_00_0002.nc"


@pytest.mark.parametrize("prefix", ["", "host:/data/"])
def test_single_source_keeps_fileloc_without_catalog_roundtrip(
    monkeypatch, tmp_path, prefix
):
    source = fileloc(
        f"{prefix}tel_toltec_2025-10-31_145609_00_0002.nc",
        local_parent_path=tmp_path,
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Single-source identification built a catalog")

    monkeypatch.setattr(ToltecFileAccessor, "from_sources", unexpected)
    monkeypatch.setattr(ToltecFileAccessor, "from_info_list", unexpected)
    monkeypatch.setattr(ToltecFileAccessor, "to_info_list", unexpected)
    record = guess_info_from_source(source)
    assert record is not None
    assert record.source is source
    assert record.source.root.local_parent_path == tmp_path
    assert record.source.root.remote_parent_path is None
    assert record.data_kind is ToltecDataKind.LmtTel


def test_accessor_binding_does_not_classify_or_mutate(monkeypatch):
    catalog = ToltecFileAccessor.from_sources(
        ["tel_toltec_2025-10-31_145609_00_0002.nc"]
    )
    before = catalog.copy(deep=True)

    def unexpected(*args, **kwargs):
        pytest.fail("Binding an accessor reidentified the source")

    monkeypatch.setattr(FileIdentifier, "parse_filepath", unexpected)
    monkeypatch.setattr(FileIdentifier, "guess_data_kinds", unexpected)
    assert catalog.toltec_file._obj is catalog
    pd.testing.assert_frame_equal(catalog, before)


def test_preclassified_source_records_are_not_classified_again(monkeypatch):
    info = FilepathInfo.from_path("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert info is not None
    source = SourceInfoModel.model_validate(
        {
            "source": "tel_toltec_2025-10-31_145609_00_0002.nc",
            "filepath_info": info,
            "data_kind": ToltecDataKind.LmtTel,
        }
    )

    def unexpected(*args, **kwargs):
        pytest.fail("An existing source record was reclassified")

    monkeypatch.setattr(FileIdentifier, "guess_data_kinds", unexpected)
    catalog = ToltecFileAccessor.from_info_list([source])
    restored = catalog.toltec_file.to_info_list()[0]
    assert restored.source.url == source.source.url
    assert restored.model_dump(
        round_trip=True, exclude={"source"}
    ) == source.model_dump(round_trip=True, exclude={"source"})
