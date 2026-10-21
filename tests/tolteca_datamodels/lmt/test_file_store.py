"""Instrument selection and delegation in the LMT acquisition tree."""

from pathlib import Path, PurePosixPath
from typing import ClassVar, assert_type

import pytest

from tolteca_datamodels.core.file_store import FileStoreBase, StoreView, Substore
from tolteca_datamodels.core.obs_spec import ObsSpecContext
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt_base.file_store import LmtFileStoreBase
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec._obs_spec import (
    _obs_spec_context as toltec_obs_spec_context,
)
from tolteca_datamodels.toltec.file_store import ToltecFileStore
from tolteca_datamodels.toltec.types import ToltecInfo


@pytest.mark.parametrize(
    "names",
    [
        ["toltec", "rsr", "sequoia"],
        ["toltec"],
        ["toltec", "rsr"],
        ["toltec", "sequoia"],
    ],
)
def test_named_selections_choose_registered_children(tmp_path, names):
    calls = []

    # Stand-ins exercise registry delegation, not unimplemented acquisition layouts.
    class Store(LmtFileStoreBase):
        _subpaths = (PurePosixPath("."),)

        def _iter_local_glob_patterns(self, spec, subpath):
            calls.append((self.path, spec))
            return []

    class RsrStore(Store):
        pass

    class SequoiaStore(Store):
        pass

    class ToltecStore(Store):
        pass

    class Root(LmtFileStore):
        _substore_paths: ClassVar = {}
        toltec = Substore(ToltecStore, subpath="toltec")
        rsr = Substore(RsrStore, subpath="rsr")
        sequoia = Substore(SequoiaStore, subpath="sequoia")

    spec = ObsSpec.parse("tcs-145609")
    store = Root(tmp_path)
    assert store.partial(*names).glob_obs_spec(spec) == []
    assert [path for path, _ in calls] == [tmp_path / name for name in names]
    assert all(delegated is spec for _, delegated in calls)


def test_nested_substore_selection_needs_no_layout_knowledge(tmp_path, monkeypatch):
    class Child(LmtFileStoreBase):
        _subpaths = (PurePosixPath("."),)
        _obs_spec_context = toltec_obs_spec_context

        def _iter_local_glob_patterns(self, spec, subpath):
            yield "*.nc"

    class Root(LmtFileStore):
        _substore_paths: ClassVar = {}
        instrument = Substore(Child, subpath="instruments/toltec")

    root = Root(tmp_path)
    telescope = root.tel.path / "tel_toltec_2025-10-31_145609_00_0002.nc"
    instrument = (
        tmp_path
        / "instruments/toltec"
        / ("toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc")
    )
    calls = []

    def glob(path, pattern):
        calls.append(path)
        return iter([telescope] if path == root.tel.path else [instrument])

    monkeypatch.setattr(Path, "glob", glob)
    assert root.partial(Root.instrument).glob_obs_spec("145609/{tel,0}") == [instrument]
    assert calls == [instrument.parent]


@pytest.mark.parametrize(
    "selector", ["145609", "145609/*", "145609/tel", "145609/{tel,0}"]
)
def test_excluded_instrument_is_not_searched_but_telescope_remains(
    data_lmt_path, monkeypatch, selector
):
    def unexpected(*args, **kwargs):
        pytest.fail("An excluded instrument was searched")

    monkeypatch.setattr(ToltecFileStore, "_iter_local_glob_patterns", unexpected)
    store = LmtFileStore(data_lmt_path)
    paths = store.tel.glob_obs_spec(selector)
    assert len(paths) == 2
    assert all(path.parent == data_lmt_path / "tel" for path in paths)


@pytest.mark.parametrize(
    "selector",
    ["145609", "145609/*", "145609/{}", "145609/[]", "145609/[:]", "145609/[-1]"],
)
def test_unconstrained_interface_probes_telescope_and_instrument(
    data_lmt_path, monkeypatch, selector
):
    searched = []
    glob = Path.glob

    def record_glob(path, pattern):
        searched.append(path)
        return glob(path, pattern)

    monkeypatch.setattr(Path, "glob", record_glob)
    store = LmtFileStore(data_lmt_path)
    paths = store.glob_obs_spec(selector)
    assert searched == [store.tel.path] + [
        store.toltec.path
        for interface, info in ToltecInfo.interface_interface_info.items()
        if info.instru == "toltec"
    ]
    assert len(paths) == 5
    assert sum(path.parent == store.tel.path for path in paths) == 2
    # Auxiliary candidates are gathered before the full ObsSpec table filter.
    assert [path.name for path in paths if path.parent.name == "hwpr"] == [
        "hwpr_156755_000_0002_2026_05_16_12_33_28.nc"
    ]


@pytest.mark.parametrize(
    "selector",
    [
        "145609/0",
        "145609/toltec0",
        "145609/hwpr",
        "145609/{0,1}",
        "145609/{toltec0,hwpr}",
        "145609/(0,3)",
        "145609/()",
        "145609/(,)",
        "145609/()[-1]",
    ],
)
def test_constrained_interface_excluding_tel_does_not_probe_telescope(
    data_lmt_path, monkeypatch, selector
):
    store = LmtFileStore(data_lmt_path)
    searched = []
    glob = Path.glob

    def guarded_glob(path, pattern):
        assert path != store.tel.path, "Excluded telescope directory was probed"
        searched.append(path)
        return glob(path, pattern)

    monkeypatch.setattr(Path, "glob", guarded_glob)
    store.glob_obs_spec(selector)
    assert searched
    assert all(path.is_relative_to(store.toltec.path) for path in searched)


@pytest.mark.parametrize("selector", ["145609/tel", "145609/{tel}"])
def test_telescope_only_does_not_probe_instrument_stores(
    data_lmt_path, monkeypatch, selector
):
    store = LmtFileStore(data_lmt_path)
    glob = Path.glob

    def guarded_glob(path, pattern):
        assert path == store.tel.path, "Telescope-only selection probed an instrument"
        return glob(path, pattern)

    monkeypatch.setattr(Path, "glob", guarded_glob)
    paths = store.glob_obs_spec(selector)
    assert len(paths) == 2
    assert all(path.parent == store.tel.path for path in paths)


def test_default_search_includes_instrument_discovery(data_lmt_path):
    store = LmtFileStore(data_lmt_path)
    assert store.glob_obs_spec("145609/0") == store.toltec.glob_obs_spec("145609/0")


def test_lmt_root_is_not_a_managed_location(data_lmt_path, monkeypatch):
    def glob(root, pattern):
        assert root != data_lmt_path
        return iter(())

    monkeypatch.setattr(Path, "glob", glob)
    store = LmtFileStore(data_lmt_path)
    assert store.glob_obs_spec("145609") == []


@pytest.mark.parametrize(
    ("subpath", "patterns"),
    [
        (PurePosixPath("tel"), ["tel_*_145609_*.nc"]),
        (PurePosixPath("."), []),
        (PurePosixPath("other"), []),
    ],
)
def test_telescope_patterns_apply_only_to_telescope_subpath(
    tmp_path, subpath, patterns
):
    store = LmtFileStore(tmp_path)
    spec = ObsSpec.parse("145609/tel")
    assert list(store._iter_local_glob_patterns(spec, subpath)) == patterns


def test_candidates_keep_discovery_order_and_first_occurrence(
    data_lmt_path, monkeypatch
):
    store = LmtFileStore(data_lmt_path)
    telescope = sorted(store.tel.path.glob("*145609*.nc"), reverse=True)
    instrument = list(store.toltec.glob_obs_spec("145609/toltec0"))
    assert len(telescope) == len(instrument) == 2

    # Repeated candidates exercise deduplication independently of traversal order.
    candidates = {store.tel.path: telescope}
    candidates[store.toltec.path] = instrument

    def glob(path, pattern):
        matches = (
            candidates.get(path, [])
            if "toltec0" in pattern or path == store.tel.path
            else []
        )
        return iter(matches + matches[:1])

    monkeypatch.setattr(Path, "glob", glob)
    assert store.glob_obs_spec("145609") == telescope + instrument


def test_typed_child_view_and_delegated_instance(tmp_path):
    store = LmtFileStore(tmp_path)
    assert_type(store.path, Path)
    assert_type(store.toltec, StoreView)
    assert_type(store.substores[ToltecFileStore], ToltecFileStore)
    assert isinstance(store, FileStoreBase)
    assert isinstance(store.toltec, StoreView)
    assert store.toltec == store.toltec
    assert store.toltec.path == tmp_path / "toltec"
    assert store.substores[ToltecFileStore].substores == {}


def test_selected_descendant_is_reached_through_unselected_parents(
    tmp_path, monkeypatch
):
    class Nested(LmtFileStoreBase):
        _subpaths = (PurePosixPath("."),)

        def _iter_local_glob_patterns(self, spec, subpath):
            yield "*.nc"

    class Instrument(LmtFileStoreBase):
        def _iter_local_glob_patterns(self, spec, subpath):
            return ()

    class Root(LmtFileStore):
        pass

    Instrument.register_substore(Nested, "nested", name="nested")
    Root.register_substore(Instrument, "instrument", name="instrument")
    path = tmp_path / "instrument/nested/example.nc"
    calls = []

    def glob(root, pattern):
        calls.append(root)
        return iter([path] if root == path.parent else [])

    monkeypatch.setattr(Path, "glob", glob)
    root = Root(tmp_path)
    assert root.partial("instrument").instrument.nested.glob_obs_spec("1") == [path]
    assert calls == [path.parent]


@pytest.mark.parametrize("store_cls", [LmtFileStore, ToltecFileStore])
@pytest.mark.parametrize("path_type", [str, Path])
def test_store_path_is_read_only(tmp_path, store_cls, path_type):
    store = store_cls(path_type(tmp_path / "missing"))
    assert isinstance(store.path, Path)
    with pytest.raises(AttributeError):
        store.path = tmp_path / "other"
    with pytest.raises(AttributeError):
        del store.path
    assert store.path == tmp_path / "missing"


def test_child_remains_consistent_with_read_only_path(tmp_path):
    store = LmtFileStore(tmp_path)
    child = store.substores[ToltecFileStore]
    with pytest.raises(AttributeError):
        store.path = tmp_path / "other"  # ty: ignore[invalid-assignment] -- test runtime protection
    assert store.substores[ToltecFileStore] is child
    assert child.path == store.path / "toltec"


def test_stores_retain_identity_equality(tmp_path):
    first, second = LmtFileStore(tmp_path), LmtFileStore(tmp_path)
    assert first != second
    stores = {first, second}
    assert first.substores[ToltecFileStore] is first.substores[ToltecFileStore]
    assert len(stores) == 2
    assert first in stores


def test_instrument_filter_is_per_search(data_lmt_path):
    store = LmtFileStore(data_lmt_path)
    expected = store.glob_obs_spec("145609/{tel,0}")
    telescope = store.tel.glob_obs_spec("145609/{tel,0}")
    assert len(telescope) == 2
    assert all(path.parent == store.tel.path for path in telescope)
    assert expected == telescope + store.toltec.glob_obs_spec("145609/{tel,0}")
    assert len(expected) > len(telescope)
    assert store.glob_obs_spec("145609/{tel,0}") == expected


def test_additional_typed_child_uses_shared_discovery(tmp_path):
    calls = []

    # A new instrument contributes its own layout without changing discovery.
    class Store(LmtFileStoreBase):
        _subpaths = (PurePosixPath("."),)

        def _iter_local_glob_patterns(self, spec, subpath):
            calls.append((self.path, spec))
            return []

    class ExtendedLmtFileStore(LmtFileStore):
        rsr = Substore(Store, subpath="rsr")

    store = ExtendedLmtFileStore(tmp_path)
    assert_type(store.rsr, StoreView)
    assert store.rsr.path == store.substores[Store].path
    assert store.toltec.path == store.substores[ToltecFileStore].path
    spec = ObsSpec.parse("tcs-145609")
    assert store.rsr.glob_obs_spec(spec) == []
    assert calls == [(tmp_path / "rsr", spec)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "1/(1,5)",
            {
                "toltec": ["toltec1", "toltec2", "toltec3", "toltec4"],
                "foo": ["bar1", "bar2", "bar3"],
            },
        ),
        (
            "1/toltec(1,5)",
            {"toltec": ["toltec1", "toltec2", "toltec3", "toltec4"], "foo": []},
        ),
        ("1/bar(1,5)", {"toltec": [], "foo": ["bar1", "bar2", "bar3"]}),
        (
            "1/(1,5)[-1]",
            {
                "toltec": ["toltec1", "toltec2", "toltec3", "toltec4"],
                "foo": ["bar1", "bar2", "bar3"],
            },
        ),
    ],
)
def test_delegated_stores_resolve_one_spec_in_their_own_context(
    tmp_path, text, expected
):
    resolved = {}

    # Synthetic declarations test the extension point, not acquisition filenames.
    class Store(LmtFileStoreBase):
        _subpaths = (PurePosixPath("."),)

        def _iter_local_glob_patterns(self, spec, subpath):
            interfaces = list(self._obs_spec_context.field_label_numbers["interface"])
            resolved[self.path.name] = spec.resolve_values(
                "interface", interfaces, context=self._obs_spec_context
            )
            return []

    class ToltecStore(Store):
        _obs_spec_context = toltec_obs_spec_context

    class FooStore(Store):
        _obs_spec_context = ObsSpecContext(
            field_label_numbers={"interface": {f"bar{i}": i for i in range(4)}},
        )

    class Root(LmtFileStore):
        _substore_paths: ClassVar = {
            ToltecStore: PurePosixPath("toltec"),
            FooStore: PurePosixPath("foo"),
        }

    assert Root(tmp_path).glob_obs_spec(text) == []
    assert resolved == expected
