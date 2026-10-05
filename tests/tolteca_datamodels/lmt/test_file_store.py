"""Instrument selection and delegation in the LMT acquisition tree."""

import re
from inspect import signature
from pathlib import Path
from typing import assert_type, cast

import pytest

from tolteca_datamodels.file_store import FileStoreBase
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt.obs_spec import obs_spec_context as lmt_obs_spec_context
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.obs_spec import ObsSpecContext
from tolteca_datamodels.toltec.file_store import ToltecFileStore
from tolteca_datamodels.toltec.obs_spec import (
    obs_spec_context as toltec_obs_spec_context,
)
from tolteca_datamodels.toltec.types import ToltecInfo


@pytest.mark.parametrize(
    ("pattern", "names"),
    [
        (".*", ["toltec", "rsr", "sequoia"]),
        ("toltec", ["toltec"]),
        ("toltec|rsr", ["toltec", "rsr"]),
        ("(?!rsr)", ["toltec", "sequoia"]),
        ("^(toltec|rsr)$", ["toltec", "rsr"]),
        ("tol", ["toltec"]),
        ("tec", []),
        ("tcs|toltec0|tel", []),
        ("TOLTEC", []),
        (re.compile("(?i)toltec"), ["toltec"]),
    ],
)
def test_regex_matches_only_registered_instrument_names(
    tmp_path, monkeypatch, pattern, names
):
    calls = []

    # Stand-ins exercise registry delegation, not unimplemented acquisition layouts.
    class Store(FileStoreBase[ObsSpec]):
        def _iter_candidate_patterns(self, spec):
            calls.append((self.path, spec))
            return []

    monkeypatch.setattr(
        LmtFileStore,
        "substores",
        property(
            lambda self: {
                name: Store(self.path / name) for name in ("toltec", "rsr", "sequoia")
            }
        ),
    )
    spec = ObsSpec.parse("tcs-145609")
    store = LmtFileStore(tmp_path)
    assert store.find_candidates(spec, instru=pattern) == []
    assert [path for path, _ in calls] == [tmp_path / name for name in names]
    assert all(delegated is spec for _, delegated in calls)


@pytest.mark.parametrize(
    "selector", ["145609", "145609/*", "145609/tel", "145609/{tel,0}"]
)
def test_excluded_instrument_is_not_searched_but_telescope_remains(
    data_lmt_path, monkeypatch, selector
):
    def unexpected(*args, **kwargs):
        pytest.fail("An excluded instrument was searched")

    monkeypatch.setattr(ToltecFileStore, "find_candidates", unexpected)
    store = LmtFileStore(data_lmt_path)
    paths = store.find_candidates(selector, instru="(?!toltec)")
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
    paths = store.find_candidates(selector)
    assert searched == [store.tel_path] + [store.toltec.path] * len(
        ToltecInfo.interfaces
    )
    assert len(paths) == 4
    assert sum(path.parent == store.tel_path for path in paths) == 2


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
        assert path != store.tel_path, "Excluded telescope directory was probed"
        searched.append(path)
        return glob(path, pattern)

    monkeypatch.setattr(Path, "glob", guarded_glob)
    store.find_candidates(selector)
    assert searched
    assert set(searched) == {store.toltec.path}


@pytest.mark.parametrize("selector", ["145609/tel", "145609/{tel}"])
def test_telescope_only_does_not_probe_instrument_stores(
    data_lmt_path, monkeypatch, selector
):
    store = LmtFileStore(data_lmt_path)
    glob = Path.glob

    def guarded_glob(path, pattern):
        assert path == store.tel_path, "Telescope-only selection probed an instrument"
        return glob(path, pattern)

    monkeypatch.setattr(Path, "glob", guarded_glob)
    paths = store.find_candidates(selector)
    assert len(paths) == 2
    assert all(path.parent == store.tel_path for path in paths)


def test_default_pattern_preserves_instrument_discovery(data_lmt_path):
    store = LmtFileStore(data_lmt_path)
    assert store.find_candidates("145609/0") == store.toltec.find_candidates("145609/0")


def test_candidates_keep_discovery_order_and_first_occurrence(
    data_lmt_path, monkeypatch
):
    store = LmtFileStore(data_lmt_path)
    telescope = sorted(store.tel_path.glob("*145609*.nc"), reverse=True)
    instrument = sorted(store.toltec.path.glob("*/*/*145609*.nc"), reverse=True)
    assert len(telescope) == len(instrument) == 2

    # Repeated candidates exercise deduplication independently of traversal order.
    candidates = {store.tel_path: telescope, store.toltec.path: instrument}

    def glob(path, pattern):
        matches = candidates[path]
        return iter(matches + matches[:1])

    monkeypatch.setattr(Path, "glob", glob)
    assert store.find_candidates("145609") == telescope + instrument


def test_typed_child_is_the_delegated_instance(tmp_path):
    store = LmtFileStore(tmp_path)
    assert_type(store.path, Path)
    assert_type(store.toltec, ToltecFileStore)
    assert isinstance(store, FileStoreBase)
    assert isinstance(store.toltec, FileStoreBase)
    assert isinstance(store.toltec, ToltecFileStore)
    assert store.toltec is store.toltec
    assert store.toltec is store.substores["toltec"]
    assert store.toltec.path == tmp_path / "toltec"
    assert store.toltec.substores == {}
    LmtFileStore.register_substore("toltec", ToltecFileStore)
    assert store.get_substore("toltec") is store.toltec


def test_instrument_filter_keeps_descendants_of_the_selected_child(
    tmp_path, monkeypatch
):
    class Nested(FileStoreBase[ObsSpec]):
        _obs_spec_context = ObsSpecContext(labels={"interface": {"nested1": 1}})

        def _iter_candidate_patterns(self, spec):
            yield self.path, "*.nc"

    class Instrument(FileStoreBase[ObsSpec]):
        def _iter_candidate_patterns(self, spec):
            return ()

    class Root(LmtFileStore):
        pass

    Instrument.register_substore("nested", Nested)
    Root.register_substore("instrument", Instrument)
    path = tmp_path / "instrument/nested/example.nc"
    calls = []

    def glob(root, pattern):
        calls.append(root)
        return iter([path] if root == path.parent else [])

    monkeypatch.setattr(Path, "glob", glob)
    root = Root(tmp_path)
    assert root.find_candidates("1", instru="^instrument$") == [path]
    assert calls == [root.tel_path, path.parent]
    assert root.get_obs_spec_context(instru="^instrument$") == (
        lmt_obs_spec_context | Nested._obs_spec_context
    )


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


def test_cached_child_remains_consistent_with_read_only_path(tmp_path):
    store = LmtFileStore(tmp_path)
    child = store.toltec
    with pytest.raises(AttributeError):
        store.path = tmp_path / "other"  # ty: ignore[invalid-assignment] -- test runtime protection
    assert store.toltec is child
    assert store.substores["toltec"] is child
    assert child.path == store.path / "toltec"


def test_stores_retain_identity_equality(tmp_path):
    first, second = LmtFileStore(tmp_path), LmtFileStore(tmp_path)
    assert first != second
    stores = {first, second}
    assert first.toltec is first.toltec
    assert len(stores) == 2
    assert first in stores


def test_only_lmt_search_accepts_instrument_filter():
    assert set(signature(FileStoreBase.find_candidates).parameters) == {"self", "spec"}
    assert set(signature(ToltecFileStore.find_candidates).parameters) == {
        "self",
        "spec",
    }
    assert set(signature(LmtFileStore.find_candidates).parameters) == {
        "self",
        "spec",
        "instru",
    }


def test_invalid_instrument_regex_fails_before_discovery(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid filter reached directory discovery")

    monkeypatch.setattr(Path, "glob", unexpected)
    with pytest.raises(re.PatternError):
        LmtFileStore(tmp_path).find_candidates("145609/{tel,0}", instru="[")


def test_instrument_filter_is_per_search(data_lmt_path):
    store = LmtFileStore(data_lmt_path)
    expected = store.find_candidates("145609/{tel,0}")
    telescope = store.find_candidates("145609/{tel,0}", instru="(?!toltec)")
    assert len(telescope) == 2
    assert all(path.parent == store.tel_path for path in telescope)
    assert expected == telescope + store.toltec.find_candidates("145609/{tel,0}")
    assert len(expected) > len(telescope)
    assert store.find_candidates("145609/{tel,0}") == expected
    assert not hasattr(store, "instru")


def test_additional_typed_child_uses_shared_discovery(tmp_path):
    calls = []

    # A new instrument contributes its own layout without changing discovery.
    class Store(FileStoreBase[ObsSpec]):
        def _iter_candidate_patterns(self, spec):
            calls.append((self.path, spec))
            return []

    class ExtendedLmtFileStore(LmtFileStore):
        @property
        def rsr(self) -> Store:
            return cast("Store", self.get_substore("rsr"))

    ExtendedLmtFileStore.register_substore("rsr", Store)

    store = ExtendedLmtFileStore(tmp_path)
    assert_type(store.rsr, Store)
    assert store.rsr is store.substores["rsr"]
    assert store.toltec is store.substores["toltec"]
    spec = ObsSpec.parse("tcs-145609")
    assert store.find_candidates(spec, instru="rsr") == []
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
    tmp_path, monkeypatch, text, expected
):
    resolved = {}

    # Synthetic declarations test the extension point, not acquisition filenames.
    class Store(FileStoreBase[ObsSpec]):
        def _iter_candidate_patterns(self, spec):
            interfaces = list(self._obs_spec_context.labels["interface"])
            resolved[self.path.name] = spec.resolve_values(
                "interface", interfaces, context=self._obs_spec_context
            )
            return []

    class ToltecStore(Store):
        _obs_spec_context = toltec_obs_spec_context

    class FooStore(Store):
        _obs_spec_context = ObsSpecContext(
            labels={"interface": {f"bar{i}": i for i in range(4)}},
        )

    monkeypatch.setattr(
        LmtFileStore,
        "substores",
        property(
            lambda self: {
                "toltec": ToltecStore(self.path / "toltec"),
                "foo": FooStore(self.path / "foo"),
            }
        ),
    )
    assert LmtFileStore(tmp_path).find_candidates(text) == []
    assert resolved == expected


def test_selector_contexts_follow_instrument_scope(tmp_path, monkeypatch):
    class Store(FileStoreBase[ObsSpec]):
        _obs_spec_context = ObsSpecContext(labels={"interface": {"bar1": 1}})

        def _iter_candidate_patterns(self, spec):
            pytest.fail("Context composition must not discover files")

    original = LmtFileStore.substores
    monkeypatch.setattr(
        LmtFileStore,
        "substores",
        property(
            lambda self: dict(original.fget(self)) | {"foo": Store(self.path / "foo")}
        ),
    )
    store = LmtFileStore(tmp_path)
    assert (
        store.get_obs_spec_context(instru="foo")
        == lmt_obs_spec_context | Store._obs_spec_context
    )
    assert store.get_obs_spec_context(instru="no_matches") is lmt_obs_spec_context
    assert (
        store.get_obs_spec_context(instru="toltec")
        == lmt_obs_spec_context | toltec_obs_spec_context
    )
