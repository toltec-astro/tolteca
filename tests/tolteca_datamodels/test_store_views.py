"""Structured selections over declared file-store locations."""

from pathlib import Path, PurePosixPath
from typing import ClassVar

import pytest

from tolteca_datamodels.core.file_store import StorePaths
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.toltec.file_store import ToltecFileStore


def test_class_and_instance_selections_use_the_same_store(tmp_path):
    store = LmtFileStore(tmp_path)
    S = type(store)
    expected = list(store.tel.iter_glob_patterns("1/{tel,7}")) + list(
        store.toltec.iter_glob_patterns("1/{tel,7}")
    )
    for view in (
        store.partial(S.tel, S.toltec),
        store.partial(store.tel, store.toltec),
        store.partial("tel", "toltec"),
    ):
        assert view.path == store.path
        assert list(view.iter_glob_patterns("1/{tel,7}")) == expected
    with pytest.raises(ValueError, match="unbound"):
        _ = S.tel.path
    with pytest.raises(ValueError, match="Bind"):
        S.tel.glob_obs_spec("1")


def test_partial_union_exclusion_and_chaining_do_not_mutate_store(tmp_path):
    store = LmtFileStore(tmp_path)
    S = type(store)
    expected = list(store.iter_glob_patterns("*"))
    selected = store.partial(S.tel, S.toltec)
    assert list(selected.iter_glob_patterns("*")) == expected
    without_tel = selected.partial(exclude=(S.tel,))
    assert list(without_tel.iter_glob_patterns("1")) == list(
        store.toltec.iter_glob_patterns("1")
    )
    assert without_tel.partial(S.tel).glob_obs_spec("1") == []
    assert without_tel.tel.glob_obs_spec("1") == []
    assert list(selected.iter_glob_patterns("*")) == expected
    assert list(store.iter_glob_patterns("*")) == expected
    assert list(selected.partial(exclude=()).iter_glob_patterns("*")) == expected


@pytest.mark.parametrize("use_view", [False, True])
def test_partial_requires_exactly_one_selection_form(tmp_path, use_view):
    store = LmtFileStore(tmp_path)
    target = store.view if use_view else store
    with pytest.raises(ValueError, match="exactly one"):
        target.partial()
    with pytest.raises(ValueError, match="exactly one"):
        target.partial("tel", exclude=())
    with pytest.raises(ValueError, match="exactly one"):
        target.partial("tel", store_types=(LmtFileStore,))
    with pytest.raises(ValueError, match="exactly one"):
        target.partial(exclude=(), store_types=(LmtFileStore,))


def test_store_types_select_local_ownership_without_enabling_children(tmp_path):
    store = LmtFileStore(tmp_path)
    tel = store.partial(store_types=(LmtFileStore,))
    assert list(tel.iter_glob_patterns("*")) == list(store.tel.iter_glob_patterns("*"))
    toltec = store.partial(store_types=(ToltecFileStore,))
    assert list(toltec.iter_glob_patterns("*")) == list(
        store.toltec.iter_glob_patterns("*")
    )
    assert (
        list(tel.partial(store_types=(ToltecFileStore,)).iter_glob_patterns("*")) == []
    )
    assert list(store.partial(store_types=()).iter_glob_patterns("*")) == []
    assert list(
        store.partial("tel")
        .partial(store_types=(LmtFileStore, ToltecFileStore))
        .iter_glob_patterns("*")
    ) == list(tel.iter_glob_patterns("*"))


def test_invalid_names_and_foreign_roots_are_not_selections(tmp_path):
    store = LmtFileStore(tmp_path)
    for name in ("no_such_path", "path", "tcs/toltec7", "_subpaths"):
        with pytest.raises(ValueError, match="Unknown store selection"):
            store.partial(name)
    with pytest.raises(ValueError, match="incompatible"):
        store.partial(ToltecFileStore(tmp_path).view)
    with pytest.raises(ValueError, match="another store"):
        store.partial(LmtFileStore(tmp_path).tel)
    with pytest.raises(TypeError, match="selection"):
        store.partial(7)  # ty: ignore[invalid-argument-type] -- test invalid public input
    with pytest.raises(AttributeError):
        _ = store.view.no_such_path


def test_patterns_need_no_filesystem_access_and_use_entry_root(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Pattern construction accessed the filesystem")

    for name in ("glob", "resolve", "stat", "iterdir"):
        monkeypatch.setattr(Path, name, unexpected)
    store = LmtFileStore(tmp_path)
    view = store.partial(LmtFileStore.tel, LmtFileStore.toltec)
    assert view.root_path == tmp_path
    assert list(view.iter_glob_patterns("tcs-1/{tel,7}")) == [
        "tel/tel_*_000001_*.nc",
        "toltec/tcs/toltec7/toltec7_000001_*.nc",
    ]
    assert store.tel.path == tmp_path / "tel"
    assert store.toltec.path == tmp_path / "toltec"
    assert list(store.toltec.iter_glob_patterns("1/toltec7")) == [
        "toltec/*/toltec7/toltec7_000001_*.nc"
    ]


def test_toltec_search_does_not_expand_globs_per_master(tmp_path, monkeypatch):
    calls = []

    def glob(root, pattern):
        calls.append((root, pattern))
        return iter(())

    monkeypatch.setattr(Path, "glob", glob)
    store = ToltecFileStore(tmp_path)
    store.glob_obs_spec("1/toltec7")
    assert calls == [(tmp_path, "*/toltec7/toltec7_000001_*.nc")]
    calls.clear()
    store.glob_obs_spec("tcs-1/toltec7")
    assert calls == [(tmp_path, "tcs/toltec7/toltec7_000001_*.nc")]


def test_candidate_paths_keep_symlinked_directory_spelling(tmp_path):
    target = tmp_path / "archive"
    target.mkdir()
    (target / "toltec7_000001_extra.nc").touch()
    root = tmp_path / "toltec"
    (root / "tcs").mkdir(parents=True)
    (root / "tcs/toltec7").symlink_to(target, target_is_directory=True)
    store = ToltecFileStore(root)
    assert store.glob_obs_spec("tcs-1/toltec7") == [
        root / "tcs/toltec7/toltec7_000001_extra.nc"
    ]


def test_named_registration_rejects_collisions_without_changing_registry(tmp_path):
    class Root(LmtFileStore):
        _substore_paths: ClassVar = {}

    for name in ("tel", "path", "bad/name", "_private"):
        with pytest.raises(ValueError, match="descriptor name"):
            Root.register_substore(ToltecFileStore, "instrument", name=name)
        assert Root(tmp_path).substores == {}
    Root.register_substore(ToltecFileStore, "instrument", name="readouts")
    with pytest.raises(ValueError, match="already registered"):
        Root.register_substore(ToltecFileStore, "instrument", name="readouts")
    assert Root(tmp_path).partial("readouts").readouts.path == tmp_path / "instrument"


def test_root_and_overlapping_declared_paths_can_be_selected_independently(tmp_path):
    class Store(LmtFileStore):
        _substore_paths: ClassVar = {}
        _subpaths = (PurePosixPath("."), PurePosixPath("foo"))
        root_files = StorePaths((PurePosixPath("."),))
        foo = StorePaths((PurePosixPath("foo"),))

        def _iter_local_glob_patterns(self, spec, subpath):
            yield "*.nc"

    (tmp_path / "root.nc").touch()
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo/nested.nc").touch()
    store = Store(tmp_path)
    assert store.partial("root_files").glob_obs_spec("*") == [tmp_path / "root.nc"]
    assert store.partial("foo").glob_obs_spec("*") == [tmp_path / "foo/nested.nc"]
    assert list(store.iter_glob_patterns("*")) == ["*.nc", "foo/*.nc"]
    with pytest.raises(ValueError, match="complete directory"):
        _ = store.partial("root_files").path
