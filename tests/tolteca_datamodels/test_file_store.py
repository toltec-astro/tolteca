"""File-store infrastructure and header identification from cached sources."""

from pathlib import Path
from typing import ClassVar

import pytest
import xarray as xr
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.file_store import FileStoreBase
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.obs_spec import ObsSpecContext
from tolteca_datamodels.toltec.file import ToltecFileAccessor


def test_base_requires_a_directory_search(tmp_path):
    with pytest.raises(TypeError, match="abstract"):
        FileStoreBase(tmp_path)


def test_generic_store_uses_its_bound_selector_without_instrument_policy(tmp_path):
    calls = []

    class Store(FileStoreBase[ObsSpec]):
        def _iter_candidate_patterns(self, spec):
            calls.append(spec)
            return []

    store = Store(str(tmp_path / "missing"))
    assert isinstance(store.path, Path)
    assert store.path == tmp_path / "missing"
    assert store.find_candidates("1") == []
    assert calls == [ObsSpec.parse("1")]
    assert store._obs_spec_cls is ObsSpec
    assert store.substores == {}
    assert not hasattr(store, "instru_file_stores")
    assert not hasattr(store, "_get_instru_file_stores")
    with pytest.raises(AttributeError):
        store.path = tmp_path


class LocalStore(FileStoreBase[ObsSpec]):
    def _iter_candidate_patterns(self, spec):
        yield self.path, "*.txt"


def test_inline_substores_are_validated_copied_and_inherited(tmp_path):
    declarations = {"child": LocalStore}

    class Parent(LocalStore):
        _substore_classes: ClassVar = declarations

    class Derived(Parent):
        pass

    declarations["unregistered"] = LocalStore
    Derived.register_substore("added", LocalStore)
    assert list(Parent(tmp_path).substores) == ["child"]
    assert list(Derived(tmp_path).substores) == ["child", "added"]
    assert Parent(tmp_path).get_substore("child").path == tmp_path / "child"
    assert LocalStore(tmp_path).substores == {}


@pytest.mark.parametrize(
    ("name", "store_cls", "error_type", "message"),
    [
        ("../outside", LocalStore, ValueError, "Invalid substore name"),
        ("child", str, TypeError, "FileStoreBase subclass"),
    ],
)
def test_inline_substores_reject_invalid_registration(
    name, store_cls, error_type, message
):
    with pytest.raises(error_type, match=message):

        class Invalid(LocalStore):
            _substore_classes: ClassVar = {name: store_cls}


def test_recursive_discovery_preserves_order_and_defers_positions(
    tmp_path, monkeypatch
):
    class Root(LocalStore):
        pass

    class Child(LocalStore):
        pass

    class Grandchild(LocalStore):
        pass

    Child.register_substore("nested", Grandchild)
    Root.register_substore("child", Child)
    store = Root(tmp_path)
    expected = [
        tmp_path / "b.txt",
        tmp_path / "a.txt",
        tmp_path / "child/c.txt",
        tmp_path / "child/nested/d.txt",
    ]
    candidates = {
        tmp_path: expected[:2] + expected[:1],
        tmp_path / "child": expected[2:3],
        tmp_path / "child/nested": expected[3:],
    }
    calls = []

    def glob(root, pattern):
        calls.append((root, pattern))
        return iter(candidates[root])

    monkeypatch.setattr(Path, "glob", glob)
    assert store.find_candidates("[-1]") == expected
    assert calls == [(root, "*.txt") for root in candidates]


def test_registration_is_parent_specific_and_visible_to_existing_instances(tmp_path):
    class Parent(LocalStore):
        pass

    class Child(LocalStore):
        pass

    parent = Parent(tmp_path)
    Parent.register_substore("first", Child)

    class Derived(Parent):
        pass

    class Sibling(Parent):
        pass

    Derived.register_substore("second", Child)
    assert list(parent.substores) == ["first"]
    assert list(Derived(tmp_path).substores) == ["first", "second"]
    assert list(Sibling(tmp_path).substores) == ["first"]
    assert LocalStore(tmp_path).substores == {}
    assert parent.get_substore("first") is parent.substores["first"]
    assert parent.get_substore("first").path == tmp_path / "first"
    with pytest.raises(KeyError, match="missing"):
        parent.get_substore("missing")


def test_registration_has_no_instrument_reserved_names(tmp_path):
    class Parent(LocalStore):
        pass

    Parent.register_substore("toltec", LocalStore)
    Parent.register_substore("lmt", LocalStore)
    Parent.register_substore("toltec", LocalStore)
    assert list(Parent(tmp_path).substores) == ["toltec", "lmt"]
    with pytest.raises(ValueError, match="already registered"):
        Parent.register_substore("toltec", Parent)
    with pytest.raises(TypeError, match="FileStoreBase subclass"):
        Parent.register_substore("bad", str)


def test_registration_rejects_cycles_before_discovery():
    class Parent(LocalStore):
        pass

    class Child(LocalStore):
        pass

    class Grandchild(LocalStore):
        pass

    Parent.register_substore("child", Child)
    Child.register_substore("nested", Grandchild)
    with pytest.raises(ValueError, match="cycle"):
        Grandchild.register_substore("root", Parent)
    with pytest.raises(ValueError, match="cycle"):
        Parent.register_substore("self", Parent)
    assert Grandchild._substore_classes == {}


def test_context_composition_follows_the_full_hierarchy_without_io(
    tmp_path, monkeypatch
):
    class Parent(LocalStore):
        _obs_spec_context = ObsSpecContext(labels={"interface": {"parent": None}})

    class Child(LocalStore):
        _obs_spec_context = ObsSpecContext(labels={"interface": {"child": 1}})

    class Grandchild(LocalStore):
        _obs_spec_context = ObsSpecContext(labels={"interface": {"nested": 2}})

    Parent.register_substore("child", Child)
    Child.register_substore("nested", Grandchild)

    def unexpected(*args):
        pytest.fail("Context composition must not inspect files")

    monkeypatch.setattr(Path, "glob", unexpected)
    assert Parent(tmp_path).get_obs_spec_context().labels == {
        "interface": {"parent": None, "child": 1, "nested": 2}
    }

    class Conflict(LocalStore):
        _obs_spec_context = ObsSpecContext(labels={"interface": {"child": 9}})

    Grandchild.register_substore("conflict", Conflict)
    with pytest.raises(ValueError, match="Conflicting numeric interpretation"):
        Parent(tmp_path).get_obs_spec_context()


@pytest.mark.parametrize(
    "relative_path",
    [
        "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        "toltec/tcs/toltec0/toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc",
    ],
)
def test_cached_source_with_remote_location_and_open_dataset(
    data_lmt_path, tmp_path, monkeypatch, relative_path
):
    from xarray.backends.netCDF4_ import NetCDF4DataStore

    cached = tmp_path / "b7164e.nc"
    cached.symlink_to(data_lmt_path / relative_path)
    location = fileloc(f"archive:/data_lmt/{relative_path}")
    table = ToltecFileAccessor.from_sources(
        [{"source": cached, "source_loc": location}]
    )
    assert table.loc[0, "source"] is cached
    assert table.loc[0, "source_loc"] is location
    with xr.open_dataset(cached) as data:

        def unexpected(*args, **kwargs):
            pytest.fail("Borrowed data was reopened or closed")

        with monkeypatch.context() as patch:
            patch.setattr(xr, "open_dataset", unexpected)
            patch.setattr(ToltecFileAccessor, "_source_opener", xr.open_dataset)
            patch.setattr(NetCDF4DataStore, "close", unexpected)
            with table.toltec_file.open(io_objs=[data]) as opened:
                assert opened.toltec_file.source.iloc[0] is cached
                assert opened.loc[0, "source_loc"] is location
                assert opened.loc[0, "file_prefix"] == table.loc[0, "file_prefix"]
                assert len(opened.toltec_file.query("obsnum == 145609")) == 1
                assert opened.io_obj.iloc[0] is data
    assert "filepath_orig" not in table
