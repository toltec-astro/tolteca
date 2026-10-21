"""File-store infrastructure and header identification from cached sources."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ClassVar

import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.core.file_store import (
    FileStoreBase,
    Substore,
    SubstoreRegistryT,
)
from tolteca_datamodels.core.obs_spec import (
    FieldConstraint,
    ObsSpecBase,
    ObsSpecContext,
    Selector,
)
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec.file import ToltecFileAccessor


def test_base_requires_a_directory_search(tmp_path):
    with pytest.raises(TypeError, match="abstract"):
        FileStoreBase(tmp_path)


def test_generic_store_uses_its_bound_selector_without_instrument_policy(tmp_path):
    calls = []

    class Store(FileStoreBase[ObsSpec]):
        _subpaths = (PurePosixPath("."),)

        def _iter_local_glob_patterns(self, spec, subpath):
            calls.append(spec)
            return []

    store = Store(str(tmp_path / "missing"))
    assert isinstance(store.path, Path)
    assert store.path == tmp_path / "missing"
    assert store.glob_obs_spec("1") == []
    assert calls == [ObsSpec.parse("1")]
    assert store._obs_spec_cls is ObsSpec
    assert not hasattr(store, "_obs_spec_context")
    assert store.substores == {}
    assert not hasattr(store, "instru_file_stores")
    assert not hasattr(store, "_get_instru_file_stores")
    with pytest.raises(AttributeError):
        store.path = tmp_path  # ty: ignore[invalid-assignment] -- read-only property


class LocalStore(FileStoreBase[ObsSpec]):
    _subpaths = (PurePosixPath("."),)

    def _iter_local_glob_patterns(self, spec, subpath):
        yield "*.txt"


def test_no_local_locations_are_implicit(tmp_path, monkeypatch):
    class Parent(FileStoreBase[ObsSpec]):
        _substore_paths: ClassVar = {LocalStore: PurePosixPath("child")}

        def _iter_local_glob_patterns(self, spec, subpath):
            pytest.fail("A store without managed locations was searched locally")

    searched = []

    def glob(root, pattern):
        searched.append((root, pattern))
        return iter(())

    monkeypatch.setattr(Path, "glob", glob)
    store = Parent(tmp_path)
    assert store.glob_obs_spec("1") == []
    assert searched == [(tmp_path / "child", "*.txt")]


@pytest.mark.parametrize(
    ("selection", "included"),
    [("all", (0, 1)), ("parent", (0,)), ("child", (1,)), ("none", ())],
)
def test_root_and_substore_own_disjoint_files(
    tmp_path, monkeypatch, selection, included
):
    class Child(LocalStore):
        pass

    class Parent(LocalStore):
        child = Substore(Child, subpath="child")

    store = Parent(tmp_path)
    child = store.substores[Child]
    child.path.mkdir()
    paths = [store.path / "local.txt", child.path / "delegated.txt"]
    for path in paths:
        path.touch()
    searched = []
    glob = Path.glob

    def record_glob(root, glob_pattern):
        found = list(glob(root, glob_pattern))
        searched.append((root, glob_pattern, found))
        return iter(found)

    monkeypatch.setattr(Path, "glob", record_glob)
    views = {
        "all": store.view,
        "parent": store.view.local,
        "child": store.child,
        "none": store.partial(exclude=(store.view,)),
    }
    assert views[selection].glob_obs_spec("1") == [paths[i] for i in included]
    # Verify ownership before deduplication can hide duplicate discovery.
    assert searched == [(paths[i].parent, "*.txt", [paths[i]]) for i in included]


def test_declared_root_is_passed_unchanged_to_local_search(tmp_path):
    received = []
    subpaths = (PurePosixPath("."),)

    class Store(LocalStore):
        _subpaths = subpaths

        def _iter_local_glob_patterns(self, spec, subpath):
            received.append(subpath)
            yield from super()._iter_local_glob_patterns(spec, subpath)

    local = tmp_path / "local.txt"
    local.touch()
    assert Store(tmp_path).glob_obs_spec("1") == [local]
    assert Store._subpaths is subpaths
    assert received[0] is subpaths[0]


def test_declared_locations_filter_without_filesystem_discovery(tmp_path, monkeypatch):
    class Leaf(LocalStore):
        pass

    class Child(LocalStore):
        archive = Substore(Leaf, subpath="archive")

    class Root(LocalStore):
        _subpaths = (PurePosixPath("tel"), PurePosixPath("calibration"))
        child = Substore(Child, subpath="instruments/child")

    store = Root(tmp_path)
    expected_root = tmp_path / "instruments/child/archive"
    calls = []

    def unexpected(*args, **kwargs):
        pytest.fail("Logical traversal inspected the filesystem")

    def glob(root, glob_pattern):
        calls.append((root, glob_pattern))
        return iter([root / "second.txt", root / "first.txt", root / "second.txt"])

    monkeypatch.setattr(Path, "stat", unexpected)
    monkeypatch.setattr(Path, "iterdir", unexpected)
    monkeypatch.setattr(Path, "resolve", unexpected)
    monkeypatch.setattr(Path, "glob", unexpected)
    assert store.partial(exclude=(store.view,)).glob_obs_spec("1") == []
    monkeypatch.setattr(Path, "glob", glob)
    assert store.partial(Root.child.archive).glob_obs_spec("1") == [
        expected_root / "second.txt",
        expected_root / "first.txt",
    ]
    assert calls == [(expected_root, "*.txt")]


@pytest.mark.parametrize(
    ("selection", "locations"),
    [
        ("root", [".", "foo", "foo_extra"]),
        ("child", ["foo/bar", "foo/bar/baz"]),
        ("both", [".", "foo", "foo_extra", "foo/bar", "foo/bar/baz"]),
    ],
)
def test_selected_store_searches_all_its_managed_locations(
    tmp_path, monkeypatch, selection, locations
):
    received = []
    searched = []

    class RecordingStore(LocalStore):
        def _iter_local_glob_patterns(self, spec, subpath):
            received.append((self.path, subpath))
            yield from super()._iter_local_glob_patterns(spec, subpath)

    class Child(RecordingStore):
        _subpaths = (PurePosixPath("."), PurePosixPath("baz"))

    class Root(RecordingStore):
        _subpaths = tuple(map(PurePosixPath, (".", "foo", "foo_extra")))
        child = Substore(Child, subpath="foo/bar")

    def glob(root, glob_pattern):
        searched.append((root, glob_pattern))
        return iter([root / "input.txt"])

    monkeypatch.setattr(Path, "glob", glob)
    store = Root(tmp_path)
    views = {
        "root": store.view.local,
        "child": store.child,
        "both": store.view,
    }
    assert views[selection].glob_obs_spec("1") == [
        tmp_path / location / "input.txt" for location in locations
    ]
    assert searched == [(tmp_path / location, "*.txt") for location in locations]
    expected_received = {
        ".": (tmp_path, PurePosixPath(".")),
        "foo": (tmp_path, PurePosixPath("foo")),
        "foo_extra": (tmp_path, PurePosixPath("foo_extra")),
        "foo/bar": (tmp_path / "foo/bar", PurePosixPath(".")),
        "foo/bar/baz": (tmp_path / "foo/bar", PurePosixPath("baz")),
    }
    assert received == [expected_received[location] for location in locations]


def test_local_glob_receives_each_selected_managed_subpath(tmp_path, monkeypatch):
    received = []
    searched = []

    class Root(LocalStore):
        _subpaths = (PurePosixPath("tel"), PurePosixPath("calibration"))

        def _iter_local_glob_patterns(self, spec, subpath):
            received.append(subpath)
            yield f"{subpath.name}-*.txt"

    def glob(root, pattern):
        searched.append((root, pattern))
        return iter(())

    monkeypatch.setattr(Path, "glob", glob)
    assert Root(tmp_path).glob_obs_spec("1") == []
    assert received == [PurePosixPath("tel"), PurePosixPath("calibration")]
    assert searched == [
        (tmp_path / subpath, f"{subpath}-*.txt") for subpath in ("tel", "calibration")
    ]


def test_typed_search_and_direct_substore_call_agree(tmp_path, monkeypatch):
    class Child(LocalStore):
        pass

    class Root(LocalStore):
        _subpaths = ()
        instrument = Substore(Child, subpath="instrument")

    paths = [tmp_path / "instrument/input.txt"]
    calls = []

    def glob(root, pattern):
        calls.append(root)
        return iter(paths)

    monkeypatch.setattr(Path, "glob", glob)
    root = Root(tmp_path)
    assert root.partial(Root.instrument).glob_obs_spec("1") == paths
    child = root.substores[Child]
    assert child.glob_obs_spec("1") == paths
    assert calls == [child.path, child.path]


@dataclass(frozen=True)
class ChannelSpec(ObsSpecBase):
    channel: Selector = None
    _primary_field: ClassVar = "channel"
    _field_constraints: ClassVar = {"channel": FieldConstraint(kind="integer")}


@pytest.mark.parametrize("inline", [False, True])
def test_unsupported_selector_skips_local_search_not_descendants(tmp_path, inline):
    calls = []

    class Grandchild(LocalStore):
        def _iter_local_glob_patterns(self, spec, subpath):
            calls.append((self.path, spec))
            yield from super()._iter_local_glob_patterns(spec, subpath)

    class Child(FileStoreBase[ChannelSpec]):
        _subpaths = (PurePosixPath("."),)
        _substore_paths: ClassVar = {Grandchild: PurePosixPath("nested")}

        def _iter_local_glob_patterns(self, spec, subpath):
            calls.append((self.path, spec))
            yield "*.txt"

    class Parent(LocalStore):
        _substore_paths: ClassVar = {Child: PurePosixPath("child")} if inline else {}

    if not inline:
        Parent.register_substore(Child, "child")
    store = Parent(tmp_path)
    child_path = tmp_path / "child"
    nested_path = child_path / "nested"
    nested_path.mkdir(parents=True)
    paths = [
        tmp_path / "parent.txt",
        child_path / "child.txt",
        nested_path / "nested.txt",
    ]
    for path in paths:
        path.touch()

    assert isinstance(store.substores[Child], Child)
    assert store.glob_obs_spec("1") == [paths[0], paths[2]]
    assert calls == [(nested_path, ObsSpec.parse("1"))]
    calls.clear()

    spec = ChannelSpec.parse("1")
    assert store.glob_obs_spec(spec) == [paths[1]]
    assert calls == [(child_path, spec)]
    assert calls[0][1] is spec
    calls.clear()

    # Direct entry uses the child's parser, not the parent's selector layout.
    assert store.substores[Child].glob_obs_spec("1") == [paths[1]]
    assert calls == [(child_path, spec)]


def test_invalid_selector_text_and_supported_store_errors_raise(tmp_path):
    with pytest.raises(ValueError):
        LocalStore(tmp_path).glob_obs_spec("tcs-(1,2")

    class Broken(LocalStore):
        def _iter_local_glob_patterns(self, spec, subpath):
            msg = "Failed to interpret a supported selector"
            raise TypeError(msg)

    class Parent(LocalStore):
        _substore_paths: ClassVar = {Broken: PurePosixPath("broken")}

    with pytest.raises(TypeError, match="Failed to interpret"):
        Parent(tmp_path).glob_obs_spec("1")


def test_inline_substores_are_validated_copied_and_inherited(tmp_path):
    registered = []

    class Added(LocalStore):
        pass

    declarations: dict[type[FileStoreBase], PurePosixPath] = {
        LocalStore: PurePosixPath("child")
    }

    class Parent(LocalStore):
        _substore_paths: ClassVar[SubstoreRegistryT] = declarations

        @classmethod
        def register_substore(cls, store_cls, subpath, *, name=None):
            registered.append((cls, store_cls, subpath))
            super().register_substore(store_cls, subpath, name=name)

    class Derived(Parent):
        pass

    assert registered == [
        (Parent, LocalStore, PurePosixPath("child")),
        (Derived, LocalStore, PurePosixPath("child")),
    ]
    declarations[Added] = PurePosixPath("unregistered")
    Derived.register_substore(Added, "added")
    assert list(Parent(tmp_path).substores) == [LocalStore]
    assert list(Derived(tmp_path).substores) == [LocalStore, Added]
    assert declarations == {
        LocalStore: PurePosixPath("child"),
        Added: PurePosixPath("unregistered"),
    }
    assert Parent._substore_paths == {LocalStore: PurePosixPath("child")}
    assert Derived._substore_paths == {
        LocalStore: PurePosixPath("child"),
        Added: PurePosixPath("added"),
    }
    assert Parent(tmp_path).substores[LocalStore].path == tmp_path / "child"
    assert LocalStore(tmp_path).substores == {}


def test_descriptors_register_normalized_paths_and_inherit_without_parent_mutation(
    tmp_path,
):
    registered = []

    class RecordingStore(LocalStore):
        @classmethod
        def register_substore(cls, store_cls, subpath, *, name=None):
            registered.append((cls, store_cls, subpath, name))
            super().register_substore(store_cls, subpath, name=name)

    class Parent(RecordingStore):
        child = Substore(LocalStore, subpath="instruments/./child")

    class Additional(LocalStore):
        pass

    class Derived(Parent):
        additional = Substore(Additional, subpath=PurePosixPath("additional"))

    assert registered == [
        (Parent, LocalStore, "instruments/./child", "child"),
        (Derived, LocalStore, PurePosixPath("instruments/child"), None),
        (Derived, Additional, PurePosixPath("additional"), "additional"),
    ]
    assert RecordingStore._substore_paths == {}
    assert Parent._substore_paths == {LocalStore: PurePosixPath("instruments/child")}
    parent, derived = Parent(tmp_path), Derived(tmp_path)
    assert list(parent.substores) == [LocalStore]
    assert list(derived.substores) == [LocalStore, Additional]
    assert parent.child.path == parent.substores[LocalStore].path
    assert derived.child.path == tmp_path / "instruments/child"
    assert derived.additional.path == tmp_path / "additional"


@pytest.mark.parametrize("subpath", ["first", "second"])
def test_descriptor_and_registry_cannot_register_the_same_child(subpath):
    with pytest.raises(ValueError, match="already registered"):

        class Invalid(LocalStore):
            _substore_paths: ClassVar = {LocalStore: PurePosixPath("first")}
            child = Substore(LocalStore, subpath=subpath)


@pytest.mark.parametrize(
    ("subpath", "store_cls", "message"),
    [
        (PurePosixPath("../outside"), LocalStore, "relative descendant path"),
        (PurePosixPath("child"), str, "subclass of FileStoreBase"),
    ],
)
@pytest.mark.parametrize("named", [False, True])
def test_inline_substores_reject_invalid_registration(
    subpath, store_cls, message, named
):
    with pytest.raises(ValidationError, match=message):

        class Invalid(LocalStore):
            if named:
                child = Substore(store_cls, subpath=subpath)
            else:
                _substore_paths: ClassVar = {store_cls: subpath}


@pytest.mark.parametrize("inline", [False, True])
def test_nested_subpaths_are_normalized_before_class_lookup(tmp_path, inline):
    class Parent(LocalStore):
        _substore_paths: ClassVar = (
            {LocalStore: PurePosixPath("instruments/toltec")} if inline else {}
        )

    if inline:
        with pytest.raises(ValueError, match="already registered"):
            Parent.register_substore(LocalStore, "instruments/./toltec")
    else:
        Parent.register_substore(LocalStore, "instruments/./toltec")
    store = Parent(tmp_path)
    child = store.substores[LocalStore]
    assert list(store.substores) == [LocalStore]
    assert Parent._substore_paths == {LocalStore: PurePosixPath("instruments/toltec")}
    assert child.path == tmp_path / "instruments/toltec"
    assert store.substores[LocalStore] is child

    with pytest.raises(ValueError, match="already registered"):
        Parent.register_substore(LocalStore, "elsewhere")
    assert store.substores[LocalStore] is child
    assert child.path == tmp_path / "instruments/toltec"


@pytest.mark.parametrize("subpath", ["", ".", "/outside", "a/../b", "../b"])
def test_subpath_contract_applies_to_registration(tmp_path, subpath):
    class Parent(LocalStore):
        pass

    with pytest.raises(ValidationError, match="relative descendant path"):
        Parent.register_substore(LocalStore, subpath)
    store = Parent(tmp_path)
    assert store.substores == {}
    with pytest.raises(KeyError):
        store.substores[LocalStore]


def test_recursive_discovery_preserves_order_and_defers_positions(
    tmp_path, monkeypatch
):
    class Root(LocalStore):
        pass

    class Child(LocalStore):
        pass

    class Grandchild(LocalStore):
        pass

    Child.register_substore(Grandchild, "nested")
    Root.register_substore(Child, "child")
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
    assert store.glob_obs_spec("[-1]") == expected
    assert calls == [(root, "*.txt") for root in candidates]


def test_registration_is_parent_specific_and_only_affects_new_instances(tmp_path):
    class Parent(LocalStore):
        pass

    class Child(LocalStore):
        pass

    parent = Parent(tmp_path)
    Parent.register_substore(Child, "first")

    class Derived(Parent):
        pass

    class Sibling(Parent):
        pass

    class SecondChild(LocalStore):
        pass

    Derived.register_substore(SecondChild, "second")
    assert parent.substores == {}
    with pytest.raises(KeyError):
        _ = parent.substores[Child]
    parent_new = Parent(tmp_path)
    assert list(parent_new.substores) == [Child]
    assert list(Derived(tmp_path).substores) == [Child, SecondChild]
    assert list(Sibling(tmp_path).substores) == [Child]
    assert LocalStore(tmp_path).substores == {}
    assert parent_new.substores[Child].path == tmp_path / "first"
    with pytest.raises(KeyError):
        parent.substores[SecondChild]


def test_substores_are_constructed_once_without_io_and_exposed_read_only(
    tmp_path, monkeypatch
):
    constructed = []

    class Child(LocalStore):
        def __init__(self, path):
            super().__init__(path)
            constructed.append(self)

    class Leaf(Child):
        pass

    Child.register_substore(Leaf, "nested")

    class Parent(LocalStore):
        _substore_paths: ClassVar = {Child: PurePosixPath("child")}

    def unexpected(*args, **kwargs):
        pytest.fail("Store construction or lookup accessed the filesystem")

    for name in ("glob", "iterdir", "stat", "resolve"):
        monkeypatch.setattr(Path, name, unexpected)
    parent = Parent(tmp_path)
    assert len(constructed) == 2
    substores = parent.substores
    child = substores[Child]
    leaf = child.substores[Leaf]
    assert constructed == [leaf, child]
    assert leaf.path == tmp_path / "child/nested"
    assert parent.substores is substores
    assert substores[Child] is child
    assert list(substores) == [Child]
    assert list(substores.values()) == [child]
    assert constructed == [leaf, child]
    with pytest.raises(TypeError):
        substores[Child] = child  # ty: ignore[invalid-assignment] -- read-only mapping
    with pytest.raises(TypeError):
        del substores[Child]  # ty: ignore[not-subscriptable] -- read-only mapping
    with pytest.raises(AttributeError):
        parent.substores = {}  # ty: ignore[invalid-assignment] -- read-only property


def test_bound_views_retain_the_constructed_tree_after_nested_registration(tmp_path):
    class Child(LocalStore):
        pass

    class Added(LocalStore):
        pass

    class Root(LocalStore):
        child = Substore(Child, subpath="child")

    store = Root(tmp_path)
    view = store.view
    child_view = store.child
    expected = ["*.txt", "child/*.txt"]
    Child.register_substore(Added, "nested", name="added")
    Root.register_substore(Added, "added", name="added")

    assert list(store.substores) == [Child]
    assert store.substores[Child].substores == {}
    assert list(store.iter_glob_patterns("1")) == expected
    assert list(view.iter_glob_patterns("1")) == expected
    assert list(store.partial(Root.child).iter_glob_patterns("1")) == expected[1:]
    assert store.view.path == tmp_path
    assert child_view.path == tmp_path / "child"
    assert [branch.store_cls for branch in view.iter_store_views()] == [Root, Child]
    with pytest.raises(KeyError):
        _ = store.view.added
    with pytest.raises(KeyError):
        _ = store.child.added

    new = Root(tmp_path)
    assert list(new.iter_glob_patterns("1")) == [
        "*.txt",
        "child/*.txt",
        "child/nested/*.txt",
        "added/*.txt",
    ]
    assert new.view.added.path == tmp_path / "added"
    assert new.child.added.path == tmp_path / "child/nested"


@pytest.mark.parametrize("subpath", ["toltec", "lmt"])
def test_registration_has_no_instrument_reserved_names(tmp_path, subpath):
    class Parent(LocalStore):
        pass

    Parent.register_substore(LocalStore, subpath)
    with pytest.raises(ValueError, match="already registered"):
        Parent.register_substore(LocalStore, subpath)
    assert list(Parent(tmp_path).substores) == [LocalStore]
    assert Parent(tmp_path).substores[LocalStore].path == tmp_path / subpath
    with pytest.raises(ValidationError, match="subclass of FileStoreBase"):
        Parent.register_substore(str, "bad")  # ty: ignore[invalid-argument-type] -- invalid registration


def test_duplicate_registration_does_not_install_a_descriptor(tmp_path):
    class Parent(LocalStore):
        child = Substore(LocalStore, subpath="child")

    with pytest.raises(ValueError, match="already registered"):
        Parent.register_substore(LocalStore, "child", name="alias")
    assert not hasattr(Parent, "alias")
    assert list(Parent(tmp_path).substores) == [LocalStore]
    assert Parent(tmp_path).child.path == tmp_path / "child"


def test_registration_rejects_cycles_before_discovery():
    class Parent(LocalStore):
        pass

    class Child(LocalStore):
        pass

    class Grandchild(LocalStore):
        pass

    Parent.register_substore(Child, "child")
    Child.register_substore(Grandchild, "nested")
    with pytest.raises(ValueError, match="cycle"):
        Grandchild.register_substore(Parent, "root")
    with pytest.raises(ValueError, match="cycle"):
        Parent.register_substore(Parent, "self")
    assert Grandchild._substore_paths == {}


@pytest.mark.parametrize(
    ("selector", "included"),
    [("1/1", [0]), ("1/2", [1]), ("1/()[-1]", [0, 1])],
)
def test_managed_and_delegated_locations_use_local_contexts(
    tmp_path, selector, included
):
    # Synthetic mappings exercise independent stores, not acquisition conventions.
    class Store(LocalStore):
        _obs_spec_context: ClassVar[ObsSpecContext]

        def _iter_local_glob_patterns(self, spec, subpath):
            for interface in spec.resolve_values(
                "interface", ["sensor"], context=self._obs_spec_context
            ):
                yield f"{interface}.txt"

    class Child(Store):
        _obs_spec_context = ObsSpecContext(
            field_label_numbers={"interface": {"sensor": 2}}
        )

    class Parent(Store):
        _obs_spec_context = ObsSpecContext(
            field_label_numbers={"interface": {"sensor": 1}}
        )
        _substore_paths: ClassVar = {Child: PurePosixPath("child")}

    paths = [tmp_path / "sensor.txt", tmp_path / "child/sensor.txt"]
    paths[1].parent.mkdir()
    for path in paths:
        path.touch()
    assert Parent(tmp_path).glob_obs_spec(selector) == [paths[i] for i in included]


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
            patch.setattr(NetCDF4DataStore, "close", unexpected)
            with table.toltec_file.open(io_objs=[data]) as opened:
                assert opened.toltec_file.source.iloc[0] is cached
                assert opened.loc[0, "source_loc"] is location
                assert opened.loc[0, "file_prefix"] == table.loc[0, "file_prefix"]
                assert len(opened.toltec_file.query("obsnum == 145609")) == 1
                assert opened.io_obj.iloc[0] is data
    assert "filepath_orig" not in table
