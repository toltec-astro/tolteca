"""Shared observation selectors for LMT telescope and instrument stores."""

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import ClassVar, assert_type

import pytest

from tolteca_datamodels.core.file_store import FileStoreBase, StoreView
from tolteca_datamodels.core.obs_spec import ObsSpecContext
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt_base.file_store import LmtFileStoreBase
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec.file_store import ToltecFileStore
from tolteca_dummy_instru.file_store import DummyInstruFileStore


def test_base_requires_local_candidate_patterns(tmp_path):
    with pytest.raises(TypeError, match="abstract"):
        LmtFileStoreBase(tmp_path)


def test_subclass_inherits_lmt_selector_parsing(tmp_path):
    parsed = []

    class Store(LmtFileStoreBase):
        _subpaths = (PurePosixPath("."),)

        def _iter_local_glob_patterns(self, spec, subpath):
            parsed.append(spec)
            return []

    store = Store(tmp_path)
    spec = ObsSpec.parse("tcs-145609/{tel,0}")
    assert store.glob_obs_spec(str(spec)) == []
    assert parsed == [spec]
    assert store.glob_obs_spec(spec) == []
    assert parsed[-1] is spec
    assert store.substores == {}


@pytest.mark.parametrize(
    "store_cls", [LmtFileStore, ToltecFileStore, DummyInstruFileStore]
)
def test_acquisition_stores_share_lmt_selector_base(store_cls):
    assert issubclass(store_cls, LmtFileStoreBase)
    assert store_cls._obs_spec_cls is ObsSpec


def test_telescope_store_needs_no_numeric_label_interpretation():
    assert LmtFileStore._obs_spec_context.field_label_numbers == {}


def test_substores_interpret_numeric_labels_independently(tmp_path):
    class Child(LmtFileStoreBase):
        _subpaths = (PurePosixPath("."),)
        _obs_spec_context = ObsSpecContext(
            field_label_numbers={"interface": {"channel": 2}}
        )

        def _iter_local_glob_patterns(self, spec, subpath):
            if spec.resolve_values(
                "interface", ["channel"], context=self._obs_spec_context
            ):
                yield "*.txt"

    class Parent(Child):
        _substore_paths: ClassVar = {Child: PurePosixPath("child")}
        _obs_spec_context = ObsSpecContext(
            field_label_numbers={"interface": {"channel": 1}}
        )

    root_file = tmp_path / "local.txt"
    child_file = tmp_path / "child/local.txt"
    child_file.parent.mkdir()
    root_file.touch()
    child_file.touch()
    store = Parent(tmp_path)
    assert store.glob_obs_spec("1/1") == [root_file]
    assert store.glob_obs_spec("1/2") == [child_file]
    assert store.glob_obs_spec("1/channel") == [root_file, child_file]


def test_substore_lookup_and_named_toltec_property_are_typed(tmp_path):
    class Parent(LmtFileStore):
        pass

    Parent.register_substore(DummyInstruFileStore, "dummy_instru")
    store = Parent(tmp_path)
    assert_type(store.substores[DummyInstruFileStore], DummyInstruFileStore)
    assert_type(store.substores[ToltecFileStore], ToltecFileStore)
    assert isinstance(store.substores, Mapping)
    assert_type(store.toltec, StoreView)
    assert isinstance(store.substores[DummyInstruFileStore], DummyInstruFileStore)


@pytest.mark.parametrize("inline", [False, True])
def test_substore_registration_does_not_require_lmt_base(tmp_path, inline):
    calls = []

    class GenericStore(FileStoreBase[ObsSpec]):
        _subpaths = (PurePosixPath("."),)

        def _iter_local_glob_patterns(self, spec, subpath):
            calls.append((self.path, spec))
            return []

    class Parent(ToltecFileStore):
        _substore_paths: ClassVar = (
            {GenericStore: PurePosixPath("other")} if inline else {}
        )

    if not inline:
        Parent.register_substore(GenericStore, "other")

    spec = ObsSpec.parse("1")
    store = Parent(tmp_path)
    assert isinstance(store.substores[GenericStore], GenericStore)
    assert store.glob_obs_spec(spec) == []
    assert calls == [(tmp_path / "other", spec)]
    assert GenericStore not in ToltecFileStore._substore_paths
