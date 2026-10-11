"""Shared observation selectors for LMT telescope and instrument stores."""

from collections.abc import Mapping
from typing import ClassVar, assert_type

import pytest

from tolteca_datamodels.file_store import FileStoreBase
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
        def _iter_candidate_patterns(self, spec):
            parsed.append(spec)
            return []

    store = Store(tmp_path)
    spec = ObsSpec.parse("tcs-145609/{tel,0}")
    assert store.find_candidates(str(spec)) == []
    assert parsed == [spec]
    assert store.find_candidates(spec) == []
    assert parsed[-1] is spec
    assert store.substores == {}


@pytest.mark.parametrize(
    "store_cls", [LmtFileStore, ToltecFileStore, DummyInstruFileStore]
)
def test_acquisition_stores_share_lmt_selector_base(store_cls):
    assert issubclass(store_cls, LmtFileStoreBase)
    assert store_cls._obs_spec_cls is ObsSpec


def test_substore_lookup_and_named_toltec_property_are_typed(tmp_path):
    class Parent(LmtFileStore):
        pass

    Parent.register_substore(DummyInstruFileStore, "dummy_instru")
    store = Parent(tmp_path)
    assert_type(store.get_substore(DummyInstruFileStore), DummyInstruFileStore)
    assert_type(store.get_substore(ToltecFileStore), ToltecFileStore)
    assert_type(store.substores, Mapping[type[FileStoreBase], FileStoreBase])
    assert_type(store._get_instru_file_stores(".*"), list[FileStoreBase])
    assert_type(store.toltec, ToltecFileStore)
    assert (
        store.get_substore(DummyInstruFileStore)
        is store.substores[DummyInstruFileStore]
    )
    assert isinstance(store.get_substore(DummyInstruFileStore), DummyInstruFileStore)
    assert store._get_instru_file_stores("dummy_instru") == [
        store.get_substore(DummyInstruFileStore)
    ]


@pytest.mark.parametrize("inline", [False, True])
def test_substore_registration_does_not_require_lmt_base(tmp_path, inline):
    calls = []

    class GenericStore(FileStoreBase[ObsSpec]):
        def _iter_candidate_patterns(self, spec):
            calls.append((self.path, spec))
            return []

    class Parent(ToltecFileStore):
        _substore_paths: ClassVar = {GenericStore: "other"} if inline else {}

    if not inline:
        Parent.register_substore(GenericStore, "other")

    spec = ObsSpec.parse("1")
    store = Parent(tmp_path)
    assert isinstance(store.get_substore(GenericStore), GenericStore)
    assert store.find_candidates(spec) == []
    assert calls == [(tmp_path / "other", spec)]
    assert GenericStore not in ToltecFileStore._substore_paths
