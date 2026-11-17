"""Selected data models supply stores and handlers, not filename registrations."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import Field, ValidationError

from tolteca_datamodels import data_model as data_model_module
from tolteca_datamodels.core.data_model import DataModelSet as CoreDataModelSet
from tolteca_datamodels.core.data_model import DataModelSpec
from tolteca_datamodels.core.file_store import (
    FileStoreBase,  # noqa: TC001 -- Pydantic needs the local model's annotation at runtime.
)
from tolteca_datamodels.core.types import FrozenDataModel
from tolteca_datamodels.data_model import DataModelSet, data_models
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt.data_model import data_model_spec as lmt_data_model_spec
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt.types import LmtDataModel
from tolteca_datamodels.toltec.data_model import (
    data_model_spec as toltec_data_model_spec,
)
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.file_store import ToltecFileStore
from tolteca_datamodels.toltec.types import ToltecDataModel
from tolteca_dummy_instru.data_model import data_model_spec as dummy_data_model_spec
from tolteca_dummy_instru.file import DummyInstruFileAccessor
from tolteca_dummy_instru.file_store import DummyInstruFileStore
from tolteca_dummy_instru.types import DummyInstruDataModel


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(data_model_module._data_models, "_entries", {})
    for model in (lmt_data_model_spec, toltec_data_model_spec, dummy_data_model_spec):
        data_models.register(model)


@pytest.mark.parametrize(
    ("info", "name", "accessor"),
    [
        (LmtDataModel, "lmt", LmtFileAccessor),
        (ToltecDataModel, "toltec", ToltecFileAccessor),
        (DummyInstruDataModel, "dummy_instru", DummyInstruFileAccessor),
    ],
)
def test_declared_names_select_handlers(info, name, accessor):
    assert info.name == name
    assert (
        DataModelSet.from_select([info.name]).data_models[name].file_accessor_cls
        is accessor
    )


def test_set_unions_store_requirements_without_enabling_companion_handlers():
    data_models = DataModelSet.from_select(["toltec", "dummy_instru"])
    assert data_models.file_store_types == (
        *lmt_data_model_spec.file_store_types,
        ToltecFileStore,
        DummyInstruFileStore,
    )
    assert set(data_models.data_models) == {"toltec", "dummy_instru"}
    assert DataModelSet.from_select([]).file_store_types == ()
    lmt_models = DataModelSet.from_select(["lmt"])
    assert lmt_models.file_store_types == lmt_data_model_spec.file_store_types
    assert issubclass(LmtFileStore, lmt_models.file_store_types)


def test_store_requirements_validate_class_type():
    with pytest.raises(ValidationError, match="subclass of FileStoreBase"):
        DataModelSpec.model_validate(
            {
                "info_cls": ToltecDataModel,
                "file_accessor_cls": ToltecFileAccessor,
                "file_store_types": (str,),
            }
        )


def test_spec_is_frozen_and_requires_declared_container_types():
    spec = DataModelSpec(
        info_cls=ToltecDataModel,
        file_accessor_cls=ToltecFileAccessor,
    )
    assert isinstance(spec, FrozenDataModel)
    assert spec.file_store_types == ()
    with pytest.raises(ValidationError, match="frozen_instance"):
        setattr(spec, "file_store_types", (ToltecFileStore,))  # noqa: B010 -- Test frozen assignment.
    with pytest.raises(ValidationError, match="tuple_type"):
        DataModelSpec.model_validate(
            {
                "info_cls": ToltecDataModel,
                "file_accessor_cls": ToltecFileAccessor,
                "file_store_types": [ToltecFileStore],
            }
        )


def test_spec_validates_default_store_requirements():
    class InvalidDefault(DataModelSpec):
        file_store_types: tuple[type[FileStoreBase], ...] = Field(default=(str,))  # ty: ignore[invalid-assignment] -- Deliberately invalid default.

    with pytest.raises(ValidationError, match="subclass of FileStoreBase"):
        InvalidDefault(info_cls=ToltecDataModel, file_accessor_cls=ToltecFileAccessor)


def test_spec_owns_identity_independently_of_accessor_tags():
    assert toltec_data_model_spec.info_cls is ToltecDataModel
    with pytest.raises(ValidationError, match="info_cls"):
        DataModelSpec.model_validate({"file_accessor_cls": ToltecFileAccessor})
    telescope_support = DataModelSpec(
        info_cls=ToltecDataModel, file_accessor_cls=LmtFileAccessor
    )
    assert telescope_support.info_cls is ToltecDataModel
    assert telescope_support.name == "toltec"
    assert telescope_support.file_accessor_cls is LmtFileAccessor
    unbound = DataModelSpec(info_cls=LmtDataModel, file_accessor_cls=FileAccessor)
    assert unbound.file_accessor_cls is FileAccessor


@pytest.mark.parametrize("set_cls", [CoreDataModelSet, DataModelSet])
def test_selection_is_read_only_and_keeps_validated_specs(
    set_cls: type[CoreDataModelSet],
) -> None:
    supplied = {"toltec": toltec_data_model_spec}
    models = set_cls.from_specs(supplied)
    supplied.clear()
    assert models.data_models == {"toltec": toltec_data_model_spec}
    assert models.data_models["toltec"] is toltec_data_model_spec
    with pytest.raises(TypeError):
        models.data_models["lmt"] = lmt_data_model_spec
    with pytest.raises(TypeError):
        del models.data_models["toltec"]
    with pytest.raises(FrozenInstanceError):
        setattr(models, "data_models", {})  # noqa: B010 -- Test frozen assignment.


def test_capability_tuples_are_prepared_and_keep_spec_associations():
    telescope_support = DataModelSpec(
        info_cls=ToltecDataModel,
        file_accessor_cls=LmtFileAccessor,
        file_store_types=(LmtFileStore, ToltecFileStore),
    )
    models = CoreDataModelSet.from_specs(
        {"toltec": telescope_support, "lmt": lmt_data_model_spec}
    )
    assert models.file_accessor_types == (LmtFileAccessor,)
    assert models.file_store_types == (LmtFileStore, ToltecFileStore)
    assert models.file_accessor_types is models.file_accessor_types
    assert models.file_store_types is models.file_store_types
    assert tuple(models.data_models) == ("toltec", "lmt")
    assert models.data_models["toltec"] is telescope_support
    assert models.data_models["lmt"] is lmt_data_model_spec


def test_core_set_is_explicit_and_narrowing_preserves_the_entry_point_type():
    from tollan.config.types import StringSelector

    supplied = {"toltec": toltec_data_model_spec}
    core = CoreDataModelSet.from_specs(supplied)
    supplied["lmt"] = lmt_data_model_spec
    selector = StringSelector.model_validate(["lmt", "toltec"])
    assert core.select(selector).data_models == {"toltec": toltec_data_model_spec}
    assert type(core.select(selector)) is CoreDataModelSet
    original = DataModelSet.from_select()
    narrowed = original.select(selector)
    assert type(narrowed) is DataModelSet
    assert narrowed is not original
    assert tuple(narrowed.data_models) == ("lmt", "toltec")
    assert narrowed.file_accessor_types == (LmtFileAccessor, ToltecFileAccessor)
    assert narrowed.file_store_types == tuple(
        dict.fromkeys(
            (
                *lmt_data_model_spec.file_store_types,
                *toltec_data_model_spec.file_store_types,
            )
        )
    )
    assert "dummy_instru" in original.data_models
    empty = DataModelSet.from_select([])
    assert empty.file_accessor_types == empty.file_store_types == ()


def test_duplicate_registration_rejects_builtin_and_identical_extension():
    for spec in (toltec_data_model_spec, dummy_data_model_spec):
        with pytest.raises(ValueError, match="already registered"):
            data_models.register(spec)
        assert data_model_module.data_models[spec.name] is spec


def test_invalid_registration_leaves_lookup_unchanged():
    previous = dict(data_models)
    with pytest.raises(ValidationError, match="info_cls"):
        data_models.register({"file_accessor_cls": ToltecFileAccessor})  # ty: ignore[invalid-argument-type] -- Test invalid registration.
    assert dict(data_models) == previous


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (["toltec"], (ToltecFileAccessor,)),
        ({"excludes": ["lmt"]}, (ToltecFileAccessor, DummyInstruFileAccessor)),
        ("lmt|toltec", (LmtFileAccessor, ToltecFileAccessor)),
        (["not_registered"], ()),
        ([], ()),
    ],
)
def test_set_selects_registered_names(selection, expected):
    assert (
        tuple(
            model.file_accessor_cls
            for model in DataModelSet.from_select(selection).data_models.values()
        )
        == expected
    )


def test_set_keeps_selected_handlers_when_new_support_is_registered(monkeypatch):
    monkeypatch.setattr(data_model_module._data_models, "_entries", {})
    for model in (lmt_data_model_spec, toltec_data_model_spec):
        data_models.register(model)
    selected = DataModelSet.from_select()
    data_models.register(dummy_data_model_spec)
    assert set(selected.data_models) == {"lmt", "toltec"}
    assert set(DataModelSet.from_select().data_models) == {
        "lmt",
        "toltec",
        "dummy_instru",
    }


def test_explicit_set_can_use_an_unregistered_definition():
    class LocalFiles(ToltecFileAccessor):
        pass

    definition = DataModelSpec(info_cls=ToltecDataModel, file_accessor_cls=LocalFiles)
    models = DataModelSet.from_specs({"toltec": definition})
    assert models.file_accessor_types == (LocalFiles,)
    file_accessor = models.file_accessor_types[0].from_sources(
        ["toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"]
    )
    assert file_accessor.table.data_kind_toltec.notna().all()
    assert "data_kind_dummy_instru" not in file_accessor.table
    assert (
        DataModelSet.from_select(["toltec"]).data_models["toltec"].file_accessor_cls
        is ToltecFileAccessor
    )
