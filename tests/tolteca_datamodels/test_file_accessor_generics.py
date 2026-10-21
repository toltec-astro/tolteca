"""Data-model bindings and registration of file-table accessors."""

from enum import Flag
from types import new_class
from typing import ClassVar, Literal, get_type_hints

import pytest

from tolteca_datamodels.core import data_model as data_model_module
from tolteca_datamodels.core.data_model import DataModelSpec
from tolteca_datamodels.core.types import DataModelInfo
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.types import LmtDataModel
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.types import ToltecDataModel
from tolteca_dummy_instru.file import DummyInstruFileAccessor
from tolteca_dummy_instru.types import DummyInstruDataModel


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(data_model_module._data_models, "_entries", {})


@pytest.mark.parametrize(
    ("accessor", "info"),
    [
        (LmtFileAccessor, LmtDataModel),
        (ToltecFileAccessor, ToltecDataModel),
        (DummyInstruFileAccessor, DummyInstruDataModel),
    ],
)
def test_data_model_is_a_class_variable_available_on_instances(accessor, info):
    assert (
        get_type_hints(FileAccessor, localns={"Flag": Flag})["data_model_cls"]
        == ClassVar[type[DataModelInfo]]
    )
    assert accessor.data_model_cls is info
    assert accessor._data_models == frozenset([info.name])
    instance = accessor(accessor.from_sources([]))
    assert instance.data_model_cls is info
    assert "data_model_cls" not in vars(instance)


def test_class_binding_through_intermediate_classes_requires_explicit_registration():
    class Intermediate[ModelT: DataModelInfo](FileAccessor[ModelT]):
        pass

    class Further[ModelT: DataModelInfo](Intermediate[ModelT]):
        pass

    assert data_model_module._data_models == {}
    assert not hasattr(FileAccessor, "data_model_cls")
    assert not hasattr(Intermediate, "data_model_cls")
    with pytest.raises(ValueError, match="concrete"):
        DataModelSpec(file_accessor_cls=Intermediate)

    class Concrete(Further[ToltecDataModel]):
        pass

    assert Concrete.data_model_cls is ToltecDataModel
    definition = DataModelSpec(file_accessor_cls=Concrete)
    assert definition.data_model_cls is ToltecDataModel
    assert definition.name == ToltecDataModel.name
    assert data_model_module._data_models == {}
    definition.register()
    assert data_model_module._data_models == {"toltec": definition}


def test_local_specialization_keeps_scope_without_replacing_registration():
    original = DataModelSpec(file_accessor_cls=ToltecFileAccessor)
    original.register()

    class LocalFiles(ToltecFileAccessor):
        pass

    assert LocalFiles.data_model_cls is ToltecFileAccessor.data_model_cls
    assert LocalFiles._data_models == ToltecFileAccessor._data_models
    assert data_model_module._data_models == {"toltec": original}
    replacement = DataModelSpec(file_accessor_cls=LocalFiles)
    for definition in (original, replacement):
        with pytest.raises(ValueError, match="already registered: 'toltec'"):
            definition.register()
    assert data_model_module._data_models == {"toltec": original}


@pytest.mark.parametrize(
    "annotation", [str, Literal["toltec"], LmtDataModel | ToltecDataModel]
)
def test_concrete_binding_requires_one_data_model_definition(annotation):
    with pytest.raises(TypeError, match="DataModelInfo subclass"):
        new_class("Invalid", (FileAccessor[annotation],))
    assert data_model_module._data_models == {}


def test_concrete_subclass_requires_a_data_model_binding():
    with pytest.raises(TypeError, match="must bind.*DataModelT"):

        class Unbound(FileAccessor):
            pass

    assert data_model_module._data_models == {}


def test_data_model_cannot_override_the_generic_binding():
    with pytest.raises(TypeError, match="explicitly sets 'data_model_cls'"):

        class Conflicting(FileAccessor[ToltecDataModel]):
            data_model_cls = LmtDataModel

    assert data_model_module._data_models == {}


@pytest.mark.parametrize("name", [None, 1, ["toltec"]])
def test_bound_definition_requires_a_string_name(name):
    class InvalidDataModel(DataModelInfo):
        pass

    InvalidDataModel.name = name
    with pytest.raises(TypeError, match="must declare a string name"):
        new_class("Invalid", (FileAccessor[InvalidDataModel],))


def test_base_definition_is_not_a_named_data_model():
    with pytest.raises(TypeError, match="must declare a string name"):
        new_class("Invalid", (FileAccessor[DataModelInfo],))
