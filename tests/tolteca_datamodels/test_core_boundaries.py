"""Neutral extension APIs and assembled application entry points."""

import importlib
import subprocess
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize(
    ("name", "exports"),
    [
        ("file", {"FileAccessor"}),
        ("filepath", {"FilepathInfo"}),
        ("data_model", {"DataModelSet", "data_models"}),
        (
            "input_discovery",
            {"InputDiscovery", "InputDiscoveryConfig", "query_file_store"},
        ),
    ],
)
def test_assembled_modules_expose_only_their_entry_points(name, exports):
    module = importlib.import_module(f"tolteca_datamodels.{name}")
    assert set(module.__all__) == exports
    assert {key for key in vars(module) if not key.startswith("_")} == exports


@pytest.mark.parametrize(
    ("feature", "owners"),
    [
        (
            "file",
            {
                "accessor": {
                    "FileAccessor",
                    "FileAccessorView",
                    "FileTableModeT",
                    "SourceErrorPolicyT",
                },
                "trace": {"SourceTraceEvent", "source_trace_issues"},
            },
        ),
        (
            "file_store",
            {
                "base": {"FileStoreBase", "SubstoreRegistryT"},
                "view": {"StorePaths", "StoreView", "Substore"},
            },
        ),
        (
            "source",
            {
                "opener": {
                    "SourceOpener",
                    "source_opener",
                },
                "reader": {
                    "SourceReader",
                    "SourceReaderRules",
                },
            },
        ),
        (
            "input_discovery",
            {
                "config": {"InputDiscovery", "InputDiscoveryConfig"},
                "query": {"query_file_store"},
                "entries": {
                    "FileStoreInputDiscovery",
                    "GlobSource",
                    "InputDiscoveryEntry",
                    "SourcesInputDiscovery",
                },
            },
        ),
        (
            "obs_spec",
            {
                "base": {"ObsSpecBase"},
                "context": {"ObsSpecContext"},
                "selectors": {
                    "ColumnSelector",
                    "FieldConstraint",
                    "FieldPattern",
                    "FieldSelection",
                    "FieldValue",
                    "Selector",
                    "ValueBounds",
                },
            },
        ),
    ],
)
def test_core_features_export_their_extension_building_blocks(feature, owners):
    package = importlib.import_module(f"tolteca_datamodels.core.{feature}")
    assert set(package.__all__) == set.union(*owners.values())
    for owner, exports in owners.items():
        module = importlib.import_module(f"{package.__name__}.{owner}")
        for name in exports:
            assert getattr(package, name) is getattr(module, name)


def test_source_initializer_has_no_undeclared_runtime_exports():
    package = importlib.import_module("tolteca_datamodels.core.source")
    assert {
        name
        for name, value in vars(package).items()
        if not name.startswith("_") and not isinstance(value, ModuleType)
    } == set(package.__all__)


def test_source_declarations_do_not_import_file_or_data_model_orchestration():
    script = """
import importlib
import pkgutil
import sys
from tolteca_datamodels.core import source

for module in pkgutil.walk_packages(source.__path__, source.__name__ + "."):
    importlib.import_module(module.name)

assert not any(
    name == prefix or name.startswith(prefix + ".")
    for prefix in (
        "tolteca_datamodels.core.file", "tolteca_datamodels.core.data_model",
    )
    for name in sys.modules
)
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_data_model_lookup_tracks_registration_with_named_builtin_access():
    script = """
from tolteca_datamodels.core.data_model import data_models as registered
from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.data_model import DataModelSet, data_models
from tolteca_datamodels.lmt.data_model import data_model as lmt
from tolteca_datamodels.toltec.data_model import data_model as toltec
from tolteca_dummy_instru import register
from tolteca_dummy_instru.data_model import data_model as dummy

assert data_models.lmt is data_models["lmt"] is lmt
assert data_models.toltec is data_models["toltec"] is toltec
assert registered == data_models
assert not hasattr(registered, "register")
assert not hasattr(data_models, "register")
assert list(data_models) == ["lmt", "toltec"]
models = DataModelSet.from_select()
register()
assert data_models["dummy_instru"] is dummy
assert registered["dummy_instru"] is dummy
assert len(data_models) == 3
assert set(models.data_models) == {"lmt", "toltec"}
try:
    data_models["toltec"] = toltec
except TypeError:
    pass
else:
    raise AssertionError("Registry lookup allowed direct mutation")
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_file_access_loads_registry_only_for_generic_default_handlers():
    script = """
import sys
from tolteca_datamodels.core.file import FileAccessor
from tolteca_datamodels.core.types import DataModelInfo

class LocalDataModel(DataModelInfo):
    name = "local"

class LocalFiles(FileAccessor[LocalDataModel]):
    pass

assert "tolteca_datamodels.core.data_model" not in sys.modules
assert LocalFiles.from_sources([]).empty
assert FileAccessor.from_sources([], accessor_types=(LocalFiles,)).empty
assert FileAccessor.from_sources([], accessor_types=()).empty
assert "tolteca_datamodels.core.data_model" not in sys.modules

assert FileAccessor.from_sources([]).empty
from tolteca_datamodels.core.data_model import data_models
assert not data_models
assert "tolteca_datamodels.lmt" not in sys.modules
assert "tolteca_datamodels.toltec" not in sys.modules
assert "tolteca_dummy_instru" not in sys.modules
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_core_imports_do_not_enable_instruments():
    script = """
import importlib
import pkgutil
import sys
from tolteca_datamodels import core
from tolteca_datamodels.core.data_model import DataModelSet
from tolteca_datamodels.core.input_discovery import InputDiscoveryConfig

for module in pkgutil.walk_packages(core.__path__, core.__name__ + "."):
    importlib.import_module(module.name)

assert DataModelSet.from_select().data_models == {}
assert InputDiscoveryConfig(inputs=[]).inputs == []
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for prefix in (
        "tolteca_datamodels.lmt_base", "tolteca_datamodels.lmt",
        "tolteca_datamodels.toltec", "tolteca_dummy_instru",
        "tolteca_datamodels.file", "tolteca_datamodels.filepath",
        "tolteca_datamodels.input_discovery", "tolteca_datamodels.data_model",
    )
    for name in sys.modules
)
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_discovery_can_register_without_builtin_entries():
    script = """
import sys
from pydantic import ValidationError
from tollan.config import RuntimeContext
from tolteca_datamodels.core.file import FileAccessor
from tolteca_datamodels.core.input_discovery import (
    InputDiscovery, InputDiscoveryConfig, InputDiscoveryEntry,
)

try:
    InputDiscoveryConfig.model_validate({"inputs": [{"archive": "test"}]})
except ValidationError as error:
    assert "No input-discovery entry types are registered" in str(error)
else:
    raise AssertionError("Unregistered input was accepted")

class ArchiveInput(InputDiscoveryEntry):
    archive: str

    def discover(self, *, data_models=None, on_error="record"):
        selected = self._select_data_models(data_models)
        files = FileAccessor.partial(accessor_types=selected.file_accessor_types)
        return files.from_sources([], on_error=on_error)

InputDiscoveryConfig.register_input("archive", ArchiveInput)
settings = {"inputs": [{"archive": "test"}]}
config = InputDiscoveryConfig.model_validate(settings)
assert isinstance(config.inputs[0], ArchiveInput)
assert InputDiscoveryConfig.model_validate_json(config.model_dump_json()) == config
runtime = RuntimeContext({"input_discovery": settings})
assert InputDiscovery(runtime).discover().empty
assert "tolteca_datamodels.lmt" not in sys.modules
assert "tolteca_datamodels.toltec" not in sys.modules

# Adding the application entries preserves extensions already registered.
from tolteca_datamodels.input_discovery import InputDiscoveryConfig as Assembled
assert Assembled is InputDiscoveryConfig
combined = Assembled.model_validate({"inputs": [
    {"archive": "test"},
    {"sources": ["https://archive.example/raw.nc"], "obs_spec": "1/tel"},
]})
assert isinstance(combined.inputs[0], ArchiveInput)
assert str(combined.inputs[1].obs_spec) == "1/tel"
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_product_type_shim_uses_the_same_model_class():
    from tolteca_datamodels.core.types import DataModelsT, FrozenDataModel
    from tolteca_datamodels.types import DataModelsT as ProductDataModelsT
    from tolteca_datamodels.types import FrozenDataModel as ProductFrozenDataModel

    assert ProductFrozenDataModel is FrozenDataModel
    assert ProductDataModelsT is DataModelsT
