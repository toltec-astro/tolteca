"""Import boundaries for filename machinery, models and file accessors."""

import importlib
import importlib.util
import subprocess
import sys

import pytest


def _run_each(script, first_imports):
    # One interpreter per import order; the interpreters run concurrently.
    processes = {
        first: subprocess.Popen(  # noqa: S603 -- current interpreter and literal test code
            [sys.executable, "-c", script, first],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for first in first_imports
    }
    failures = {}
    for first, process in processes.items():
        _, stderr = process.communicate(timeout=60)
        if process.returncode:
            failures[first] = stderr
    assert not failures, failures


@pytest.mark.parametrize(
    ("name", "class_name"),
    [
        ("tolteca_datamodels.lmt.file_store", "LmtFileStore"),
        ("tolteca_datamodels.toltec.file_store", "ToltecFileStore"),
        ("tolteca_datamodels.core.file_store", "FileStoreBase"),
        ("tolteca_datamodels.lmt_base.file_store", "LmtFileStoreBase"),
    ],
)
def test_file_store_module_names(name, class_name):
    module = importlib.import_module(name)
    assert getattr(module, class_name).__name__ == class_name


def test_file_store_infrastructure_does_not_load_domain_modules():
    script = """
import sys
from tolteca_datamodels.core.file_store import FileStoreBase

assert not any(
    name == prefix or name.startswith(prefix + ".")
    for prefix in (
        "tolteca_datamodels.lmt_base", "tolteca_datamodels.lmt",
        "tolteca_datamodels.toltec",
    )
    for name in sys.modules
)
assert "tolteca_datamodels.file" not in sys.modules
assert "tolteca_datamodels.input_discovery" not in sys.modules
assert "tolteca_datamodels.core.input_discovery.entries" not in sys.modules
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


_STORE_FIRST_IMPORTS = ["lmt.file_store", "toltec.file_store", "toltec", "data_model"]


def test_public_store_assembles_builtin_requirements_in_any_import_order():
    script = """
import importlib
import sys

importlib.import_module("tolteca_datamodels." + sys.argv[1])
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.toltec.file_store import ToltecFileStore
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.lmt.data_model import data_model_spec as lmt_data_model_spec

store = LmtFileStore("/data_lmt")
assert LmtFileStore.__module__ == "tolteca_datamodels.lmt.file_store"
assert "toltec" in vars(LmtFileStore)
assert store.toltec.path == store.substores[ToltecFileStore].path
assert list(store.toltec.iter_glob_patterns("tcs-1/toltec7")) == [
    "toltec/tcs/toltec7/toltec7_000001_*.nc"
]
assert DataModelSet.from_select(["toltec"]).file_store_types == (
    *lmt_data_model_spec.file_store_types, ToltecFileStore,
)
assert issubclass(LmtFileStore, lmt_data_model_spec.file_store_types)
assert store.partial(
    store_types=DataModelSet.from_select(["lmt"]).file_store_types
).path == store.tel.path
selected = store.partial(
    store_types=DataModelSet.from_select(["toltec"]).file_store_types
)
assert list(selected.iter_glob_patterns("1/{tel,7}")) == [
    "tel/tel_*_000001_*.nc", "toltec/*/toltec7/toltec7_000001_*.nc"
]
assert "tolteca_dummy_instru" not in sys.modules
"""
    _run_each(script, _STORE_FIRST_IMPORTS)


_SHARED_MODULES = (
    "tolteca_datamodels.core.filepath",
    "tolteca_datamodels.core.types",
    "tolteca_datamodels.core.file.accessor",
    "tolteca_datamodels.core.data_model",
    "tolteca_datamodels.core.input_discovery.entries",
    "tolteca_datamodels.lmt_base.filepath",
    "tolteca_datamodels.lmt_base.obs_spec",
    "tolteca_datamodels.lmt_base.file_store",
    "tolteca_datamodels.core.file_store",
    "tolteca_datamodels.core.source",
)


def test_shared_machinery_does_not_load_instruments():
    # One interpreter imports each module in turn and checks after each import,
    # so a module that loads an instrument package is reported by name.
    script = """
import importlib
import sys

def loaded_instruments():
    return sorted(
        name
        for name in sys.modules
        for prefix in ("tolteca_datamodels.lmt", "tolteca_datamodels.toltec")
        if name == prefix or name.startswith(prefix + ".")
    )

for module in sys.argv[1:]:
    importlib.import_module(module)
    assert not loaded_instruments(), (module, loaded_instruments())
    assert "tolteca_datamodels.file" not in sys.modules, module

from tolteca_datamodels.core.filepath import FilepathInfo
from tolteca_datamodels.lmt_base.filepath import FilepathOrigMetadataMixin

assert FilepathInfo.model_json_schema()["type"] == "object"
schema = FilepathOrigMetadataMixin.model_json_schema(
    mode="serialization"
)
assert schema["properties"]["filepath_info_orig"]["readOnly"]
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, *_SHARED_MODULES],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


_COMPONENT_PACKAGES = (
    "tolteca_datamodels.lmt",
    "tolteca_datamodels.toltec",
    "tolteca_dummy_instru",
)
_COMPONENT_MODULES = (
    "",
    ".types",
    ".filepath",
    ".file_store",
    ".file",
    ".metadata",
    ".input_discovery",
)


def test_component_imports_do_not_construct_data_models():
    # One interpreter imports each component in turn and checks after each
    # import, so a component that assembles data models is reported by name.
    script = """
import importlib
import sys

assembled = {
    "tolteca_datamodels.lmt.data_model", "tolteca_datamodels.toltec.data_model",
    "tolteca_dummy_instru.data_model", "tolteca_datamodels.data_model",
}
for name in sys.argv[1:]:
    importlib.import_module(name)
    assert not assembled & sys.modules.keys(), (name, assembled & sys.modules.keys())
    assert "tolteca_datamodels.file" not in sys.modules, name
    assert "tolteca_datamodels.input_discovery" not in sys.modules, name
"""
    components = [p + m for p in _COMPONENT_PACKAGES for m in _COMPONENT_MODULES]
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, *components],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_filename_entry_point_enables_parsers_without_workflow_assembly():
    script = """
import sys
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.core.data_model import DataModelSet
from tolteca_datamodels.core.filepath import FilepathInfo as CoreFilepathInfo

assert FilepathInfo is CoreFilepathInfo
assert {
    "tolteca_datamodels.lmt.filepath", "tolteca_datamodels.toltec.filepath",
} <= sys.modules.keys()

from tolteca_datamodels.toltec.filepath import (
    ToltecFilepathInfo, ToltecLmtFilepathInfo,
)

assert issubclass(ToltecFilepathInfo, FilepathInfo)
tel = FilepathInfo.from_source(
    "tel_toltec_2025-10-31_145609_00_0002.nc"
)
kids = FilepathInfo.from_source(
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
)
assert type(tel) is ToltecLmtFilepathInfo
assert type(kids) is ToltecFilepathInfo
assert tel.data_models == {'lmt', 'toltec'}
assert kids.data_models == {'toltec'}
assert (tel.interface, kids.interface) == ("tel", "toltec8")
assert not {
    "tolteca_datamodels.lmt.data_model", "tolteca_datamodels.toltec.data_model",
    "tolteca_datamodels.data_model", "tolteca_datamodels.file",
    "tolteca_datamodels.input_discovery",
} & sys.modules.keys()
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


_ENTRY_POINT_FIRST_IMPORTS = [
    "data_model",
    "lmt.data_model",
    "toltec.data_model",
    "file",
    "input_discovery",
]


def test_public_entry_points_assemble_registered_definitions():
    script = """
import importlib
import sys

importlib.import_module(f"tolteca_datamodels.{sys.argv[1]}")
from tolteca_datamodels.core.data_model import DataModelSet

if sys.argv[1] in {"lmt.data_model", "toltec.data_model"}:
    assert "tolteca_datamodels.data_model" not in sys.modules
else:
    assert "tolteca_datamodels.data_model" in sys.modules
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.lmt.data_model import data_model_spec as lmt
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.toltec.data_model import data_model_spec as toltec
from tolteca_datamodels.toltec.file import ToltecFileAccessor

assert lmt.file_accessor_cls is LmtFileAccessor
assert toltec.file_accessor_cls is ToltecFileAccessor
assert DataModelSet.from_select().data_models == {"lmt": lmt, "toltec": toltec}
assert "tolteca_dummy_instru" not in sys.modules
"""
    _run_each(script, _ENTRY_POINT_FIRST_IMPORTS)


@pytest.mark.parametrize(
    ("package_name", "model_name", "filename"),
    [
        (
            "tolteca_datamodels.lmt",
            "LmtFilepathInfo",
            "tel_other_2025-10-31_145609_00_0002.nc",
        ),
        (
            "tolteca_datamodels.toltec",
            "ToltecLmtFilepathInfo",
            "tel_toltec_2025-10-31_145609_00_0002.nc",
        ),
        (
            "tolteca_datamodels.toltec",
            "ToltecFilepathInfo",
            "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
        ),
        (
            "tolteca_dummy_instru",
            "DummyInstruFilepathInfo",
            "chan01-tcs-1-0-2-20251031T120000.nc",
        ),
        (
            "tolteca_dummy_instru",
            "DummyInstruLmtFilepathInfo",
            "tel_dummy_instru_2025-10-31_000001_00_0002.nc",
        ),
    ],
)
def test_public_filepath_types_identify_universal_factory_results(
    package_name, model_name, filename
):
    from tolteca_datamodels.filepath import FilepathInfo

    module = importlib.import_module(f"{package_name}.filepath")
    model = getattr(module, model_name)
    info = FilepathInfo.from_source(filename)
    assert isinstance(info, model)
    assert type(info) is model
    assert model.__name__ == model_name
    assert model_name in module.__all__


_DISCOVERY_FIRST_IMPORTS = [
    "tolteca_datamodels.core.file_store",
    "tolteca_datamodels.core.input_discovery.entries",
    "tolteca_datamodels.lmt_base.input_discovery",
    "tolteca_datamodels.lmt.input_discovery",
    "tolteca_datamodels.toltec.input_discovery",
    "tolteca_dummy_instru.input_discovery",
    "tolteca_datamodels.input_discovery",
]


def test_input_discovery_import_order():
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])
if sys.argv[1] != "tolteca_datamodels.input_discovery":
    assert "tolteca_datamodels.input_discovery" not in sys.modules

from tolteca_datamodels.core.data_model import DataModelSet
from tolteca_datamodels.lmt.input_discovery import LmtInputDiscovery
from tolteca_datamodels.lmt_base.input_discovery import SourcesInputDiscovery
from tolteca_datamodels.toltec.input_discovery import ToltecInputDiscovery
from tolteca_datamodels import input_discovery
from tolteca_datamodels.data_model import DataModelSet

config = input_discovery.InputDiscoveryConfig.model_validate({"inputs": [
    {"data_lmt_path": "/data_lmt"},
    {"data_toltec_path": "/data_lmt/toltec"},
    {"sources": ["https://archive.example/raw.nc"]},
]})
assert isinstance(config.inputs[0], LmtInputDiscovery)
assert isinstance(config.inputs[1], ToltecInputDiscovery)
assert isinstance(config.inputs[2], SourcesInputDiscovery)
assert "tolteca_datamodels.lmt.file" in sys.modules
assert "tolteca_datamodels.toltec.file" in sys.modules
from tolteca_datamodels.lmt.data_model import data_model_spec as lmt
from tolteca_datamodels.toltec.data_model import data_model_spec as toltec
assert DataModelSet.from_select().data_models == {"lmt": lmt, "toltec": toltec}
"""
    _run_each(script, _DISCOVERY_FIRST_IMPORTS)


def test_obs_spec_has_one_instrument_independent_definition():
    script = """
import sys
from pydantic import TypeAdapter
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec, ObsSpecField

spec = TypeAdapter(ObsSpecField).validate_python("1/{tel,1}")
assert type(spec) is ObsSpec
assert spec.interface == (1, "tel")
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for prefix in ("tolteca_datamodels.lmt", "tolteca_datamodels.toltec")
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
    module = importlib.import_module("tolteca_datamodels.toltec._obs_spec")
    assert module._obs_spec_context.field_label_numbers["interface"]["toltec1"] == 1


def test_public_file_enables_builtin_support():
    script = """
import sys
from tolteca_datamodels.core.file import FileAccessorBase

assert "tolteca_datamodels.lmt.file" not in sys.modules
assert "tolteca_datamodels.toltec.file" not in sys.modules

from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.toltec.filepath import ToltecFilepathInfo, ToltecLmtFilepathInfo
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.lmt.data_model import data_model_spec as lmt
from tolteca_datamodels.toltec.data_model import data_model_spec as toltec
from tolteca_datamodels.data_model import DataModelSet

assert issubclass(FileAccessor, FileAccessorBase)
assert issubclass(ToltecFileAccessor, FileAccessorBase)
assert "tolteca_dummy_instru" not in sys.modules
assert "tolteca_datamodels.input_discovery" not in sys.modules
sources = [
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
]
assert type(FilepathInfo.from_source(sources[0])) is ToltecLmtFilepathInfo
assert type(FilepathInfo.from_source(sources[1])) is ToltecFilepathInfo
table = FileAccessor.from_sources(sources).table
assert {
    "source", "source_loc", "source_loc_is_local",
    "source_loc_host", "source_loc_scheme", "source_trace",
    "file_prefix", "file_ext", "file_obsnum",
} <= set(table)
assert table.obsnum.tolist() == [145609, 145605]
assert table.data_kind_toltec.notna().all()
assert "data_kind_toltec" in FileAccessor.from_sources([]).table
assert "data_kind_toltec" in ToltecFileAccessor.from_sources([]).table
assert toltec.file_accessor_cls is ToltecFileAccessor
assert DataModelSet.from_select().data_models == {"lmt": lmt, "toltec": toltec}
core_table = FileAccessorBase.from_sources(sources).table
assert "data_kind_toltec" not in core_table
assert core_table.file_roach.iloc[1] == 8
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
