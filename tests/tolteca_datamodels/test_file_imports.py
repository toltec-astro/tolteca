"""Import boundaries for filename machinery, models and file accessors."""

import importlib
import importlib.util
import subprocess
import sys

import pytest


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
    old_name = name.replace(".file_store", ".filestore")
    with pytest.raises(ModuleNotFoundError) as exc:
        importlib.import_module(old_name)
    assert exc.value.name == old_name


def test_file_store_infrastructure_does_not_load_domain_modules():
    script = """
import sys
from tolteca_datamodels.core.file import source_trace_issues
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


@pytest.mark.parametrize(
    "first", ["lmt.file_store", "toltec.file_store", "toltec", "data_model"]
)
def test_public_store_assembles_builtin_requirements_in_any_import_order(first):
    script = """
import importlib
import sys

importlib.import_module("tolteca_datamodels." + sys.argv[1])
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.toltec.file_store import ToltecFileStore
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.lmt.data_model import data_model as lmt_data_model

store = LmtFileStore("/data_lmt")
assert LmtFileStore.__module__ == "tolteca_datamodels.lmt.file_store"
assert "toltec" in vars(LmtFileStore)
assert importlib.util.find_spec("tolteca_datamodels.lmt._file_store") is None
assert store.toltec.path == store.substores[ToltecFileStore].path
assert list(store.toltec.iter_glob_patterns("tcs-1/toltec7")) == [
    "toltec/tcs/toltec7/toltec7_000001_*.nc"
]
assert DataModelSet.from_select(["toltec"]).file_store_types == (
    *lmt_data_model.file_store_types, ToltecFileStore,
)
assert issubclass(LmtFileStore, lmt_data_model.file_store_types)
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
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, first],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "module",
    [
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
    ],
)
def test_shared_machinery_does_not_load_instruments(module):
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])
from tolteca_datamodels.core.filepath import FilepathInfo
from tolteca_datamodels.lmt_base.filepath import FilepathOrigMetadataMixin

assert FilepathInfo.model_json_schema()["type"] == "object"
schema = FilepathOrigMetadataMixin.model_json_schema(
    mode="serialization"
)
assert schema["properties"]["filepath_info_orig"]["readOnly"]
assert "tolteca_datamodels.file" not in sys.modules
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for prefix in ("tolteca_datamodels.lmt", "tolteca_datamodels.toltec")
    for name in sys.modules
)
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, module],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "package",
    ["tolteca_datamodels.lmt", "tolteca_datamodels.toltec", "tolteca_dummy_instru"],
)
@pytest.mark.parametrize(
    "module",
    [
        "",
        ".types",
        ".filepath",
        ".file_store",
        ".file",
        ".metadata",
        ".input_discovery",
    ],
)
def test_component_imports_do_not_construct_data_models(package, module):
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1] + sys.argv[2])
package = importlib.import_module(sys.argv[1])
from tolteca_datamodels.core.data_model import DataModelSet

assert DataModelSet.from_select().data_models == {}
assert "data_model" not in vars(package)
assert "data_model" not in package.__all__
assert not {
    "tolteca_datamodels.lmt.data_model", "tolteca_datamodels.toltec.data_model",
    "tolteca_dummy_instru.data_model", "tolteca_datamodels.data_model",
} & sys.modules.keys()
assert "tolteca_datamodels.file" not in sys.modules
assert "tolteca_datamodels.input_discovery" not in sys.modules
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, package, module],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_file_entry_points_export_the_canonical_classes():
    from tolteca_datamodels.core.file.accessor import FileAccessor as FileAccessorBase
    from tolteca_datamodels.core.filepath import FilepathInfo as FilepathInfoModel
    from tolteca_datamodels.file import FileAccessor
    from tolteca_datamodels.filepath import FilepathInfo

    assert FileAccessor is FileAccessorBase
    assert FileAccessor.__module__ == "tolteca_datamodels.core.file.accessor"
    assert FilepathInfo.__module__ == "tolteca_datamodels.core.filepath"
    assert FilepathInfo is FilepathInfoModel


@pytest.mark.parametrize(
    "package",
    ["tolteca_datamodels.lmt", "tolteca_datamodels.toltec", "tolteca_dummy_instru"],
)
def test_definition_can_drive_discovery_without_registration(package):
    script = """
import importlib
import sys
from tolteca_datamodels.core.data_model import DataModelSpec, DataModelSet

module = importlib.import_module(sys.argv[1] + ".data_model")
assert isinstance(module.data_model, DataModelSpec)
assert DataModelSet.from_select().data_models == {}
definitions = {
    "tolteca_datamodels.lmt.data_model", "tolteca_datamodels.toltec.data_model",
    "tolteca_dummy_instru.data_model",
}
assert definitions & sys.modules.keys() == {module.__name__}
from tolteca_datamodels.lmt_base.input_discovery import SourcesInputDiscovery

sources = {
    "lmt": "tel_toltec_2025-10-31_000001_00_0002.nc",
    "toltec": "toltec7_000001_000_0002_2025_10_31_09_20_06_tune.nc",
    "dummy_instru": "chan01-tcs-1-0-2-20251031T120000.nc",
}
models = DataModelSet(data_models={module.data_model.name: module.data_model})
entry = SourcesInputDiscovery(sources=[sources[module.data_model.name]], mode="guess")
table = entry.discover(data_models=models)
assert table.obsnum.tolist() == [1]
assert table.scannum.tolist() == [2]
assert DataModelSet.from_select().data_models == {}
assert "tolteca_datamodels.data_model" not in sys.modules
assert "tolteca_datamodels.file" not in sys.modules
assert "tolteca_datamodels.input_discovery" not in sys.modules
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, package],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
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
assert tel.data_models == ["lmt", "toltec"]
assert kids.data_models == ["toltec"]
assert (tel.interface, kids.interface) == ("tel", "toltec8")
assert DataModelSet.from_select().data_models == {}
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


@pytest.mark.parametrize(
    "first",
    ["data_model", "lmt.data_model", "toltec.data_model", "file", "input_discovery"],
)
def test_public_entry_points_assemble_registered_definitions(first):
    script = """
import importlib
import sys

importlib.import_module(f"tolteca_datamodels.{sys.argv[1]}")
from tolteca_datamodels.core.data_model import DataModelSet

if sys.argv[1] in {"lmt.data_model", "toltec.data_model"}:
    assert DataModelSet.from_select().data_models == {}
else:
    assert set(DataModelSet.from_select().data_models) == {"lmt", "toltec"}
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.lmt.data_model import data_model as lmt
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.toltec.data_model import data_model as toltec
from tolteca_datamodels.toltec.file import ToltecFileAccessor

assert lmt.file_accessor_cls is LmtFileAccessor
assert toltec.file_accessor_cls is ToltecFileAccessor
assert DataModelSet.from_select().data_models == {"lmt": lmt, "toltec": toltec}
assert "tolteca_dummy_instru" not in sys.modules
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, first],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


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

    package = importlib.import_module(package_name)
    module = importlib.import_module(f"{package_name}.filepath")
    model = getattr(module, model_name)
    info = FilepathInfo.from_source(filename)
    assert isinstance(info, model)
    assert type(info) is model
    assert model.__name__ == model_name
    assert model_name in module.__all__
    assert not hasattr(package, model_name)
    assert importlib.util.find_spec(f"{package_name}._filepath") is None


@pytest.mark.parametrize(
    "first",
    [
        "tolteca_datamodels.core.file_store",
        "tolteca_datamodels.core.input_discovery.entries",
        "tolteca_datamodels.lmt_base.input_discovery",
        "tolteca_datamodels.lmt.input_discovery",
        "tolteca_datamodels.toltec.input_discovery",
        "tolteca_dummy_instru.input_discovery",
        "tolteca_datamodels.input_discovery",
    ],
)
def test_input_discovery_import_order(first):
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
from tolteca_datamodels.lmt.data_model import data_model as lmt
from tolteca_datamodels.toltec.data_model import data_model as toltec
assert DataModelSet.from_select().data_models == {"lmt": lmt, "toltec": toltec}
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, first],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_obs_spec_has_one_instrument_independent_definition():
    script = """
import sys
from pydantic import TypeAdapter
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec, ObsSpecField

spec = TypeAdapter(ObsSpecField).validate_python("1/{tel,1}")
assert type(spec) is ObsSpec
assert spec.interface == (1, "tel")
assert not hasattr(ObsSpec, "_label_numbers")
assert not hasattr(ObsSpec, "_label_number_registry")
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
    assert not hasattr(module, "ObsSpec")
    assert module._obs_spec_context.field_label_numbers["interface"]["toltec1"] == 1


def test_public_file_enables_builtin_support():
    script = """
import sys
import pandas as pd
from tolteca_datamodels.core.file.accessor import FileAccessor as FileAccessorBase

assert "tolteca_datamodels.lmt.file" not in sys.modules
assert "tolteca_datamodels.toltec.file" not in sys.modules
assert not hasattr(pd.DataFrame, "toltec_file")

from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.toltec.filepath import ToltecFilepathInfo
from tolteca_datamodels.toltec.filepath import ToltecLmtFilepathInfo

assert FileAccessor is FileAccessorBase
assert "tolteca_dummy_instru" not in sys.modules
assert "tolteca_datamodels.input_discovery" not in sys.modules
assert type(FilepathInfo.from_source(
    "tel_toltec_2025-10-31_145609_00_0002.nc"
)) is ToltecLmtFilepathInfo
assert type(FilepathInfo.from_source(
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
)) is ToltecFilepathInfo
table = FileAccessor.from_sources([])
assert {
    "source", "source_loc", "source_loc_is_local",
    "source_loc_host", "source_loc_scheme", "source_trace",
    "file_prefix", "file_ext", "file_obsnum",
} <= set(table)
assert "data_kind_toltec" in table

from tolteca_datamodels.toltec.file import ToltecFileAccessor
assert issubclass(ToltecFileAccessor, FileAccessor)
assert pd.DataFrame.toltec_file is ToltecFileAccessor
table = ToltecFileAccessor.from_sources([])
assert isinstance(table.toltec_file, ToltecFileAccessor)
assert "data_kind_toltec" in table
from tolteca_datamodels.lmt.data_model import data_model as lmt
from tolteca_datamodels.toltec.data_model import data_model
from tolteca_datamodels.core.data_model import DataModelSet
assert data_model.file_accessor_cls is ToltecFileAccessor
assert DataModelSet.from_select().data_models == {"lmt": lmt, "toltec": data_model}
assert "data_kind_toltec" in FileAccessor.from_sources([])
assert "data_kind_toltec" in FileAccessorBase.from_sources([])
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
