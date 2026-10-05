"""Import boundaries for filename machinery, models and file accessors."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("name", "class_name"),
    [
        ("tolteca_datamodels.lmt.file_store", "LmtFileStore"),
        ("tolteca_datamodels.toltec.file_store", "ToltecFileStore"),
        ("tolteca_datamodels.file_store", "FileStoreBase"),
    ],
)
def test_file_store_module_names(name, class_name):
    module = importlib.import_module(name)
    assert getattr(module, class_name).__module__ == name
    old_name = name.replace(".file_store", ".filestore")
    with pytest.raises(ModuleNotFoundError) as exc:
        importlib.import_module(old_name)
    assert exc.value.name == old_name


@pytest.mark.parametrize(
    "name",
    ["tolteca_datamodels.lmt_base.file_store", "tolteca_datamodels.lmt_base.filestore"],
)
def test_lmt_base_no_longer_owns_file_store(name):
    with pytest.raises(ModuleNotFoundError) as exc:
        importlib.import_module(name)
    assert exc.value.name == name


def test_file_store_infrastructure_does_not_load_domain_modules():
    script = """
import sys
from tolteca_datamodels.file_store import FileStoreBase

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
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "module",
    [
        "tolteca_datamodels._filepath",
        "tolteca_datamodels.lmt_base._filepath",
        "tolteca_datamodels.lmt_base.obs_spec",
        "tolteca_datamodels.file_store",
        "tolteca_datamodels.source",
    ],
)
def test_shared_machinery_does_not_load_instruments(module):
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])
from tolteca_datamodels._filepath import FilepathInfo
from tolteca_datamodels.lmt_base._filepath import FilepathOrigMetadataMixin

assert FilepathInfo.model_json_schema()["type"] == "object"
schema = FilepathOrigMetadataMixin[FilepathInfo].model_json_schema(
    mode="serialization"
)
assert schema["properties"]["filepath_info_orig"]["readOnly"]
assert "tolteca_datamodels.file" not in sys.modules
assert "tolteca_datamodels.model_table" not in sys.modules
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


@pytest.mark.parametrize("instrument", ["lmt", "toltec"])
@pytest.mark.parametrize("module", ["_filepath", "metadata"])
def test_filename_and_metadata_models_do_not_load_file_accessors(instrument, module):
    script = """
import importlib
import sys

importlib.import_module(f"tolteca_datamodels.{sys.argv[1]}.{sys.argv[2]}")
assert "tolteca_datamodels.file" not in sys.modules
assert "tolteca_datamodels.toltec.file" not in sys.modules
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, instrument, module],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_file_entry_points_export_the_canonical_classes():
    from tolteca_datamodels._filepath import FilepathInfo as FilepathInfoModel
    from tolteca_datamodels.file import FileAccessor, FilepathInfo
    from tolteca_datamodels.lmt import LmtFilepathInfo
    from tolteca_datamodels.lmt._filepath import LmtFilepathInfo as LmtFilepathInfoModel
    from tolteca_datamodels.toltec import ToltecFilepathInfo
    from tolteca_datamodels.toltec._filepath import (
        ToltecFilepathInfo as ToltecFilepathInfoModel,
    )

    assert FileAccessor.__module__ == "tolteca_datamodels.file"
    assert FilepathInfo.__module__ == "tolteca_datamodels._filepath"
    assert FilepathInfo is FilepathInfoModel
    assert LmtFilepathInfo is LmtFilepathInfoModel
    assert ToltecFilepathInfo is ToltecFilepathInfoModel


@pytest.mark.parametrize(
    "first",
    [
        "tolteca_datamodels.file_store",
        "tolteca_datamodels.lmt.input_discovery",
        "tolteca_datamodels.toltec.input_discovery",
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

from tolteca_datamodels.file_store import FileStoreInputDiscovery
from tolteca_datamodels.lmt.input_discovery import LmtInputDiscovery
from tolteca_datamodels.toltec.input_discovery import ToltecInputDiscovery
from tolteca_datamodels import input_discovery

for cls, domain in ((LmtInputDiscovery, "lmt"), (ToltecInputDiscovery, "toltec")):
    assert cls.__module__ == f"tolteca_datamodels.{domain}.input_discovery"
    assert issubclass(cls, FileStoreInputDiscovery)
    assert getattr(input_discovery, cls.__name__) is cls
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
    module = importlib.import_module("tolteca_datamodels.toltec.obs_spec")
    assert not hasattr(module, "ObsSpec")
    assert module.obs_spec_context.labels["interface"]["toltec1"] == 1


def test_toltec_file_owns_dataframe_accessor_registration():
    script = """
import sys
import pandas as pd
from tolteca_datamodels.file import FileAccessor, FilepathInfo
from tolteca_datamodels.lmt import LmtFilepathInfo
from tolteca_datamodels.toltec import ToltecFilepathInfo

assert "tolteca_datamodels.toltec.file" not in sys.modules
assert not hasattr(pd.DataFrame, "toltec_file")
assert type(FilepathInfo.from_source(
    "tel_toltec_2025-10-31_145609_00_0002.nc"
)) is LmtFilepathInfo
assert type(FilepathInfo.from_source(
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
)) is ToltecFilepathInfo
assert list(FileAccessor.from_sources([])) == [
    "source", "source_loc", "source_loc_is_local",
    "source_loc_host", "source_loc_scheme"
]

from tolteca_datamodels.toltec.file import ToltecFileAccessor
assert issubclass(ToltecFileAccessor, FileAccessor)
assert pd.DataFrame.toltec_file is ToltecFileAccessor
table = ToltecFileAccessor.from_sources([])
assert isinstance(table.toltec_file, ToltecFileAccessor)
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
