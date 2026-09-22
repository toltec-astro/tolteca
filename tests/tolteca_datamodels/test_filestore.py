"""Shared file settings and header identification."""

import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.config import RuntimeContext
from tollan.utils.fileloc import fileloc
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.file import SourceInfoModel, guess_info_from_source
from tolteca_datamodels.filestore import FileAccess, FileAccessConfig, identify_source
from tolteca_datamodels.lmt import LmtTelIOMapper
from tolteca_datamodels.toltec.kids.source import (
    identify_source as identify_kids_source,
)
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids.core import KidsConfig


def test_shared_file_config_does_not_create_products(tmp_path):
    products = tmp_path / "products"
    runtime = RuntimeContext(
        {
            "files": {
                "data_lmt_path": str(tmp_path),
                "dataprod_toltec_path": str(products),
                "select": "roach < 7",
            },
            "kids": {"sweep_reduce": {"block": 1}},
        }
    )
    files = FileAccess(runtime)
    assert files.config.data_lmt_path == tmp_path
    assert files.config.dataprod_toltec_path == products
    assert files.config.select == "roach < 7"
    assert not products.exists()


def test_shared_file_config_snapshot_and_typo_validation():
    runtime = RuntimeContext({"files": {"select": "roach < 7"}})
    files = FileAccess(runtime)
    runtime.update_override_config({"files": {"select": "roach >= 7"}})
    assert files.config.select == "roach < 7"
    assert files.recreate().config.select == "roach >= 7"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FileAccessConfig.model_validate({"unknown_setting": "roach == 0"})


@pytest.mark.parametrize("name", ["data_lmt_path", "dataprod_toltec_path", "select"])
def test_kids_config_does_not_accept_file_access_settings(name):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        KidsConfig.model_validate({name: "unused"})


@pytest.mark.parametrize("renamed", [False, True])
def test_telescope_header_identification(data_lmt_path, tmp_path, monkeypatch, renamed):
    source = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    record = (
        SourceInfoModel(source=fileloc(tmp_path / "downloaded.nc"))
        if renamed
        else guess_info_from_source(source)
    )
    assert record is not None
    reads = []
    getitem = NetCDF4ArrayWrapper._getitem

    def read(array, key):
        if array.variable_name.startswith("Data."):
            assert array.variable_name == "Data.TelescopeBackend.TelTime"
            assert key in ((0,), (array.shape[0] - 1,))
            reads.append(key)
        return getitem(array, key)

    with xr.open_dataset(source) as data:
        monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
        identified = identify_source(data, record)
        assert identified.source == record.source
        assert (identified.interface, identified.roach, identified.master) == (
            "tel",
            None,
            "tcs",
        )
        assert (identified.obsnum, identified.subobsnum, identified.scannum) == (
            145609,
            0,
            2,
        )
        assert identified.data_kind == ToltecDataKind.LmtTel
        assert identified.filepath_orig == (
            "/data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc"
        )
        assert identified.file_prefix == "tel_toltec"
        assert identified.filepath_info_orig is not None
        assert identified.file_ut == identified.filepath_info_orig.ut
        assert reads == [(0,), (data.sizes["time"] - 1,)]
    assert record.filepath_orig is None


def test_telescope_header_values_and_opt_in_consistency(data_lmt_path):
    source = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    record = guess_info_from_source(source)
    assert record is not None
    with xr.open_dataset(source) as data:
        mapper = LmtTelIOMapper.from_data_source(data)
        # Deliberately inconsistent in-memory header for the corruption check.
        changed = data.assign({mapper.get_name(mapper.schema.obsnum): 1})
        assert identify_source(changed, record).obsnum == 1
        with pytest.raises(ValueError, match="Conflicting obsnum"):
            identify_source(changed, record, check_filepath_orig=True)
        master_name = mapper.get_name(mapper.schema.master)
        assert master_name is not None
        no_master = data.drop_vars(master_name)
        assert identify_source(no_master, record).master is None
        goal_name = mapper.get_name(mapper.schema.obs_goal)
        assert goal_name is not None
        incomplete = data.drop_vars(goal_name)
        with pytest.raises(KeyError):
            identify_source(incomplete, record)


@pytest.mark.parametrize("renamed", [False, True])
def test_shared_kids_identification_preserves_projection(
    data_lmt_path, tmp_path, renamed
):
    for source in sorted((data_lmt_path / "toltec/tcs/toltec0").glob("*.nc")):
        record = (
            SourceInfoModel(source=fileloc(tmp_path / "downloaded.nc"))
            if renamed
            else guess_info_from_source(source)
        )
        assert record is not None
        with xr.open_dataset(source) as data:
            expected = identify_kids_source(data, record)
            actual = identify_source(data, record)
            assert actual.model_dump(exclude={"metadata"}) == expected.model_dump(
                exclude={"metadata"}
            )
            assert actual.metadata is not None
            assert expected.metadata is None


def test_unknown_payload_retains_filename_guesses():
    record = SourceInfoModel(source="unsupported.nc")
    result = identify_source(xr.Dataset(), record)
    assert result == record


@pytest.mark.parametrize("known", [False, True])
def test_unknown_original_path_does_not_invent_a_telescope_stream(data_lmt_path, known):
    source = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    record = guess_info_from_source(source) if known else SourceInfoModel(source="a.nc")
    assert record is not None
    with xr.open_dataset(source) as data:
        mapper = LmtTelIOMapper.from_data_source(data)
        # An unrecognized header path is an explicit fallback/error case.
        changed = data.assign({mapper.get_name(mapper.schema.filepath_orig): "a.nc"})
        result = identify_source(changed, record)
        assert result.interface == "tel"
        assert result.filepath_info_orig is None
        assert result.data_kind == (
            ToltecDataKind.LmtTel if known else ToltecDataKind.Unknown
        )
        assert result.metadata is not None
        restored = SourceInfoModel.model_validate_json(
            result.model_dump_json(round_trip=True)
        )
        assert type(restored.metadata) is type(result.metadata)
