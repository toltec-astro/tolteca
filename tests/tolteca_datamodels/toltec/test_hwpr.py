"""HWPR identification, native arrays, and mixed readout file tables."""

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.config import RuntimeContext
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.input_discovery import InputDiscovery
from tolteca_datamodels.toltec import open_toltec_hwpr, open_toltec_kids
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.file_store import ToltecFileStore
from tolteca_datamodels.toltec.filepath import ToltecFilepathInfo
from tolteca_datamodels.toltec.hwpr import (
    ToltecHwprMetadata,
)
from tolteca_datamodels.toltec.toltec_db import ToltecDBRawObsMasterCode
from tolteca_datamodels.toltec.types import ToltecDataKind, ToltecInfo


@pytest.fixture
def hwpr_path(data_lmt_path):
    path = data_lmt_path / (
        "toltec/tcs/hwpr/hwpr_156755_000_0002_2026_05_16_12_33_28.nc"
    )
    if not path.is_file():
        pytest.skip(f"HWPR reference file not found: {path}")
    return path


@pytest.mark.parametrize("as_tree", [False, True])
def test_hwpr_accessor_binding_is_structural(hwpr_path, monkeypatch, as_tree):
    def unexpected(*args, **kwargs):
        pytest.fail("Identification must not read HWPR headers or sample arrays")

    with xr.open_dataset(hwpr_path) as ds:
        data = xr.DataTree(ds) if as_tree else ds
        monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", unexpected)
        reader = data.toltec_hwpr
        assert reader is data.toltec_hwpr
        assert reader.data_kind == ToltecDataKind.hwpr
        assert "meta" not in reader.__dict__


def test_metadata_preserves_readout_code_without_roach(hwpr_path, monkeypatch):
    getitem = NetCDF4ArrayWrapper._getitem

    def headers_only(array, key):
        assert array.variable_name.startswith("Header.")
        assert array.variable_name != "Header.Toltec.AdcSnapData"
        return getitem(array, key)

    with open_toltec_hwpr(hwpr_path) as ds:
        monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", headers_only)
        meta = ds.toltec_hwpr.meta
        assert isinstance(meta, ToltecHwprMetadata)
        assert (meta.interface, meta.data_models, meta.instru, meta.interface_kind) == (
            "hwpr",
            {"toltec"},
            "toltec",
            "hwpr",
        )
        assert meta.readout_code == 13
        assert not hasattr(meta, "roach")
        assert not hasattr(meta, "nw")
        assert not hasattr(meta, "array_name")
        assert (meta.master, meta.master_code) == ("tcs", 0)
        assert type(meta.master) is str
        assert meta.master_code is ToltecDBRawObsMasterCode.TCS
        assert (meta.obsnum, meta.subobsnum, meta.scannum) == (156755, 0, 2)
        assert (meta.obs_type, meta.obs_type_code) == ("nominal", 0)
        assert meta.uid_raw_obs_file == "tcs-156755-0-2-hwpr"
        assert meta.n_times == 131
        assert meta.f_smp == 2 * u.Hz
        assert meta.t_int == 65.5 * u.s
        assert meta.t_elapsed.to_value(u.s) == pytest.approx(72)
        assert meta.t0.unix == 1778934808
        assert meta.t1.unix == 1778934880
        assert meta.filepath_orig == f"/data_lmt/toltec/tcs/hwpr/{hwpr_path.name}"
        assert isinstance(meta.filepath_info_orig, ToltecFilepathInfo)
        assert meta.filepath_info_orig.interface == "hwpr"
        assert meta.filepath_info_orig.roach is None
        assert meta.hwp_installed == 0
        assert meta.repeat == 0
        restored = ToltecHwprMetadata.model_validate_json(meta.model_dump_json())
        assert restored.master == meta.master
        assert restored.master_code is meta.master_code
        assert restored.readout_code == meta.readout_code
        assert restored.uid_raw_obs_file == meta.uid_raw_obs_file


@pytest.mark.parametrize(
    "name", ["data_models", "instru", "interface", "interface_kind"]
)
def test_hwpr_identity_rejects_other_values(hwpr_path, name):
    with open_toltec_hwpr(hwpr_path) as ds:
        payload = ds.toltec_hwpr.meta.model_dump(round_trip=True)
    with pytest.raises(ValidationError, match=name):
        ToltecHwprMetadata.model_validate(payload | {name: "unrelated"})


def test_all_readout_arrays_retain_native_values_and_dimensions(hwpr_path):
    names = {
        "enc_ts": "HwprEncTs",
        "enc_val": "HwprEncVal",
        "enc_uts": "HwprEncUts",
        "zeropt_ts": "HwprZeroptTs",
        "zeropt_val": "HwprZeroptVal",
        "zeropt_uts": "HwprZeroptUts",
        "pps_ts": "HwprPpsTs",
        "pps_val": "HwprPpsVal",
        "pps_uts": "HwprPpsUts",
        "sensor_chan": "HwprSensorChan",
        "sensor_volt": "HwprSensorVolt",
        "sensor_uts": "HwprSensorUts",
        "ts": "Ts",
        "recv_time": "RecvTime",
    }
    with open_toltec_hwpr(hwpr_path) as ds:
        view = ds.toltec_hwpr
        for attr, native in names.items():
            array = getattr(view, attr)
            original = ds[f"Data.Toltec.{native}"]
            assert not array.variable._in_memory
            assert array.dims == original.dims
            assert array.shape == original.shape
            assert array.dtype == original.dtype
            assert array.attrs == original.attrs
            np.testing.assert_array_equal(array.to_numpy(), original.to_numpy())
        assert view.enc_val.shape == (131, 500)
        np.testing.assert_array_equal(np.unique(view.enc_val), [2137])


def test_invalid_metadata_headers(hwpr_path):
    with open_toltec_hwpr(hwpr_path) as ds:
        incomplete = ds.drop_vars("Header.Toltec.Master")
        with pytest.raises(ValueError, match="Header.Toltec.Master"):
            _ = incomplete.toltec_hwpr.meta
        for value in (0, -1, float("nan")):
            invalid = ds.assign({"Header.Toltec.SampleFreq": value})
            with pytest.raises(ValueError, match="Sample frequency"):
                _ = invalid.toltec_hwpr.meta


def test_explicit_openers_reject_wrong_payload_and_close(hwpr_path, monkeypatch):
    open_dataset = xr.open_dataset
    resources = []

    def tracked(*args, **kwargs):
        ds = open_dataset(*args, **kwargs)
        resources.append(ds)
        return ds

    monkeypatch.setattr(xr, "open_dataset", tracked)
    with pytest.raises(ValueError, match="KIDs dataset"):
        open_toltec_kids(hwpr_path)
    assert resources[-1]._close is None
    roach = (
        hwpr_path.parent.parent
        / "toltec7"
        / ("toltec7_156755_000_0002_2026_05_16_12_33_28.nc")
    )
    with pytest.raises(ValueError, match="HWPR dataset"):
        open_toltec_hwpr(roach)
    assert resources[-1]._close is None
    with pytest.raises(ValueError, match="Remote sources"):
        open_toltec_hwpr(f"archive:{hwpr_path}")


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_mixed_discovery_selection_and_open(
    data_lmt_path, hwpr_path, mode, monkeypatch
):
    getitem = NetCDF4ArrayWrapper._getitem

    def headers_only(array, key):
        assert array.variable_name.startswith("Header.")
        assert array.variable_name != "Header.Toltec.AdcSnapData"
        return getitem(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", headers_only)
    discovery = InputDiscovery(
        RuntimeContext(
            {
                "input_discovery": {
                    "inputs": [
                        {
                            "data_toltec_path": data_lmt_path / "toltec",
                            "obs_spec": "tcs-156755/{hwpr,7,11}",
                            "mode": mode,
                        }
                    ]
                }
            }
        )
    )
    table = discovery.discover(
        data_models=DataModelSet.from_select(
            selection=["toltec"],
        )
    )
    assert len(table) == 5
    assert table.interface.value_counts().to_dict() == {
        "toltec7": 2,
        "toltec11": 2,
        "hwpr": 1,
    }
    hwpr = table.loc[table.interface.eq("hwpr")].iloc[0]
    assert pd.isna(hwpr.roach)
    assert hwpr.data_kind_toltec == ToltecDataKind.hwpr.value
    assert len(table.toltec_file.query("bits_any_set(data_kind_toltec, hwpr)")) == 1
    assert (
        len(table.toltec_file.query("bits_any_set(data_kind_toltec, raw_kids_data)"))
        == 4
    )
    assert table.toltec_file.query_obs_spec("156755/13").empty
    assert table.toltec_file.query_obs_spec("156755/(7,12)").shape[0] == 4
    assert ToltecFileStore(data_lmt_path / "toltec").glob_obs_spec("156755/13") == []
    assert 13 not in ToltecInfo.roaches
    assert "hwpr" not in ToltecInfo.interface_roach
    with table.toltec_file.open(on_error="raise") as opened:
        opened.toltec_file.read_metadata(on_error="raise")
        assert [list(trace.iter_error_entries()) for trace in opened.source_trace] == [
            [] for _ in range(len(opened))
        ]
        resources = opened.io_obj.tolist()
        for row in opened.itertuples(index=False):
            if row.interface == "hwpr":
                assert row.readout_code == 13
                assert pd.isna(row.roach)
                assert pd.isna(row.nw)
                assert pd.isna(row.array_name)
                assert row.uid_raw_obs_file.endswith("-hwpr")
                assert row.io_obj.toltec_hwpr.enc_val.shape == (131, 500)
            else:
                assert row.readout_code == row.roach
                assert row.readout_code in (7, 11)
                assert row.io_obj.toltec_kids.meta.readout_code == row.readout_code
    assert all(ds._close is None for ds in resources)


def test_current_hwpr_identity_selects_reader_without_losing_filename_provenance(
    hwpr_path,
):
    # Deliberately misleading source_loc exercises authoritative header population.
    wrong = (
        hwpr_path.parent.parent
        / "toltec7"
        / ("toltec7_156755_000_0002_2026_05_16_12_33_28.nc")
    )
    table = ToltecFileAccessor.from_sources(
        [{"source": hwpr_path, "source_loc": wrong}], mode="guess"
    ).table
    table[["interface", "interface_kind"]] = None
    table.toltec_file.read_metadata(on_error="raise")
    row = next(table.itertuples(index=False))
    assert row.source == hwpr_path
    assert row.file_roach == 7
    assert row.file_interface == "toltec7"
    assert row.interface == "hwpr"
    assert row.readout_code == 13
    assert row.data_kind_toltec == ToltecDataKind.hwpr.value


def test_renamed_hwpr_and_telescope_companion(data_lmt_path, hwpr_path, tmp_path):
    renamed = tmp_path / "cached.nc"
    renamed.symlink_to(hwpr_path)
    tel = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    table = ToltecFileAccessor.from_sources([renamed, tel], mode="guess_or_read").table
    assert table.interface.tolist() == ["hwpr", "tel"]
    with table.toltec_file.open(on_error="raise") as opened:
        assert opened.readout_code.iloc[0] == 13
        assert pd.isna(opened.readout_code.iloc[1])
        assert opened.data_kind_toltec.tolist() == [
            ToltecDataKind.hwpr.value,
            ToltecDataKind.lmt_tel.value,
        ]
        assert opened.io_obj.iloc[0].toltec_hwpr.meta.interface == "hwpr"
        assert opened.io_obj.iloc[1].lmt_tel.meta.interface == "tel"
