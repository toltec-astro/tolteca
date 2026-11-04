"""LO-center metadata is shared by raw sweeps and timestreams."""

import pytest
from pydantic import TypeAdapter
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_recorded_lo_center(data_lmt_path, monkeypatch, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    reads = []
    original = NetCDF4ArrayWrapper.__getitem__

    def read(array, key):
        reads.append(array.variable_name)
        return original(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", read)
    with open_toltec_kids(path) as data:
        meta = data.toltec_kids.meta
        assert isinstance(meta, ToltecRawObsMetadata)
        assert reads.count("Header.Toltec.LoCenterFreq") == 1
        payloads = {"Data.Toltec.Is", "Data.Toltec.Qs", "Header.Toltec.AdcSnapData"}
        assert not payloads.intersection(reads)
        assert meta.f_lo_center == data["Header.Toltec.LoCenterFreq"].item()
        adapter = TypeAdapter(type(meta))
        restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
        assert restored.f_lo_center == meta.f_lo_center


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_raw_metadata_requires_lo_center(data_lmt_path, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with open_toltec_kids(path) as data:
        missing = data.drop_vars("Header.Toltec.LoCenterFreq")
        with pytest.raises(ValueError, match="Header.Toltec.LoCenterFreq"):
            _ = missing.toltec_kids.meta
