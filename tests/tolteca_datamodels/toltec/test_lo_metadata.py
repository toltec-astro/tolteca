"""LO-center metadata is shared by raw sweeps and timestreams."""

import astropy.units as u
import pytest
from pydantic import TypeAdapter
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import open_toltec


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_recorded_lo_center(data_lmt_path, monkeypatch, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    reads = []
    original = NetCDF4ArrayWrapper.__getitem__

    def read(array, key):
        reads.append(array.variable_name)
        return original(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", read)
    with open_toltec(path) as data:
        meta = data.toltec_kids.meta
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
    pattern = f"*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with open_toltec(path) as data:
        missing = data.drop_vars("Header.Toltec.LoCenterFreq")
        with pytest.raises(ValueError, match="Header.Toltec.LoCenterFreq"):
            _ = missing.toltec_kids.meta


@pytest.mark.parametrize("processed", [False, True])
def test_timestream_lo_units(data_lmt_path, processed):
    root = data_lmt_path / "toltec/tcs/toltec0"
    (path,) = root.glob("*_09_27_57.nc")
    with open_toltec(path) as data:
        name = "Header.Toltec.LoCenterFreq"
        frequency = data[name].item()
        # Equivalent unit representation of the recorded value.
        changed = data.assign({name: ((), frequency / 1e6, {"units": "MHz"})})
        if processed:
            changed = changed.assign_attrs({"Header.Kids.kind": "processed_timestream"})
        assert changed.toltec_kids.meta.f_lo_center * u.Hz == frequency * u.Hz


@pytest.mark.parametrize("kind", ["processed_sweep", "processed_timestream"])
def test_processed_lo_center_can_be_absent(real_targsweep_file, kind):
    with open_toltec(real_targsweep_file) as data:
        # Deliberately synthetic processed encoding to check the omission policy.
        changed = data.drop_vars("Header.Toltec.LoCenterFreq").assign_attrs(
            {"Header.Kids.kind": kind}
        )
        assert changed.toltec_kids.meta.f_lo_center is None
