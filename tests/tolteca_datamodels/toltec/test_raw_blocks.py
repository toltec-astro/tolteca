"""Native acquisition block identification and recorded probe axes."""

import numpy as np
import pytest
import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.kids.core import _TIMESTREAM_LO_CHECK_SAMPLES


@pytest.mark.parametrize("obs_type_code", [0, 1])
@pytest.mark.parametrize("chunks", [None, {}, {"time": 256}])
def test_timestream_lo_reads_only_endpoint_windows(
    data_lmt_path, monkeypatch, obs_type_code, chunks
):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*_09_27_57.nc")
    original = NetCDF4ArrayWrapper.__getitem__
    lo_reads = []

    def read(array, key):
        assert array.variable_name not in {"Data.Toltec.Is", "Data.Toltec.Qs"}
        result = original(array, key)
        if array.variable_name == "Data.Toltec.LoFreq":
            assert result.size <= _TIMESTREAM_LO_CHECK_SAMPLES
            lo_reads.append(result.copy())
        return result

    with open_toltec_kids(path, chunks=chunks) as original_source:
        source = original_source.assign(
            {"Header.Toltec.ObsType": xr.DataArray(obs_type_code)}
        )
        monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", read)
        index = source.toltec_kids.block_index
        assert sum(part.size for part in lo_reads) == 2 * _TIMESTREAM_LO_CHECK_SAMPLES
        table = index.meta.probe_axis_table
        assert table is not None
        assert table["n_samples"].tolist() == [source.sizes["time"]]
        assert table["sample_start"].tolist() == [0]
        assert table["sample_end"].tolist() == [source.sizes["time"]]
        assert table["f_lo"].to_value("Hz").tolist() == [lo_reads[0][0]]
        assert source.toltec_kids.block_index is index


@pytest.mark.parametrize("n_times", [0, 1, 64, 65, 128, 129])
def test_timestream_lo_short_arrays_and_units(data_lmt_path, n_times):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*_09_27_57.nc")
    with open_toltec_kids(path) as original_source:
        source = original_source.isel(time=slice(0, n_times))
        lo = source["Data.Toltec.LoFreq"]
        values = lo.to_numpy() / 1e6
        source["Data.Toltec.LoFreq"] = xr.DataArray(
            values, dims="time", attrs={"units": "MHz"}
        )
        table = source.toltec_kids.block_index.meta.probe_axis_table
        assert table is not None
        assert len(table) == bool(n_times)
        assert table["n_samples"].tolist() == ([n_times] if n_times else [])
        if n_times:
            assert table["f_lo"].to_value("Hz").tolist() == [values[0] * 1e6]


@pytest.mark.parametrize("changed_index", [1, -1])
def test_timestream_lo_disagreement_falls_back_to_full_scan(
    data_lmt_path, changed_index
):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*_09_27_57.nc")
    with open_toltec_kids(path) as original_source:
        source = original_source.isel(time=slice(0, 4 * _TIMESTREAM_LO_CHECK_SAMPLES))
        lo = source["Data.Toltec.LoFreq"].load().copy(deep=True)
        lo[changed_index] += 1_000_000
        source["Data.Toltec.LoFreq"] = lo
        table = source.toltec_kids.block_index.meta.probe_axis_table
        assert table is not None
        expected_starts = np.r_[0, np.flatnonzero(np.diff(lo.values)) + 1]
        np.testing.assert_array_equal(table["sample_start"], expected_starts)
        np.testing.assert_array_equal(
            table["n_samples"], np.diff(np.r_[expected_starts, source.sizes["time"]])
        )
        np.testing.assert_array_equal(table["f_lo"].value, lo.values[expected_starts])


def test_processed_timestream_missing_lo_does_not_invent_frequency(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*_09_27_57.nc")
    with open_toltec_kids(path) as source:
        processed = source.drop_vars(
            ["Data.Toltec.LoFreq", "Header.Toltec.LoCenterFreq"]
        )
        processed = processed.assign_attrs({"Header.Kids.kind": "processed_timestream"})
        meta = processed.toltec_kids.block_index.meta
        assert meta.probe_axis_table is not None
        assert meta.probe_axis_table["n_samples"][0] == source.sizes["time"]
        assert "f_lo" not in meta.probe_axis_table.colnames
        assert "f_sweep" not in meta.probe_axis_table.colnames
