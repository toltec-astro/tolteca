"""Design metadata and lazy ADC snapshot access."""

import pytest
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_raw_requires_design_dimension(data_lmt_path, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with open_toltec_kids(path) as data:
        changed = data.drop_dims("loclen")
        with pytest.raises(KeyError, match="loclen"):
            _ = changed.toltec_kids.meta


def test_design_metadata_uses_no_design_or_adc_payload(
    real_targsweep_file, monkeypatch
):
    original = NetCDF4ArrayWrapper.__getitem__

    def read(array, key):
        assert array.variable_name not in {
            "Header.Toltec.Xloc",
            "Header.Toltec.Yloc",
            "Header.Toltec.AdcSnapData",
            "Data.Toltec.Is",
            "Data.Toltec.Qs",
        }
        return original(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", read)
    with open_toltec_kids(real_targsweep_file) as data:
        meta = data.toltec_kids.meta
        assert isinstance(meta, ToltecRawObsMetadata)
        assert meta.n_kids_design == data.sizes["loclen"] == 684
        snap = data.toltec_kids.adc_snap
        assert snap.variable is data["Header.Toltec.AdcSnapData"].variable
        assert not snap.variable._in_memory


def test_missing_adc_does_not_prevent_reading_metadata(real_targsweep_file):
    with open_toltec_kids(real_targsweep_file) as data:
        changed = data.drop_vars("Header.Toltec.AdcSnapData")
        meta = changed.toltec_kids.meta
        assert isinstance(meta, ToltecRawObsMetadata)
        assert meta.n_kids_design == 684
        with pytest.raises(KeyError):
            _ = changed.toltec_kids.adc_snap
