"""Raw design dimensions and ADC references are required without reading arrays."""

import pytest
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import open_toltec


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
@pytest.mark.parametrize("missing", ["loclen", "Header.Toltec.AdcSnapData"])
def test_raw_requires_design_dimension_and_adc(data_lmt_path, suffix, missing):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with open_toltec(path) as data:
        changed = (
            data.drop_dims(missing) if missing in data.dims else data.drop_vars(missing)
        )
        with pytest.raises(KeyError, match=missing):
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
    with open_toltec(real_targsweep_file) as data:
        meta = data.toltec_kids.meta
        assert meta.n_kids_design == data.sizes["loclen"] == 684
        assert meta.adc_snap == "Header.Toltec.AdcSnapData"
        assert not data[meta.adc_snap].variable._in_memory


def test_processed_design_and_adc_can_be_absent(real_targsweep_file):
    with open_toltec(real_targsweep_file) as data:
        # Synthetic processed encoding, not a supported acquisition filename.
        changed = (
            data.drop_dims("loclen")
            .drop_vars("Header.Toltec.AdcSnapData")
            .assign_attrs({"Header.Kids.kind": "processed_sweep"})
        )
        meta = changed.toltec_kids.meta
        assert meta.n_kids_design is None
        assert meta.adc_snap is None
