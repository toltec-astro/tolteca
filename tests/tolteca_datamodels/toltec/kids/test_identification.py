"""KIDs header and structural identification boundaries."""

import pytest
import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.file import SourceInfoModel, guess_info_from_source
from tolteca_datamodels.toltec.kids.source import identify_source
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_telescope_time_dimension_is_not_kids_evidence(
    data_lmt_path, tmp_path, monkeypatch
):
    sources = sorted((data_lmt_path / "tel").glob("*.nc"))
    assert len(sources) == 4

    def unexpected(*args, **kwargs):
        pytest.fail("Structural identification must not read any telescope values")

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", unexpected)
    for source in sources:
        record = guess_info_from_source(source)
        assert record is not None
        with xr.open_dataset(source) as data:
            assert data.toltec_kids.data_kind == ToltecDataKind.Unknown
            assert identify_source(data, record).data_kind == ToltecDataKind.LmtTel
            unknown = identify_source(
                data, SourceInfoModel(source=tmp_path / "downloaded.nc")
            )
            assert unknown.data_kind == ToltecDataKind.Unknown
            assert unknown.obsnum is None


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", None])
def test_raw_signal_structure_remains_identifiable(data_lmt_path, suffix, monkeypatch):
    # Remove the explicit type from actual payloads to exercise fallback only.
    pattern = "*09_27_57.nc" if suffix is None else f"*_{suffix}.nc"
    source = next((data_lmt_path / "toltec/tcs/toltec0").glob(pattern))

    def unexpected(*args, **kwargs):
        pytest.fail("Structural identification must not read signal/header values")

    with xr.open_dataset(source) as data:
        field = data.toltec_kids.mapper.schema.obs_type
        name = data.toltec_kids.mapper.get_name(field)
        without_type = data.drop_vars(name)
        monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", unexpected)
        expected = (
            ToltecDataKind.RawTimeStream if suffix is None else ToltecDataKind.RawSweep
        )
        assert without_type.toltec_kids.data_kind == expected
