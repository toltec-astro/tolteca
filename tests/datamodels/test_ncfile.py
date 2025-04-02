from pathlib import Path

import netCDF4
import pytest

from tolteca_datamodels.toltec.file import guess_info_from_sources
from tolteca_datamodels.toltec.ncfile import NcFileIO
from tolteca_datamodels.toltec.types import ToltecDataKind

data_root = Path(__file__).with_name("data_lmt")

pytestmark = pytest.mark.skipif(
    not data_root.exists(),
    reason=f"test data {data_root} does not exist.",
)


def test_ncfile_no_open():
    filepath = data_root.joinpath(
        "toltec/ics/toltec0/toltec0_017596_000_0000_2023_05_02_19_09_42_vnasweep.nc",
    )

    # do not open, guess meta from filepath
    ncfile = NcFileIO(source=filepath, open=False)
    assert ncfile.meta["obsnum"] == 17596
    assert ncfile.meta["data_kind"] == ToltecDataKind.VnaSweep
    assert not ncfile.io_state.is_open()

    # now open the file
    with ncfile.open():
        assert ncfile.meta["obsnum"] == 17596
        assert ncfile.meta["data_kind"] == ToltecDataKind.VnaSweep
        assert ncfile.meta["n_chans"] == 1000
        assert ncfile.io_state.is_open()
    # closed
    assert not ncfile.io_state.is_open()


def test_ncfile():
    filepath = data_root.joinpath(
        "toltec/ics/toltec0/toltec0_017596_000_0000_2023_05_02_19_09_42_vnasweep.nc",
    )

    ncfile = NcFileIO(source=filepath, open=True)
    assert ncfile.meta["obsnum"] == 17596
    assert ncfile.meta["n_chans"] == 1000
    assert ncfile.io_state.is_open()
    ncfile.close()
    assert not ncfile.io_state.is_open()
    assert ncfile.file_loc_orig is not None
    assert ncfile.file_loc_orig.path == filepath
    assert ncfile.file_loc is not None
    assert ncfile.file_loc.path == filepath
    assert ncfile.file_obj is None


def test_ncfile_pre_opened():
    filepath = data_root.joinpath(
        "toltec/ics/toltec0/toltec0_017596_000_0000_2023_05_02_19_09_42_vnasweep.nc",
    )
    ds = netCDF4.Dataset(filepath)

    ncfile = NcFileIO(source=ds, open=True)
    assert ncfile.meta["obsnum"] == 17596
    assert ncfile.meta["n_chans"] == 1000
    assert ncfile.io_state.is_open()
    ncfile.close()
    assert not ncfile.io_state.is_open()
    assert ncfile.file_loc_orig is not None
    assert ncfile.file_loc_orig.path == filepath
    assert ncfile.file_loc is not None
    assert ncfile.file_loc.path == filepath
    # TODO: revisit this
    assert ncfile.file_obj is None


def test_ncfile_vnasweep_read():
    filepath = data_root.joinpath(
        "toltec/ics/toltec0/toltec0_017596_000_0000_2023_05_02_19_09_42_vnasweep.nc",
    )
    with NcFileIO(source=filepath) as ncfile:
        swp = ncfile.read()
    assert swp.meta["obsnum"] == 17596
    assert swp.frequency.shape == (1000, 491)


def test_ncfile_timestream_read():
    filepath = data_root.joinpath(
        "toltec/ics/toltec0/toltec0_018230_111_0000_2024_05_20_13_40_08_timestream.nc",
    )
    with NcFileIO(source=filepath) as ncfile:
        swp = ncfile.read()
    assert swp.meta["obsnum"] == 18230
    assert swp.I.shape == (649, 610)
    assert swp.r is None


def test_ncfile_nominal_read():
    filepath = data_root.joinpath(
        "toltec/tcs/toltec0/toltec0_131289_000_0002_2025_03_19_12_25_40.nc",
    )
    with NcFileIO(source=filepath) as ncfile:
        ts = ncfile.read()
    assert ts.meta["obsnum"] == 131289
    assert ts.meta["fsmp"] == 122.0703125


def test_ncfile_read_bad():
    filepath = data_root.joinpath(
        "toltec/tcs/toltec11/toltec11_130764_000_0001_2025_03_16_02_23_39_tune.nc",
    )
    with (
        pytest.raises(ValueError, match="invalid f_lo"),
        NcFileIO(source=filepath) as ncfile,
    ):
        ncfile.read()


def test_guess_info_table_read_with_bad():
    tbl_info = guess_info_from_sources(
        [
            data_root.joinpath(p)
            for p in [
                "toltec/tcs/toltec11/toltec11_130764_000_0001_2025_03_16_02_23_39_tune.nc",
            ]
        ],
    )
    with pytest.raises(ValueError, match="invalid f_lo"):
        tbl_info.toltec_file.read(raise_on_error=True)

    with tbl_info.toltec_file.open(raise_on_error=False):
        assert tbl_info.toltec_file.io_objs.iloc[0] is None


def test_guess_info_table_open_bad_metadata():
    tbl_info = guess_info_from_sources(
        [
            data_root.joinpath(p)
            for p in [
                "toltec/tcs/toltec0/toltec0_131289_000_0001_2025_03_19_12_24_21_tune.nc",
                "toltec/tcs/toltec0/toltec0_131289_000_0003_2025_03_19_12_25_40.nc",
            ]
        ],
    )
    with tbl_info.toltec_file.open(raise_on_error=False):
        assert tbl_info["instru"].to_list() == ["toltec", "toltec"]
