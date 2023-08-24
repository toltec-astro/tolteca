from pathlib import Path

import netCDF4

from tolteca_datamodels.toltec.ncfile import NcFileIO
from tolteca_datamodels.toltec.types import ToltecDataKind

data_root = Path(__file__).with_name("data_lmt")


def test_ncfile_no_open():
    filepath = data_root.joinpath(
        "toltec/ics/toltec0/toltec0_017596_000_0000_2023_05_02_19_09_42_vnasweep.nc"
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
        "toltec/ics/toltec0/toltec0_017596_000_0000_2023_05_02_19_09_42_vnasweep.nc"
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
        "toltec/ics/toltec0/toltec0_017596_000_0000_2023_05_02_19_09_42_vnasweep.nc"
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
    # TODO revisit this
    assert ncfile.file_obj is None


def test_ncfile_read():
    filepath = data_root.joinpath(
        "toltec/ics/toltec0/toltec0_017596_000_0000_2023_05_02_19_09_42_vnasweep.nc"
    )
    with NcFileIO(source=filepath) as ncfile:
        swp = ncfile.read()
    assert swp.meta["obsnum"] == 17596
    assert swp.frequency.shape == (1000, 491)
