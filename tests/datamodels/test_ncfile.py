from tolteca_datamodels.toltec.ncfile import NcFileIO
from tolteca_datamodels.toltec.types import ToltecDataKind
from pathlib import Path

data_root = Path(__file__).with_name("data_lmt")


def test_ncfile_no_open():
    filepath = data_root.joinpath(
        "toltec/ics/toltec0/toltec0_017596_000_0000_2023_05_02_19_09_42_vnasweep.nc"
    )

    # do not open, guess meta from filepath
    ncfile = NcFileIO(source=filepath, open=False)
    assert ncfile.meta["obsnum"] == 17596
    assert ncfile.meta["data_kind"] == ToltecDataKind.VnaSweep
    assert ncfile.io_obj is None

    # now open the file
    with ncfile.open():
        assert ncfile.meta["obsnum"] == 17596
        assert ncfile.meta["data_kind"] == ToltecDataKind.VnaSweep
        assert ncfile.meta["n_chans"] == 1000
        assert ncfile.io_obj is not None
    # closed
    assert ncfile.io_obj is None


def test_ncfile():
    filepath = data_root.joinpath(
        "toltec/ics/toltec0/toltec0_017596_000_0000_2023_05_02_19_09_42_vnasweep.nc"
    )

    ncfile = NcFileIO(source=filepath, open=True)
    assert ncfile.meta["obsnum"] == 17596
    assert ncfile.meta["n_chans"] == 1000
    assert ncfile.io_obj is not None
    ncfile.close()
    assert ncfile.io_obj is None
    assert ncfile.file_loc_orig is not None
    assert ncfile.file_loc_orig.path == filepath
    assert ncfile.file_loc is not None
    assert ncfile.file_loc.path == filepath
