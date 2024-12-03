from pathlib import Path

import pytest
from astropy.time import Time

from tolteca_datamodels.lmt.tel import LmtTelFileIO

data_root = Path(__file__).with_name("data_lmt")

pytestmark = pytest.mark.skipif(
    not data_root.exists(),
    reason=f"test data {data_root} does not exist.",
)


def test_tel_no_open():
    filepath = data_root.joinpath("tel/tel_toltec_2023-06-03_110245_00_0001.nc")

    # do not open, guess meta from filepath
    ncfile = LmtTelFileIO(source=filepath, open=False)
    assert ncfile.meta.instru == "lmt"
    assert ncfile.meta.instru_component == "tcs"
    assert ncfile.meta.interface == "tel_toltec"
    assert ncfile.meta.obsnum == 110245
    assert not ncfile.io_data.is_open()

    # now open the file
    with ncfile.open():
        assert ncfile.meta.obsnum == 110245
        assert ncfile.meta.obs_goal == "calibration"
        assert ncfile.io_data.is_open()
    # closed
    assert not ncfile.io_data.is_open()


def test_tel_read():
    filepath = data_root.joinpath("tel/tel_toltec_2023-06-03_110245_00_0001.nc")
    with LmtTelFileIO(source=filepath) as ncfile:
        tel = ncfile.read()
    assert tel.meta.obsnum == 110245
    assert tel.meta.t0 == Time("2023-06-03T10:23:48.554")
