import numpy as np
from tollan.utils.fileloc import FileLoc

from tolteca_datamodels.toltec.file import (
    guess_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_guess_info_tel():
    info = guess_info_from_source("tel_toltec_2024-04-18_117054_00_0001.nc")
    assert info.source == FileLoc("tel_toltec_2024-04-18_117054_00_0001.nc")
    assert info.instru == "lmt"
    assert info.instru_component == "tcs"
    assert info.interface == "tel_toltec"
    assert info.roach is None
    assert info.obsnum == 117054
    assert info.subobsnum == 0
    assert info.scannum == 1
    assert info.file_timestamp.isoformat() == "2024-04-18T00:00:00+00:00"
    assert info.file_suffix is None
    assert info.file_ext == ".nc"
    assert info.data_kind == ToltecDataKind.LmtTel


def test_guess_info_roach():
    info = guess_info_from_source(
        "toltec1_117062_002_0001_2024_04_18_03_53_43_targsweep.nc",
    )
    assert info.source == FileLoc(
        "toltec1_117062_002_0001_2024_04_18_03_53_43_targsweep.nc",
    )
    assert info.instru == "toltec"
    assert info.instru_component == "roach"
    assert info.interface == "toltec1"
    assert info.roach == 1
    assert info.obsnum == 117062
    assert info.subobsnum == 2
    assert info.scannum == 1
    assert info.file_timestamp.isoformat() == "2024-04-18T03:53:43+00:00"
    assert info.file_suffix == "targsweep"
    assert info.file_ext == ".nc"
    assert info.data_kind == ToltecDataKind.TargetSweep


def test_guess_info_table():
    tbl_info = guess_info_from_sources(
        [
            "tel_toltec_2024-04-18_117054_00_0001.nc",
            "toltec1_117062_002_0001_2024_04_18_03_53_43_targsweep.nc",
            "toltec11_117062_003_0001_2024_04_18_03_54_58_targsweep.nc",
        ],
    )
    assert len(tbl_info) == 3
    assert tbl_info["interface"].to_list() == ["tel_toltec", "toltec1", "toltec11"]

    g_obs = tbl_info.toltec_file.make_obs_groups()
    assert len(g_obs) == 2
    np.testing.assert_equal(
        g_obs.groups,
        {
            "117054": np.r_[0],
            "117062": np.r_[1, 2],
        },
    )

    g_raw_obs = tbl_info.toltec_file.make_raw_obs_groups()
    assert len(g_raw_obs) == 3
    np.testing.assert_equal(
        g_raw_obs.groups,
        {
            "117054-0-1": np.r_[0],
            "117062-2-1": np.r_[1],
            "117062-3-1": np.r_[2],
        },
    )


def test_guess_info_reduced_kids():
    info = guess_info_from_source(
        "toltec1_117062_002_0001_2024_04_18_03_53_43_targsweep_kids_find.ecsv",
    )
    assert info.instru == "toltec"
    assert info.instru_component == "roach"
    assert info.interface == "toltec1"
    assert info.roach == 1
    assert info.obsnum == 117062
    assert info.subobsnum == 2
    assert info.scannum == 1
    assert info.file_timestamp.isoformat() == "2024-04-18T03:53:43+00:00"
    assert info.file_suffix == "targsweep_kids_find"
    assert info.file_ext == ".ecsv"
    assert info.data_kind == ToltecDataKind.KidsPropTable
