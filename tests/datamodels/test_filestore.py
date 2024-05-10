from pathlib import Path

import pytest
from tollan.utils.log import logger

from tolteca_datamodels.lmt.filestore import LmtFileStore
from tolteca_datamodels.toltec.filestore import ToltecFileStore

data_root = Path(__file__).with_name("data_lmt")

pytestmark = pytest.mark.skipif(
    not data_root.exists(),
    reason=f"test data {data_root} does not exist.",
)


def test_toltec_fs():
    path = data_root / "toltec"
    toltec_fs = ToltecFileStore(path=path)

    assert toltec_fs.path.samefile(path)
    ptbl = toltec_fs.get_path_info_table()
    logger.debug(f"path info table:\n{ptbl}")
    assert ptbl.query("master == 'ics'").to_dict(orient="records") == [
        {
            "master": "ics",
            "interface": "toltec0",
            "roach": 0,
            "path": toltec_fs.path / "ics/toltec0",
        },
    ]

    ltbl = toltec_fs.get_symlink_info_table()
    logger.debug(f"link info table:\n{ltbl}")
    assert ltbl.query("master.isnull()").to_dict(orient="records") == [
        {
            "master": None,
            "interface": "toltec0",
            "roach": 0,
            "path": toltec_fs.path / "toltec0.nc",
        },
        {
            "master": None,
            "interface": "toltec1",
            "roach": 1,
            "path": toltec_fs.path / "toltec1.nc",
        },
    ]

    assert set(toltec_fs.masters) == {"ics", "tcs"}
    assert set(toltec_fs.interfaces) == set(
        ["hwpr"] + [f"toltec{i}" for i in range(13)],
    )

    assert (
        toltec_fs.get_roach_path(master="ics", roach=10)
        == toltec_fs.path / "ics/toltec10"
    )


def test_lmt_fs():
    path = data_root
    lmt_fs = LmtFileStore(path=path)
    assert lmt_fs.toltec.get_roach_path("tcs", 11).samefile(
        path / "toltec/tcs/toltec11",
    )

    assert lmt_fs.path.samefile(path)
    ptbl = lmt_fs.get_path_info_table()
    logger.debug(f"path info table:\n{ptbl}")
    assert ptbl.query("interface == 'toltec'").to_dict(orient="records") == [
        {
            "interface": "toltec",
            "instru": "toltec",
            "path": lmt_fs.path / "toltec",
            "master": None,
            "roach": None,
        },
    ]

    ltbl = lmt_fs.get_symlink_info_table()
    logger.debug(f"link info table:\n{ltbl}")
    assert ltbl.query("master.isnull()").to_dict(orient="records") == [
        {
            "interface": "tel",
            "instru": None,
            "master": None,
            "roach": None,
            "path": lmt_fs.path / "tel/tel.nc",
        },
        {
            "interface": "toltec0",
            "instru": "toltec",
            "master": None,
            "roach": 0,
            "path": lmt_fs.path / "toltec/toltec0.nc",
        },
        {
            "interface": "toltec1",
            "instru": "toltec",
            "master": None,
            "roach": 1,
            "path": lmt_fs.path / "toltec/toltec1.nc",
        },
    ]

    assert set(lmt_fs.instruments) == {"toltec"}
    assert set(lmt_fs.interfaces) == {"toltec", "tel"}
