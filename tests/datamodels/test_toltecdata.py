from pathlib import Path

import pytest

from tolteca_datamodels.toltec import ToltecData, ToltecDataKind

data_root = Path(__file__).with_name("data_lmt")

pytestmark = pytest.mark.skipif(
    not data_root.exists(),
    reason=f"test data {data_root} does not exist.",
)


def test_toltecdata1():
    filepath = data_root.joinpath(
        "toltec/reduced/toltec1_110245_000_0001_2023_06_03_10_23_49_vnasweep.txt",
    )
    with ToltecData(source=filepath, data_kind="TableData") as tblfile:
        tbl = tblfile.read()
    assert tbl.meta["data_kind"] == ToltecDataKind.KidsModelParamsTable


def test_toltecdata_guess_kind():
    filepath = data_root.joinpath(
        "toltec/reduced/toltec1_110245_000_0001_2023_06_03_10_23_49_vnasweep.txt",
    )
    with ToltecData(source=filepath) as tblfile:
        tbl = tblfile.read()
    assert tbl.meta["data_kind"] == ToltecDataKind.KidsModelParamsTable
