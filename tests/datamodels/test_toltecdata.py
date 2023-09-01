from pathlib import Path

from tolteca_datamodels.toltec import ToltecData, ToltecDataKind

data_root = Path(__file__).with_name("data_lmt")


def test_toltecdata():
    filepath = data_root.joinpath(
        "toltec/reduced/toltec1_110245_000_0001_2023_06_03_10_23_49_vnasweep.txt",
    )
    with ToltecData(source=filepath, data_kind="TableData") as tblfile:
        tbl = tblfile.read()
    assert tbl.meta["data_kind"] == ToltecDataKind.KidsModelParamsTable
