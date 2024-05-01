from pathlib import Path

import numpy as np
import pytest

from tolteca_datamodels.base import FileIOError
from tolteca_datamodels.toltec.table import TableIO
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kidsproc.kidsmodel import KidsSweepGainWithLinTrend

data_root = Path(__file__).with_name("data_lmt")

pytestmark = pytest.mark.skipif(
    not data_root.exists(),
    reason=f"test data {data_root} does not exist.",
)


def test_table_no_open():
    filepath = data_root.joinpath(
        "toltec/reduced/toltec1_110245_000_0001_2023_06_03_10_23_49_vnasweep_tonelist.ecsv",
    )

    # do not open, guess meta from filepath
    tblfile = TableIO(source=filepath, open=False)
    assert tblfile.meta["obsnum"] == 110245
    assert tblfile.meta["data_kind"] == ToltecDataKind.KidsPropTable
    assert not tblfile.io_state.is_open()

    # now open the file
    with tblfile.open():
        assert tblfile.meta["obsnum"] == 110245
        assert tblfile.meta["data_kind"] == ToltecDataKind.KidsPropTable
        assert tblfile.meta["n_chans"] == 642
        assert tblfile.meta["n_tones"] == 479
        assert tblfile.io_state.is_open()
    # closed
    assert not tblfile.io_state.is_open()


def test_table_read():
    filepath = data_root.joinpath(
        "toltec/reduced/toltec1_110245_000_0001_2023_06_03_10_23_49_vnasweep_tonelist.ecsv",
    )
    with TableIO(source=filepath) as tblfile:
        tbl = tblfile.read()
    assert tbl.meta["data_kind"] == ToltecDataKind.KidsPropTable
    assert tbl.meta["obsnum"] == 110245
    assert tbl["model_id"][0] == 0


def test_table_read_kids_params():
    filepath = data_root.joinpath(
        "toltec/reduced/toltec1_110245_000_0001_2023_06_03_10_23_49_vnasweep.txt",
    )
    with TableIO(source=filepath) as tblfile:
        tbl = tblfile.read()
    assert tbl.meta["data_kind"] == ToltecDataKind.KidsModelParamsTable
    assert tbl.meta["obsnum"] == 110245
    assert tbl.model_cls is KidsSweepGainWithLinTrend
    assert np.allclose(
        tbl.get_model(0)(tbl["fr"][0]),
        -89127.07902014 + 141685.03758128j,
    )


def test_table_read_data_kind():
    filepath = data_root.joinpath(
        "toltec/reduced/toltec1_110245_000_0001_2023_06_03_10_23_49_vnasweep.txt",
    )
    with TableIO(source=filepath, data_kind="TableData") as tblfile:
        tbl = tblfile.read()
    assert tbl.meta["data_kind"] == ToltecDataKind.KidsModelParamsTable

    with pytest.raises(FileIOError, match="is not a valid"):
        tblfile = TableIO(source=filepath, data_kind="ReducedSweep")
