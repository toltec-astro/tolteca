"""Filename identification for the synthetic instrument."""

import pytest

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_dummy_instru.filepath import (
    DummyInstruFilepathInfo,
    DummyInstruLmtFilepathInfo,
)
from tolteca_dummy_instru.types import DummyInstruInfo


@pytest.mark.parametrize("interface", DummyInstruInfo.interfaces)
def test_filename_prefix_is_interface(interface):
    info = FilepathInfo.from_source(f"{interface}-tcs-1-0-2-20251031T120000.nc")
    assert isinstance(info, DummyInstruFilepathInfo)
    assert info.prefix == info.interface == interface
    assert info.instru == "dummy_instru"
    assert info.channel == DummyInstruInfo.interface_channel[interface]
    assert (info.master, info.obsnum, info.subobsnum, info.scannum) == ("tcs", 1, 0, 2)


@pytest.mark.parametrize("prefix", ["di00", "chan04", "chan0", "chan-1"])
def test_unrecognized_prefix(prefix):
    assert (
        FilepathInfo.from_source(
            f"{prefix}-tcs-1-0-2-20251031T120000.nc", raise_on_error=False
        )
        is None
    )


def test_telescope_companion_keeps_telescope_identity(tmp_path):
    path = tmp_path / "tel" / "tel_dummy_instru_2025-10-31_000001_00_0002.nc"
    path.parent.mkdir()
    path.touch()
    assert LmtFileStore(tmp_path).tel.glob_obs_spec("1/tel") == [path]
    info = FilepathInfo.from_source(path)
    assert type(info) is DummyInstruLmtFilepathInfo
    assert info.prefix == "tel_dummy_instru"
    assert (info.interface, info.instru, info.interface_kind) == (
        "tel",
        "lmt",
        "tel",
    )
    assert info.data_models == ["lmt", "dummy_instru"]
    assert (info.master, info.obsnum, info.subobsnum, info.scannum) == ("tcs", 1, 0, 2)
