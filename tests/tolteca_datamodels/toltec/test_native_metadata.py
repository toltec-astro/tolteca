"""Native repeat codes and registered ROACH instrument identity."""

import json

import pytest
from pydantic import TypeAdapter

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata
from tolteca_datamodels.toltec.types import ToltecInfo


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_native_repeat_and_instrument_component(data_lmt_path, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with open_toltec_kids(path) as data:
        meta = data.toltec_kids.meta
        assert isinstance(meta, ToltecRawObsMetadata)
        repeat = data["Header.Toltec.RepeatLevel"].item()
        assert type(meta.repeat) is int
        assert meta.repeat == repeat == 0
        identity = ToltecInfo.interface_interface_info[meta.interface]
        assert meta.instru == identity.instru == "toltec"
        assert meta.interface_kind == identity.interface_kind == "roach"
        assert not data["Data.Toltec.Is"].variable._in_memory

    adapter = TypeAdapter(type(meta))
    encoded = adapter.dump_json(meta, round_trip=True)
    fields = json.loads(encoded)
    assert fields["repeat"] == 0
    assert "repeatvar" not in fields
    restored = adapter.validate_json(encoded)
    assert restored.repeat == repeat
    assert restored.interface_kind == "roach"


def test_missing_repeat_is_not_fabricated(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc")
    with open_toltec_kids(path) as data:
        missing = data.drop_vars("Header.Toltec.RepeatLevel")
        with pytest.raises(ValueError, match="Header.Toltec.RepeatLevel"):
            _ = missing.toltec_kids.meta
