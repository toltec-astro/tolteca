"""Recorded KIDs master codes and derived observation identifiers."""

import dataclasses
import json

import pytest
import xarray as xr
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.toltec import open_toltec
from tolteca_datamodels.toltec.toltecdb import ToltecDBRawObsMaster


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_recorded_master_is_preserved(data_lmt_path, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with xr.open_dataset(path) as raw:
        expected = raw["Header.Toltec.Master"].item()
    with open_toltec(path) as raw:
        meta = raw.toltec_kids.meta
        assert isinstance(meta.master, ToltecDBRawObsMaster)
        assert meta.master == expected == 0
        assert meta.master_str == "tcs"
        assert meta.master_str is meta.__dict__["master_str"]
        quartet = f"tcs-{meta.obsnum}-{meta.subobsnum}-{meta.scannum}"
        assert meta.quartet == quartet
        assert meta.uid_raw_obs_file == f"{quartet}-toltec0"
        assert not raw["Data.Toltec.Is"].variable._in_memory
        adapter = TypeAdapter(type(meta))
        encoded = adapter.dump_json(meta, round_trip=True)
        fields = json.loads(encoded)
        assert fields["master"] == expected
        assert "mastervar" not in fields
        assert "master_str" not in fields
        restored = adapter.validate_json(encoded)
        assert restored.master == expected
        assert restored.master_str == "tcs"
        assert restored.uid_raw_obs_file == meta.uid_raw_obs_file


def test_master_names_are_not_raw_codes(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*_targsweep.nc")
    with open_toltec(path) as raw:
        meta = raw.toltec_kids.meta
    with pytest.raises(ValidationError, match="master"):
        dataclasses.replace(meta, master="tcs")
