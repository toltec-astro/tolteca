"""Recorded KIDs master codes and derived observation identifiers."""

import json

import pytest
import xarray as xr
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.toltec_db import ToltecDBRawObsMasterCode


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_recorded_master_is_preserved(data_lmt_path, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with xr.open_dataset(path) as raw:
        expected = raw["Header.Toltec.Master"].item()
    with open_toltec_kids(path) as raw:
        meta = raw.toltec_kids.meta
        assert meta.master_code is ToltecDBRawObsMasterCode.TCS
        assert meta.master_code == expected == 0
        assert meta.master == "tcs"
        assert type(meta.master) is str
        payload = meta.model_dump(mode="json")
        assert payload["master"] == "tcs"
        assert payload["master_code"] == expected
        assert payload["filepath_info_orig"]["master"] == "tcs"
        assert "master_str" not in payload
        assert "master_str" not in payload["filepath_info_orig"]
        assert "obs_type_code" in payload
        assert payload["obs_type"] == meta.obs_type
        quartet = f"tcs-{meta.obsnum}-{meta.subobsnum}-{meta.scannum}"
        assert meta.quartet == quartet
        assert meta.uid_raw_obs_file == f"{quartet}-toltec0"
        assert not raw["Data.Toltec.Is"].variable._in_memory
        adapter = TypeAdapter(type(meta))
        encoded = adapter.dump_json(meta, round_trip=True)
        fields = json.loads(encoded)
        assert fields["master_code"] == expected
        assert "mastervar" not in fields
        assert fields["master"] == "tcs"
        restored = adapter.validate_json(encoded)
        assert restored.master_code is meta.master_code
        assert restored.master == meta.master
        assert restored.uid_raw_obs_file == meta.uid_raw_obs_file


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("master", 0),
        ("master", True),
        ("master", None),
        ("master", "not-a-master"),
        ("master_code", 0),
        ("master_code", "tcs"),
        ("master_code", "0"),
        ("master_code", True),
        ("obs_type_code", "tune"),
        ("obs_type_code", "4"),
        ("obs_type_code", True),
    ],
)
def test_master_labels_and_codes_cannot_exchange_types(data_lmt_path, name, value):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc")
    with open_toltec_kids(path) as raw:
        meta = raw.toltec_kids.meta
    with pytest.raises(ValidationError, match=name):
        type(meta).model_validate(meta.model_dump(round_trip=True) | {name: value})
