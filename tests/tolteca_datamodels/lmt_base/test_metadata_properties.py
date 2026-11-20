"""Shared time and acquisition-path properties on frozen metadata."""

from typing import assert_type

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tolteca_datamodels.lmt.types import (
    LmtInterfaceInfo,
    LmtObsGoal,
    LmtObsPgm,
    LmtRawObsMasterCode,
)
from tolteca_datamodels.lmt_base.filepath import FilepathOrigMetadataMixin
from tolteca_datamodels.lmt_base.metadata import LmtRawObsMetadata
from tolteca_datamodels.toltec.filepath import ToltecFilepathInfo
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata
from tolteca_datamodels.toltec.toltec_db import ToltecDBRawObsMasterCode


@pytest.fixture
def meta():
    return LmtRawObsMetadata(
        master="tcs",
        obsnum=145609,
        subobsnum=0,
        scannum=2,
        data_models=frozenset({"lmt", "toltec"}),
        instru="lmt",
        interface="tel",
        interface_kind="tel",
        t0=Time("2025-10-31T09:27:56"),
        t1=Time("2025-10-31T09:28:56"),
        t_int=0 * u.s,
    )


@pytest.mark.parametrize("integration_s", [0, 60.32])
def test_elapsed_uses_endpoints_independently_of_integration_time(meta, integration_s):
    meta = type(meta).model_validate(
        meta.model_dump(round_trip=True) | {"t_int": integration_s * u.s}
    )
    elapsed = meta.t_elapsed
    assert elapsed == (meta.t1 - meta.t0).to(u.s)
    assert elapsed.unit == u.s
    assert meta.t_int == integration_s * u.s
    assert elapsed != meta.t_int
    other = type(meta).model_validate(
        meta.model_dump(round_trip=True) | {"t1": meta.t1 + 1 * u.s}
    )
    assert other.t_elapsed == (other.t1 - other.t0).to(u.s)
    assert other.t_elapsed > elapsed
    assert meta.t_elapsed == elapsed


def test_computed_fields_are_readonly_and_not_roundtrip_inputs(meta):
    adapter = TypeAdapter(LmtRawObsMetadata)
    payload = adapter.dump_json(meta, round_trip=True)
    assert "t_elapsed" not in meta.__dict__
    assert "filepath_info_orig" not in meta.__dict__
    assert b'"t_int"' in payload
    assert b'"t_exp"' not in payload
    restored = adapter.validate_json(payload)
    assert restored.t_elapsed == meta.t_elapsed
    assert not hasattr(restored, "filepath_orig")
    assert not hasattr(restored, "filepath_info_orig")
    schema = adapter.json_schema(mode="serialization")["properties"]
    assert "filepath_orig" not in schema
    assert "filepath_info_orig" not in schema
    assert schema["t_elapsed"]["readOnly"]
    assert "t_elapsed" not in adapter.json_schema()["properties"]
    # Integration time remains a required input under its public name.
    with pytest.raises(ValidationError, match="t_int"):
        adapter.validate_json(payload.replace(b'"t_int"', b'"t_exp"'))


def test_toltec_original_path_retains_subclass_fields(meta):
    metadata_data = TypeAdapter(LmtRawObsMetadata).dump_python(meta, round_trip=True)
    metadata_data.update(
        master=ToltecDBRawObsMasterCode.get_master(ToltecDBRawObsMasterCode.TCS),
        master_code=ToltecDBRawObsMasterCode.TCS,
        data_models=frozenset({"toltec"}),
        instru="toltec",
        interface="toltec8",
        readout_code=8,
        roach=8,
        nw=8,
        array=1,
        array_name="a1400",
        filepath_orig=(
            "/data_lmt/toltec/tcs/toltec8/"
            "toltec8_145609_000_0002_2025_10_31_09_27_57.nc"
        ),
    )
    meta = ToltecRawObsMetadata(
        **metadata_data, n_blocks_max=1, n_times=1000, n_chans=256, f_smp=500 * u.Hz
    )
    parsed = meta.filepath_info_orig
    assert parsed is not None
    assert_type(parsed, FilepathInfo)
    assert type(parsed) is ToltecFilepathInfo
    assert parsed.roach == 8
    expected = parsed.model_dump(mode="json")
    assert expected["roach"] == 8
    assert TypeAdapter(FilepathInfo).dump_python(parsed, mode="json") == expected
    for cls in (FilepathOrigMetadataMixin, ToltecRawObsMetadata):
        # The common computed field preserves the domain-specific record.
        payload = TypeAdapter(cls).dump_python(meta, mode="json")
        assert payload["filepath_info_orig"] == expected
    assert "filepath_orig" not in TypeAdapter(LmtRawObsMetadata).dump_python(meta)
    adapter = TypeAdapter(ToltecRawObsMetadata)
    restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
    assert restored.filepath_info_orig == parsed
    assert restored.t_elapsed == meta.t_elapsed
    for change in ({}, {"filepath_orig": None}):
        data = meta.model_dump(round_trip=True, exclude={"filepath_orig"}) | change
        with pytest.raises(ValidationError, match="filepath_orig"):
            type(meta).model_validate(data)


@pytest.mark.parametrize(
    ("model_cls", "additional_fields", "frozen_fields"),
    [
        (LmtInterfaceInfo, {}, ("interface",)),
        (LmtRawObsMetadata, {}, ("obsnum",)),
        (
            LmtTelMetadata,
            {
                "master_code": LmtRawObsMasterCode.TCS,
                "obs_pgm": LmtObsPgm.lissajous,
                "obs_goal": LmtObsGoal.pointing,
                "target": SkyCoord(ra=1 * u.rad, dec=0.5 * u.rad),
                "target_ref": SkyCoord(ra=1.1 * u.rad, dec=0.6 * u.rad),
                "filepath_orig": "tel_toltec_2025-10-31_145609_00_0002.nc",
            },
            ("obsnum",),
        ),
        (
            ToltecRawObsMetadata,
            {
                "master_code": ToltecDBRawObsMasterCode.TCS,
                "data_models": frozenset({"toltec"}),
                "instru": "toltec",
                "interface": "toltec8",
                "interface_kind": "roach",
                "readout_code": 8,
                "roach": 8,
                "nw": 8,
                "array": 1,
                "array_name": "a1400",
                "filepath_orig": "toltec8_145609_000_0002_2025_10_31_09_27_57.nc",
                "n_blocks_max": 1,
                "n_times": 1000,
                "n_chans": 256,
                "f_smp": 500 * u.Hz,
            },
            ("obsnum", "roach"),
        ),
    ],
    ids=["interface", "acquisition", "telescope", "readout"],
)
def test_metadata_boundaries_keep_declared_fields_frozen(
    meta, model_cls, additional_fields, frozen_fields
):
    values = meta.model_dump(round_trip=True) | additional_fields
    model = model_cls.model_validate(
        {
            name: value
            for name, value in values.items()
            if name in model_cls.model_fields
        }
    )
    for name in frozen_fields:
        original = getattr(model, name)
        with pytest.raises(ValidationError, match="frozen"):
            setattr(model, name, original)
        assert getattr(model, name) == original
