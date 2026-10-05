"""Required telescope metadata, numeric identity, and serialization."""

from datetime import UTC, datetime
from typing import Literal, assert_type

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.lmt import LmtFilepathInfo
from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tolteca_datamodels.lmt.types import (
    LmtInfo,
    LmtMaster,
    LmtObsGoal,
    LmtObsPgm,
    LmtRawObsMasterCode,
)
from tolteca_datamodels.lmt_base.metadata import LmtRawObsMetadata
from tolteca_datamodels.lmt_base.types import LabelT


@pytest.fixture
def base_tel_fields():
    return {
        "obsnum": 145609,
        "subobsnum": 0,
        "scannum": 2,
        "master": "tcs",
        "master_code": LmtRawObsMasterCode.TCS.value,
        "interface": "tel",
        "t0": Time("2025-10-31T09:27:56", format="isot", scale="utc"),
        "t1": Time("2025-10-31T09:28:56", format="isot", scale="utc"),
        "t_int": 60.32 * u.s,
        "obs_pgm": LmtObsPgm.lissajous,
        "obs_goal": LmtObsGoal.pointing,
        "target": SkyCoord(ra=1 * u.rad, dec=0.5 * u.rad, frame="icrs"),
        "target_ref": SkyCoord(ra=1.1 * u.rad, dec=0.6 * u.rad, frame="icrs"),
        "filepath_orig": "/data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc",
    }


def test_telescope_identity(base_tel_fields):
    meta = LmtTelMetadata(**base_tel_fields)
    assert_type(meta.instru, LmtInfo.InstruT)
    assert_type(meta.instru_component, Literal["tcs"])
    assert isinstance(meta, LmtRawObsMetadata)
    assert_type(meta.master_code, int)
    assert meta.master_code == LmtRawObsMasterCode.TCS.value
    assert type(meta.master_code) is int
    assert_type(meta.master, LabelT)
    assert meta.master == LmtMaster.tcs
    assert type(meta.master) is str
    assert (meta.instru, meta.instru_component, meta.interface) == ("lmt", "tcs", "tel")
    assert_type(meta.obs_pgm, LmtObsPgm)
    assert meta.obs_pgm == "lissajous"
    assert meta.obs_pgm is LmtObsPgm.lissajous
    assert_type(meta.obs_goal, LmtObsGoal)
    assert meta.obs_goal is LmtObsGoal.pointing


@pytest.mark.parametrize("master_code", [None, "tcs", "0", True, 0.5])
def test_telescope_master_code_requires_integer(base_tel_fields, master_code):
    base_tel_fields["master_code"] = master_code
    with pytest.raises(ValidationError):
        LmtTelMetadata(**base_tel_fields)


def test_telescope_identifiers_and_serialized_master(base_tel_fields):
    meta = LmtTelMetadata(**base_tel_fields)
    expected = "tcs-145609-0-2"
    assert meta.quartet == expected
    assert meta.uid_raw_obs == expected
    assert meta.uid_raw_obs_file == f"{expected}-tel"
    payload = TypeAdapter(LmtTelMetadata).dump_python(meta, mode="json")
    assert payload["master_code"] == 0
    assert type(payload["master_code"]) is int
    assert payload["master"] == "tcs"
    assert type(payload["master"]) is str
    assert payload["quartet"] == expected
    assert payload["uid_raw_obs_file"] == meta.uid_raw_obs_file


def test_telescope_master_label_is_a_frozen_input(base_tel_fields):
    meta = LmtTelMetadata(**base_tel_fields)
    payload = meta.model_dump(round_trip=True)
    assert payload["master"] == "tcs"
    assert type(meta).model_validate(payload).master == "tcs"
    with pytest.raises(ValidationError, match="frozen"):
        meta.master = "ics"


@pytest.mark.parametrize("master", [None, 0, True, 0.5])
def test_telescope_master_requires_label(base_tel_fields, master):
    with pytest.raises(ValidationError, match="master"):
        LmtTelMetadata.model_validate(base_tel_fields | {"master": master})


@pytest.mark.parametrize(
    ("name", "value"), [("instru", "toltec"), ("instru_component", "roach")]
)
def test_instrument_identity_is_constant(base_tel_fields, name, value):
    with pytest.raises(ValidationError, match=name):
        LmtTelMetadata(**base_tel_fields, **{name: value})


def test_other_telescope_interface(base_tel_fields):
    base_tel_fields["interface"] = "tel2"
    meta = LmtTelMetadata(**base_tel_fields)
    assert meta.interface == "tel2"
    assert meta.quartet == "tcs-145609-0-2"
    assert meta.uid_raw_obs_file == "tcs-145609-0-2-tel2"


@pytest.mark.parametrize(
    "missing",
    [
        "master",
        "master_code",
        "obs_pgm",
        "obs_goal",
        "target",
        "target_ref",
        "filepath_orig",
    ],
)
def test_required_metadata(base_tel_fields, missing):
    del base_tel_fields[missing]
    with pytest.raises(ValidationError, match=missing):
        LmtTelMetadata(**base_tel_fields)


@pytest.mark.parametrize(
    "field", ["obs_pgm", "obs_goal", "target", "target_ref", "filepath_orig"]
)
def test_required_metadata_cannot_be_none(base_tel_fields, field):
    base_tel_fields[field] = None
    with pytest.raises(ValidationError, match=field):
        LmtTelMetadata(**base_tel_fields)


@pytest.mark.parametrize("goal", ["", "azscan", "pointing"])
def test_python_goal_requires_enum_instance(base_tel_fields, goal):
    base_tel_fields["obs_goal"] = goal
    with pytest.raises(ValidationError, match="obs_goal"):
        LmtTelMetadata(**base_tel_fields)


@pytest.mark.parametrize("program", ["", "new-program", "lissajous"])
def test_python_program_requires_enum_instance(base_tel_fields, program):
    base_tel_fields["obs_pgm"] = program
    with pytest.raises(ValidationError, match="obs_pgm"):
        LmtTelMetadata(**base_tel_fields)


def test_frozen(base_tel_fields):
    meta = LmtTelMetadata(**base_tel_fields)
    with pytest.raises(ValidationError, match="frozen"):
        meta.obsnum = 999


def test_json_roundtrip(base_tel_fields):
    adapter = TypeAdapter(LmtTelMetadata)
    meta = LmtTelMetadata(**base_tel_fields)
    restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
    assert restored.master_code == LmtRawObsMasterCode.TCS.value
    assert type(restored.master_code) is int
    assert restored.master == LmtMaster.tcs
    assert type(restored.master) is str
    assert restored.uid_raw_obs_file == "tcs-145609-0-2-tel"
    assert restored.filepath_orig == meta.filepath_orig
    assert restored.filepath_info_orig == meta.filepath_info_orig
    assert restored.obs_pgm == meta.obs_pgm
    assert restored.obs_goal == meta.obs_goal
    assert restored.t0 == meta.t0
    assert restored.t1 == meta.t1
    assert restored.t_int == meta.t_int
    assert restored.target.separation(meta.target).rad == pytest.approx(0, abs=1e-14)
    assert restored.target_ref.separation(meta.target_ref).rad == pytest.approx(
        0, abs=1e-14
    )


def test_master_json_schema():
    schema = LmtTelMetadata.model_json_schema(mode="serialization")
    for name in ("master", "master_code"):
        assert name in schema["required"]
    assert schema["properties"]["master_code"]["type"] == "integer"
    assert schema["properties"]["master"]["$ref"] == "#/$defs/LabelT"
    assert schema["$defs"]["LabelT"]["type"] == "string"


def test_elapsed_time_is_derived_cached_and_serializable(base_tel_fields):
    meta = LmtTelMetadata(**base_tel_fields)
    assert "t_elapsed" not in meta.__dict__
    elapsed = meta.t_elapsed
    assert elapsed == (meta.t1 - meta.t0).to(u.s)
    assert elapsed.unit == u.s
    assert meta.t_elapsed is elapsed
    assert meta.t_int == base_tel_fields["t_int"]
    assert elapsed != meta.t_int

    adapter = TypeAdapter(LmtTelMetadata)
    payload = adapter.dump_python(meta, mode="json")
    assert "t_elapsed" in payload
    assert "target_ref" in payload and "target_off" not in payload
    inputs = adapter.dump_json(meta, round_trip=True)
    assert b'"t_elapsed"' not in inputs
    restored = adapter.validate_json(inputs)
    assert "t_elapsed" not in restored.__dict__
    assert restored.t_elapsed == elapsed

    replacement = type(meta).model_validate(
        meta.model_dump(round_trip=True) | {"t1": meta.t1 + 1 * u.s}
    )
    assert "t_elapsed" not in replacement.__dict__
    assert replacement.t_elapsed == (replacement.t1 - replacement.t0).to(u.s)
    assert replacement.t_elapsed != elapsed
    assert meta.t_elapsed is elapsed


def test_original_filename_is_cached_and_serialized(base_tel_fields):
    meta = LmtTelMetadata(**base_tel_fields)
    assert "filepath_info_orig" not in meta.__dict__
    parsed = meta.filepath_info_orig
    assert parsed is not None
    assert_type(parsed, LmtFilepathInfo)
    assert type(parsed) is LmtFilepathInfo
    assert parsed is meta.filepath_info_orig
    assert parsed.ut == datetime(2025, 10, 31, tzinfo=UTC)
    assert parsed.interface == "tel"
    assert parsed.suffix is None
    assert parsed.ext == "nc"
    assert (parsed.obsnum, parsed.subobsnum, parsed.scannum) == (145609, 0, 2)
    payload = TypeAdapter(LmtTelMetadata).dump_python(meta, mode="json")
    assert payload["filepath_info_orig"] == parsed.model_dump(mode="json")
    assert payload["master"] == "tcs"
    assert not {"source", "file_timestamp", "file_suffix", "file_ext"} & payload.keys()
    schema = LmtTelMetadata.model_json_schema(mode="serialization")
    assert schema["properties"]["filepath_info_orig"]["anyOf"] == [
        {"$ref": "#/$defs/LmtFilepathInfo"},
        {"type": "null"},
    ]


def test_toltec_filename_does_not_identify_telescope_metadata(base_tel_fields):
    base_tel_fields["filepath_orig"] = "toltec8_145609_000_0002_2025_10_31_09_27_57.nc"
    assert LmtTelMetadata(**base_tel_fields).filepath_info_orig is None


def test_roundtrip_serialization_does_not_parse_filename(base_tel_fields):
    base_tel_fields["filepath_orig"] = "unrecognized.nc"
    meta = LmtTelMetadata(**base_tel_fields)
    adapter = TypeAdapter(LmtTelMetadata)
    restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
    assert "filepath_info_orig" not in meta.__dict__
    assert restored.filepath_orig == "unrecognized.nc"
    assert restored.filepath_info_orig is None
    payload = adapter.dump_python(restored, mode="json")
    assert payload["filepath_orig"] == "unrecognized.nc"
    assert payload["filepath_info_orig"] is None


@pytest.mark.parametrize("goal", list(LmtObsGoal))
def test_goal_constants_remain_string_comparable(base_tel_fields, goal):
    base_tel_fields["obs_goal"] = goal
    meta = LmtTelMetadata(**base_tel_fields)
    assert meta.obs_goal is goal
    assert meta.obs_goal == goal.value


@pytest.mark.parametrize("goal", list(LmtObsGoal))
def test_goal_json_roundtrip_retains_enum(base_tel_fields, goal):
    base_tel_fields["obs_goal"] = goal
    meta = LmtTelMetadata(**base_tel_fields)
    restored = LmtTelMetadata.model_validate_json(meta.model_dump_json(round_trip=True))
    assert restored.obs_goal is goal


@pytest.mark.parametrize("program", list(LmtObsPgm))
def test_program_json_roundtrip_retains_enum(base_tel_fields, program):
    base_tel_fields["obs_pgm"] = program
    meta = LmtTelMetadata(**base_tel_fields)
    assert meta.obs_pgm is program
    assert meta.obs_pgm == program.value
    assert meta.model_dump(mode="json")["obs_pgm"] == program.value
    restored = LmtTelMetadata.model_validate_json(meta.model_dump_json(round_trip=True))
    assert restored.obs_pgm is program


def test_program_json_schema():
    schema = LmtTelMetadata.model_json_schema()
    assert "obs_pgm" in schema["required"]
    assert schema["properties"]["obs_pgm"]["$ref"] == "#/$defs/LmtObsPgm"
    assert schema["$defs"]["LmtObsPgm"]["enum"] == [
        program.value for program in LmtObsPgm
    ]
