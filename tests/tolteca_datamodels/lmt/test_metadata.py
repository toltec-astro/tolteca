"""Required telescope metadata, numeric identity, and serialization."""

import json
import subprocess
import sys
from datetime import UTC, datetime
from typing import assert_type

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.lmt.metadata import (
    LmtTelInterfaceKindT,
    LmtTelInterfaceT,
    LmtTelMetadata,
)
from tolteca_datamodels.lmt.types import (
    LmtInfo,
    LmtObsGoal,
    LmtObsPgm,
    LmtRawObsMasterCode,
)
from tolteca_datamodels.lmt_base.metadata import LmtRawObsMetadata
from tolteca_datamodels.toltec.filepath import ToltecFilepathInfo


def test_original_filepath_uses_available_telescope_parsers(data_lmt_path):
    script = """
import sys
from tolteca_datamodels.lmt import open_lmt_tel
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo

assert 'tolteca_datamodels.toltec' not in sys.modules
with open_lmt_tel(sys.argv[1]) as ds:
    info = ds.lmt_tel.meta.filepath_info_orig
    assert type(info) is LmtFilepathInfo
    assert info.data_models == {'lmt'}

from tolteca_datamodels.toltec.filepath import ToltecLmtFilepathInfo
with open_lmt_tel(sys.argv[1]) as ds:
    info = ds.lmt_tel.meta.filepath_info_orig
    assert type(info) is ToltecLmtFilepathInfo
    assert info.data_models == {'lmt', 'toltec'}
"""
    source = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def base_tel_fields():
    return {
        "obsnum": 145609,
        "subobsnum": 0,
        "scannum": 2,
        "master": "tcs",
        "master_code": LmtRawObsMasterCode.TCS,
        "instru": "lmt",
        "interface": "tel",
        "interface_kind": "tel",
        "data_models": ["lmt", "toltec"],
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
    assert_type(meta.data_models, frozenset[str])
    assert_type(meta.instru, LmtInfo.InstruT)
    assert_type(meta.interface, LmtTelInterfaceT)
    assert_type(meta.interface_kind, LmtTelInterfaceKindT)
    assert isinstance(meta, LmtRawObsMetadata)
    assert_type(meta.master_code, LmtRawObsMasterCode)
    assert meta.master_code is LmtRawObsMasterCode.TCS
    assert_type(meta.master, LmtInfo.MasterT)
    assert meta.master == "tcs"
    assert type(meta.master) is str
    assert (meta.data_models, meta.instru, meta.interface_kind, meta.interface) == (
        {"lmt", "toltec"},
        "lmt",
        "tel",
        "tel",
    )
    assert "receiver" not in meta.model_dump()
    assert (
        LmtTelMetadata.model_fields["instru"].description
        == "Large Millimeter Telescope"
    )
    assert_type(meta.obs_pgm, LmtObsPgm)
    assert meta.obs_pgm == "lissajous"
    assert meta.obs_pgm is LmtObsPgm.lissajous
    assert_type(meta.obs_goal, LmtObsGoal)
    assert meta.obs_goal is LmtObsGoal.pointing


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
        setattr(meta, "master", "ics")  # noqa: B010 -- Test frozen validation.


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("instru", "toltec"),
        ("interface", "tel2"),
        ("interface_kind", "roach"),
        *[("master", value) for value in ("ics", "simu", None, 0, True, 0.5)],
        *[
            ("master_code", value)
            for value in (1, 2, 99, None, "tcs", "0", True, False, 0.0, 0.5)
        ],
    ],
)
@pytest.mark.parametrize("mode", ["python", "json"])
def test_telescope_identity_rejects_invalid_values(base_tel_fields, name, value, mode):
    if mode == "python":
        with pytest.raises(ValidationError, match=name):
            LmtTelMetadata(**(base_tel_fields | {name: value}))
    else:
        payload = json.loads(
            LmtTelMetadata(**base_tel_fields).model_dump_json(round_trip=True)
        )
        with pytest.raises(ValidationError, match=name):
            LmtTelMetadata.model_validate_json(json.dumps(payload | {name: value}))


@pytest.mark.parametrize("data_models", [["lmt", "toltec"], ["lmt", "dummy_instru"]])
def test_telescope_data_model_membership_is_variable(base_tel_fields, data_models):
    meta = LmtTelMetadata(**(base_tel_fields | {"data_models": data_models}))
    assert meta.data_models == frozenset(data_models)
    assert meta.instru == "lmt"


def test_python_master_code_requires_enum_member(base_tel_fields):
    with pytest.raises(ValidationError, match="master_code"):
        LmtTelMetadata(**(base_tel_fields | {"master_code": 0}))


@pytest.mark.parametrize(
    "missing",
    [
        "data_models",
        "instru",
        "interface",
        "interface_kind",
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
    "field",
    ["data_models", "obs_pgm", "obs_goal", "target", "target_ref", "filepath_orig"],
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


def test_json_roundtrip(base_tel_fields):
    adapter = TypeAdapter(LmtTelMetadata)
    meta = LmtTelMetadata(**base_tel_fields)
    restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
    assert restored.master_code is LmtRawObsMasterCode.TCS
    assert restored.master == "tcs"
    assert restored.uid_raw_obs_file == "tcs-145609-0-2-tel"
    assert restored.filepath_orig == meta.filepath_orig
    assert restored.filepath_info_orig == meta.filepath_info_orig
    assert restored.obs_pgm == meta.obs_pgm
    assert restored.obs_goal == meta.obs_goal
    assert restored.t0 == meta.t0
    assert restored.t1 == meta.t1
    assert restored.t_int == meta.t_int
    assert restored.t_elapsed == meta.t_elapsed
    payload = adapter.dump_python(meta, mode="json")
    assert "t_elapsed" in payload
    assert "target_ref" in payload and "target_off" not in payload
    inputs = adapter.dump_python(meta, round_trip=True)
    assert "t_elapsed" not in inputs
    assert restored.target.separation(meta.target).rad == pytest.approx(0, abs=1e-14)
    assert restored.target_ref.separation(meta.target_ref).rad == pytest.approx(
        0, abs=1e-14
    )


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_identity_json_schema(mode, resolve_schema_ref):
    schema = LmtTelMetadata.model_json_schema(mode=mode)
    for name, value in {
        "instru": "lmt",
        "interface": "tel",
        "interface_kind": "tel",
        "master": "tcs",
        "master_code": 0,
    }.items():
        assert name in schema["required"]
        field = schema["properties"][name]
        assert "default" not in field
        definition = resolve_schema_ref(schema, field)
        assert definition.get("enum", [definition.get("const")]) == [value]
        assert definition["type"] == ("integer" if name == "master_code" else "string")


def test_original_filename_values_and_serialized_provenance(base_tel_fields):
    meta = LmtTelMetadata(**base_tel_fields)
    parsed = meta.filepath_info_orig
    assert parsed is not None
    assert_type(parsed, FilepathInfo)
    assert isinstance(parsed, LmtFilepathInfo)
    assert parsed.ut == datetime(2025, 10, 31, tzinfo=UTC)
    assert parsed.interface == "tel"
    assert parsed.suffix is None
    assert parsed.ext == "nc"
    assert (parsed.obsnum, parsed.subobsnum, parsed.scannum) == (145609, 0, 2)
    payload = TypeAdapter(LmtTelMetadata).dump_python(meta, mode="json")
    assert payload["filepath_info_orig"] == parsed.model_dump(mode="json")
    assert payload["master"] == "tcs"
    assert not {"source", "file_timestamp", "file_suffix", "file_ext"} & payload.keys()


def test_original_filename_does_not_reclassify_header_metadata(base_tel_fields):
    # A deliberately inconsistent header tests independent filename parsing.
    base_tel_fields["filepath_orig"] = "toltec8_145609_000_0002_2025_10_31_09_27_57.nc"
    meta = LmtTelMetadata(**base_tel_fields)
    assert meta.filepath_info_orig == FilepathInfo.from_source(meta.filepath_orig)
    assert isinstance(meta.filepath_info_orig, ToltecFilepathInfo)
    assert meta.filepath_info_orig.interface == "toltec8"
    assert meta.interface == "tel"


def test_roundtrip_serialization_does_not_parse_filename(base_tel_fields):
    base_tel_fields["filepath_orig"] = "unrecognized.nc"
    meta = LmtTelMetadata(**base_tel_fields)
    adapter = TypeAdapter(LmtTelMetadata)
    restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
    assert "filepath_info_orig" not in meta.__dict__
    assert restored.filepath_orig == "unrecognized.nc"
    assert type(restored.filepath_info_orig) is FilepathInfo
    assert restored.filepath_info_orig.ext == "nc"
    assert restored.filepath_info_orig.data_models == frozenset()
    payload = adapter.dump_python(restored, mode="json")
    assert payload["filepath_orig"] == "unrecognized.nc"
    assert payload["filepath_info_orig"] == {"ext": "nc", "data_models": []}


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


def test_program_json_schema(resolve_schema_ref):
    schema = LmtTelMetadata.model_json_schema()
    assert "obs_pgm" in schema["required"]
    field = resolve_schema_ref(schema, schema["properties"]["obs_pgm"])
    assert field["type"] == "string"
    assert set(field["enum"]) == {program.value for program in LmtObsPgm}
