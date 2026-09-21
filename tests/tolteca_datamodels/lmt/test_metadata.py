"""Required telescope metadata, numeric identity, and serialization."""

from dataclasses import replace
from datetime import UTC, datetime

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.lmt import LmtObsGoalType, LmtTelMetadata
from tolteca_datamodels.lmt_base import LmtRawObsMetadata
from tolteca_datamodels.toltec.toltecdb import ToltecDBRawObsMaster


@pytest.fixture
def base_tel_fields():
    return {
        "obsnum": 145609,
        "subobsnum": 0,
        "scannum": 2,
        "master": ToltecDBRawObsMaster.TCS,
        "interface": "tel",
        "t0": Time("2025-10-31T09:27:56", format="isot", scale="utc"),
        "t1": Time("2025-10-31T09:28:56", format="isot", scale="utc"),
        "t_exp": 60.32 * u.s,
        "obs_pgm": "Lissajous",
        "obs_goal": "pointing",
        "target": SkyCoord(ra=1 * u.rad, dec=0.5 * u.rad, frame="icrs"),
        "target_off": SkyCoord(ra=1.1 * u.rad, dec=0.6 * u.rad, frame="icrs"),
        "filepath_orig": "/data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc",
    }


def test_telescope_identity(base_tel_fields):
    meta = LmtTelMetadata(**base_tel_fields)
    assert isinstance(meta, LmtRawObsMetadata)
    assert meta.master == 0
    assert isinstance(meta.master, int)
    assert meta.master is ToltecDBRawObsMaster.TCS
    assert meta.master_str == "tcs"
    assert (meta.instru, meta.instru_component, meta.interface) == ("lmt", "tcs", "tel")
    assert meta.obs_pgm == "Lissajous"
    assert meta.obs_goal == LmtObsGoalType.pointing


@pytest.mark.parametrize("master", ["tcs", 99])
def test_telescope_master_rejects_unknown_values(base_tel_fields, master):
    base_tel_fields["master"] = master
    with pytest.raises(ValidationError):
        LmtTelMetadata(**base_tel_fields)


@pytest.mark.parametrize("master", [None, *ToltecDBRawObsMaster])
def test_telescope_master_is_nullable(base_tel_fields, master):
    base_tel_fields["master"] = master
    meta = LmtTelMetadata(**base_tel_fields)
    assert meta.master is master
    assert meta.master_str == (
        None if master is None else ToltecDBRawObsMaster.get_master_type(master)
    )


@pytest.mark.parametrize("master", [None, *ToltecDBRawObsMaster])
def test_telescope_master_name_is_cached(base_tel_fields, master):
    base_tel_fields["master"] = master
    meta = LmtTelMetadata(**base_tel_fields)
    assert "master_str" not in meta.__dict__
    name = meta.master_str
    assert meta.__dict__["master_str"] is name
    assert meta.master_str is name
    assert TypeAdapter(LmtTelMetadata).dump_python(meta)["master_str"] == name

    replacement = replace(meta, master=ToltecDBRawObsMaster.TCS)
    assert "master_str" not in replacement.__dict__
    assert replacement.master_str == "tcs"
    assert meta.master_str is name


@pytest.mark.parametrize(
    ("name", "value"), [("instru", "toltec"), ("instru_component", "roach")]
)
def test_instrument_identity_is_constant(base_tel_fields, name, value):
    with pytest.raises(ValidationError, match=name):
        LmtTelMetadata(**base_tel_fields, **{name: value})


def test_other_telescope_interface(base_tel_fields):
    base_tel_fields["interface"] = "tel2"
    assert LmtTelMetadata(**base_tel_fields).interface == "tel2"


@pytest.mark.parametrize(
    "missing",
    ["master", "obs_pgm", "obs_goal", "target", "target_off", "filepath_orig"],
)
def test_required_metadata(base_tel_fields, missing):
    del base_tel_fields[missing]
    with pytest.raises(ValidationError, match=missing):
        LmtTelMetadata(**base_tel_fields)


@pytest.mark.parametrize(
    "field", ["obs_pgm", "obs_goal", "target", "target_off", "filepath_orig"]
)
def test_required_metadata_cannot_be_none(base_tel_fields, field):
    base_tel_fields[field] = None
    with pytest.raises(ValidationError, match=field):
        LmtTelMetadata(**base_tel_fields)


def test_empty_goal_is_not_unspecified(base_tel_fields):
    base_tel_fields["obs_goal"] = ""
    assert LmtTelMetadata(**base_tel_fields).obs_goal == ""


def test_frozen(base_tel_fields):
    meta = LmtTelMetadata(**base_tel_fields)
    with pytest.raises(AttributeError):
        meta.obsnum = 999  # ty: ignore[invalid-assignment]


def test_json_roundtrip(base_tel_fields):
    adapter = TypeAdapter(LmtTelMetadata)
    meta = LmtTelMetadata(**base_tel_fields)
    restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
    assert restored.master == 0
    assert restored.master_str == "tcs"
    assert restored.filepath_orig == meta.filepath_orig
    assert restored.filepath_info_orig == meta.filepath_info_orig
    assert restored.obs_pgm == meta.obs_pgm
    assert restored.obs_goal == meta.obs_goal
    assert restored.t0 == meta.t0
    assert restored.t1 == meta.t1
    assert restored.t_exp == meta.t_exp
    assert restored.target.separation(meta.target).rad == pytest.approx(0, abs=1e-14)
    assert restored.target_off.separation(meta.target_off).rad == pytest.approx(
        0, abs=1e-14
    )


def test_original_filename_is_cached_and_serialized(base_tel_fields):
    meta = LmtTelMetadata(**base_tel_fields)
    assert "filepath_info_orig" not in meta.__dict__
    parsed = meta.filepath_info_orig
    assert parsed is meta.filepath_info_orig
    assert parsed.ut == datetime(2025, 10, 31, tzinfo=UTC)
    assert parsed.interface == "tel_toltec"
    assert parsed.suffix is None
    assert parsed.ext == "nc"
    assert (parsed.obsnum, parsed.subobsnum, parsed.scannum) == (145609, 0, 2)
    payload = TypeAdapter(LmtTelMetadata).dump_python(meta, mode="json")
    assert payload["filepath_info_orig"] == parsed.model_dump(mode="json")
    assert payload["master_str"] == "tcs"
    assert not {"source", "file_timestamp", "file_suffix", "file_ext"} & payload.keys()


def test_roundtrip_serialization_does_not_parse_filename(base_tel_fields):
    base_tel_fields["filepath_orig"] = "unrecognized.nc"
    meta = LmtTelMetadata(**base_tel_fields)
    adapter = TypeAdapter(LmtTelMetadata)
    restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
    assert "filepath_info_orig" not in meta.__dict__
    assert restored.filepath_orig == "unrecognized.nc"
    with pytest.raises(ValueError):
        _ = restored.filepath_info_orig


@pytest.mark.parametrize("goal", list(LmtObsGoalType))
def test_goal_constants_remain_string_comparable(base_tel_fields, goal):
    base_tel_fields["obs_goal"] = goal
    meta = LmtTelMetadata(**base_tel_fields)
    assert meta.obs_goal == goal.value


def test_unlisted_goal_is_retained(base_tel_fields):
    base_tel_fields["obs_goal"] = "azscan"
    assert LmtTelMetadata(**base_tel_fields).obs_goal == "azscan"
