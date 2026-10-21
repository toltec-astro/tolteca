"""LMT physical facts and telescope-owned interfaces."""

from zoneinfo import ZoneInfo

import astropy.units as u
import pytest
from astropy.coordinates import EarthLocation
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.lmt.types import (
    LmtInfo,
    LmtInterfaceInfo,
    LmtObsPgm,
    LmtRawObsMasterCode,
)
from tolteca_datamodels.lmt_base.metadata import InterfaceInfoMixin


def test_telescope_names():
    assert LmtInfo.instru == "lmt"
    assert LmtInfo.name == "LMT"
    assert LmtInfo.name_long == "Large Millimeter Telescope"


def test_instrument_literal_type():
    adapter = TypeAdapter(LmtInfo.InstruT)
    assert adapter.validate_python("lmt") == "lmt"
    assert adapter.validate_json(adapter.dump_json("lmt")) == "lmt"
    with pytest.raises(ValidationError, match="literal_error"):
        adapter.validate_python("other")


def test_site_constants():
    # Preserve the geodetic facts used by v1 and v2 observing scripts.
    assert LmtInfo.location == {
        "lon": "-97d18m52.6s",
        "lat": "+18d59m10s",
        "height": 4640 << u.m,
    }
    location = EarthLocation.from_geodetic(**LmtInfo.location)
    assert location.lon.deg == pytest.approx(-(97 + 18 / 60 + 52.6 / 3600))
    assert location.lat.deg == pytest.approx(18 + 59 / 60 + 10 / 3600)
    assert location.height.to_value(u.m) == pytest.approx(4640)
    assert ZoneInfo(LmtInfo.timezone_local).key == "America/Mexico_City"


def test_telescope_interfaces():
    assert LmtInfo.interface_interface_info == {
        "tel": LmtInterfaceInfo(
            interface="tel", data_models=["lmt"], instru="lmt", interface_kind="tel"
        )
    }
    assert LmtInfo.interfaces == ["tel"]


@pytest.mark.parametrize("data_models", [["lmt"], ["lmt", "toltec"]])
def test_telescope_interface_allows_overlapping_membership(data_models):
    info = LmtInterfaceInfo.model_validate(
        {
            "interface": "tel",
            "instru": "lmt",
            "interface_kind": "tel",
            "data_models": data_models,
        }
    )
    assert isinstance(info, InterfaceInfoMixin)
    assert info.data_models == data_models


@pytest.mark.parametrize(
    "field", ["instru", "data_models", "interface_kind", "interface"]
)
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "null"])
def test_telescope_interface_requires_shared_identity(field, missing):
    data = {
        "interface": "tel",
        "instru": "lmt",
        "interface_kind": "tel",
        "data_models": ["lmt"],
    }
    if missing:
        del data[field]
    else:
        data[field] = None
    with pytest.raises(ValidationError) as exc:
        LmtInterfaceInfo.model_validate(data)
    assert [error["loc"] for error in exc.value.errors()] == [(field,)]


def test_master_lookups():
    assert LmtRawObsMasterCode.TCS.value == 0
    assert LmtRawObsMasterCode.get_master(LmtRawObsMasterCode.TCS) == "tcs"
    assert LmtInfo.master_code_master == {0: "tcs"}
    assert type(LmtInfo.master_code_master[0]) is str
    assert LmtInfo.master_master_code == {"tcs": 0}
    assert LmtInfo.master_description == {"tcs": "Telescope control system"}
    with pytest.raises(KeyError):
        LmtInfo.master_code_master[1]


@pytest.mark.parametrize("code", [1, 2, 3, 99])
def test_unsupported_master_codes(code):
    with pytest.raises(ValueError):
        LmtRawObsMasterCode(code)


def test_master_code_json_roundtrip():
    member = LmtRawObsMasterCode.TCS
    adapter = TypeAdapter(type(member))
    assert adapter.dump_python(member, mode="json") == 0
    assert adapter.validate_json(adapter.dump_json(member)) is member


def test_master_label_validation_and_roundtrip():
    adapter = TypeAdapter(LmtInfo.MasterT)
    label = adapter.validate_python("tcs", strict=True)
    assert type(label) is str
    assert adapter.validate_json(adapter.dump_json(label), strict=True) == label
    for invalid in ("ics", "TCS", "", 0, None):
        with pytest.raises(ValidationError, match="literal_error"):
            adapter.validate_python(invalid, strict=True)


def test_predefined_identity_is_frozen():
    identity = LmtInfo.interface_interface_info["tel"]
    assert identity.interface == "tel"
    with pytest.raises(ValidationError, match="frozen"):
        setattr(identity, "interface_kind", "other")  # noqa: B010 -- Test frozen validation.


def test_observation_programs_cover_recorded_lmtmc_values(lmtmc_records):
    programs = {row["ObsPgm"] for row in lmtmc_records}
    assert programs
    for program in programs:
        assert LmtObsPgm(program.lower()).value == program.lower()
