"""LMT physical facts and telescope-owned interfaces."""

from zoneinfo import ZoneInfo

import astropy.units as u
import pytest
from astropy.coordinates import EarthLocation
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.lmt.types import (
    LmtInfo,
    LmtMaster,
    LmtObsPgm,
    LmtRawObsMasterCode,
)
from tolteca_datamodels.lmt_base.metadata import LmtInstruMixin


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
    assert LmtInfo.interface_instru == {
        "tel": LmtInstruMixin(interface="tel", instru="lmt", instru_component="tcs")
    }
    assert LmtInfo.interfaces == ["tel"]


def test_master_lookups():
    assert LmtRawObsMasterCode.TCS.value == 0
    assert LmtRawObsMasterCode.get_master(LmtRawObsMasterCode.TCS) is LmtMaster.tcs
    assert LmtInfo.master_code_master == {0: "tcs"}
    assert LmtInfo.master_code_master[0] is LmtMaster.tcs
    assert LmtInfo.master_master_code == {"tcs": 0}
    with pytest.raises(KeyError):
        LmtInfo.master_code_master[1]


@pytest.mark.parametrize("code", [1, 2, 3, 99])
def test_unsupported_master_codes(code):
    with pytest.raises(ValueError):
        LmtRawObsMasterCode(code)


@pytest.mark.parametrize(
    ("member", "value"), [(LmtRawObsMasterCode.TCS, 0), (LmtMaster.tcs, "tcs")]
)
def test_master_enum_json_roundtrip(member, value):
    adapter = TypeAdapter(type(member))
    assert adapter.dump_python(member, mode="json") == value
    assert adapter.validate_json(adapter.dump_json(member)) is member


def test_predefined_identity_is_frozen():
    identity = LmtInfo.interface_instru["tel"]
    assert identity.interface == "tel"
    with pytest.raises(ValidationError, match="frozen"):
        identity.instru_component = "other"


def test_observation_programs_cover_recorded_lmtmc_values(lmtmc_records):
    programs = {row["ObsPgm"] for row in lmtmc_records}
    assert programs
    for program in programs:
        assert LmtObsPgm(program.lower()).value == program.lower()
