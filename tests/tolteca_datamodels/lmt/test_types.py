"""LMT physical facts and telescope-owned interfaces."""

from dataclasses import FrozenInstanceError
from zoneinfo import ZoneInfo

import astropy.units as u
import pytest
from astropy.coordinates import EarthLocation

from tolteca_datamodels.lmt.types import LmtInfo
from tolteca_datamodels.lmt_base.metadata import LmtInstruMixin


def test_telescope_names():
    assert LmtInfo.instru == "lmt"
    assert LmtInfo.name == "LMT"
    assert LmtInfo.name_long == "Large Millimeter Telescope"


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


def test_predefined_identity_is_frozen():
    identity = LmtInfo.interface_instru["tel"]
    assert identity.interface == "tel"
    with pytest.raises(FrozenInstanceError):
        identity.instru_component = "other"  # ty: ignore[invalid-assignment]
