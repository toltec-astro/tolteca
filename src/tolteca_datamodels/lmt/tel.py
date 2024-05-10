from dataclasses import dataclass
from typing import ClassVar, Literal

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pydantic import BaseModel
from tollan.utils.nc import ncopen

from ..io.core import FileIO
from ..io.ncfile import NcFileIOData, NcFileIOMixin

ObsGoalType = Literal["science", "pointing", "focus", "astigmatism"]


@dataclass(kw_only=True)
class LmtTelIOMetadata:
    """Metadata model for LMT telescope file IO."""

    obs_goal: ObsGoalType


@dataclass(kw_only=True)
class LmtTelMetadata(LmtTelIOMetadata):
    """Metadata model for LMT telescope data."""


class LmtTelData(BaseModel):
    """LMT telescope data."""

    meta: LmtTelMetadata


class LmtTelFileIOData(NcFileIOData):
    """LMT telescope IO data."""

    @classmethod
    def validate_source(cls, source, **kwargs):
        """Validate source as LMT telescope file."""
        source = super().validate_source(source, **kwargs)
        attrs_to_check = [
            "Data.TelescopeBackend.TelTime",
        ]
        filepath = source.path
        try:
            with ncopen(filepath) as ds:
                if all(a not in ds.variables for a in attrs_to_check):
                    raise ValueError("invalid netCDF content.")
        except OSError as e:
            raise ValueError(f"unable to open file {filepath} as netCDF dataset") from e
        return source


class LmtTelFileIO(FileIO[LmtTelIOMetadata, LmtTelFileIOData], NcFileIOMixin):
    """The LMT telescope file IO."""

    def __init__(self, source, **kwargs):
        super().__init__(source, **kwargs)
        super()._init_node_mappers()

    _nc_node_map_defs: ClassVar[dict[str, dict]] = {
        "meta": {
            "mapping_type": "Header.Dcs.ObsPgm",
            "target_ra": "Header.Source.Ra",
            "target_dec": "Header.Source.Dec",
            "target_az": "Header.Source.Az",
            "target_alt": "Header.Source.El",
            "obs_goal": "Header.Dcs.ObsGoal",
            "mastervar": "Header.Toltec.Master",
            "repeatvar": "Header.Toltec.RepeatLevel",
            "time": "Data.TelescopeBackend.TelTime",
        },
        "read": {
            "time": "Data.TelescopeBackend.TelTime",
            "az": "Data.TelescopeBackend.TelAzAct",
            "alt": "Data.TelescopeBackend.TelElAct",
            "ra": "Data.TelescopeBackend.SourceRaAct",
            "dec": "Data.TelescopeBackend.SourceDecAct",
            "holdflag": "Data.TelescopeBackend.Hold",
        },
    }

    def _update_meta_from_io_data(self):
        if self.is_open():
            nm = self.node_mappers["meta"]
            m = {}
            m["mapping_type"] = nm.getstr("mapping_type")
            if self.hasvar("obs_goal"):
                m["obs_goal"] = nm.getstr("obs_goal").lower()
            else:
                m["obs_goal"] = None
            m["master"] = 0
            m["master_name"] = "tcs"
            m["interface"] = "lmt"

            # target info
            t_ra, t_ra_off = nm.getvar("target_ra")[:]
            t_dec, t_dec_off = nm.getvar("target_dec")[:]
            if nm.hasvar("target_az"):
                t_az, t_az_off = nm.getvar("target_az")[:]
                t_alt, t_alt_off = nm.getvar("target_alt")[:]
            else:
                # TODO: fix this in simu tel
                pass

            t0 = Time(self.getvar("time")[0], format="unix")
            t0.format = "isot"
            m["t0"] = t0
            m["target"] = SkyCoord(t_ra << u.rad, t_dec << u.rad, frame="icrs")
            m["target_off"] = SkyCoord(
                t_ra_off << u.rad,
                t_dec_off << u.rad,
                frame="icrs",
            )
            m["target_frame"] = "icrs"
            self.meta.__dict__.update(m)

    def read(self):
        """Return LMT telescope data object."""
        meta = self.meta
        nm = self.node_mappers["data"]
        t = nm.getvar("time")[:]
        t = (t - t[0]) << u.s
        ra = nm.getvar("ra")[:] << u.rad
        dec = nm.getvar("dec")[:] << u.rad
        az = nm.getvar("az")[:] << u.rad
        alt = nm.getvar("alt")[:] << u.rad
        holdflag = nm.getvar("holdflag")[:].astype(int)
        return LmtTelData(
            time=t,
            ra=ra,
            dec=dec,
            az=az,
            alt=alt,
            t0=meta["t0"],
            meta=meta,
            holdflag=holdflag,
            target=meta["target"],
            ref_frame="icrs",
        )
