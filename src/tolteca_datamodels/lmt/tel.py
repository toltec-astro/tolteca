import re
from datetime import datetime, timezone
from typing import Annotated, ClassVar, Literal

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pydantic import BaseModel
from pydantic.types import StringConstraints
from tollan.config.types import SkyCoordField, TimeField
from tollan.utils.general import dict_from_regex_match
from tollan.utils.nc import ncopen

from ..io.core import FileIOMetadataModelBase
from ..io.ncfile import NcFileIO, NcFileIOData

ObsGoalType = Literal[
    None,
    "engineering",
    "science",
    "calibration",
    "pointing",
    "focus",
    "astigmatism",
]

RE_LMT_TEL_FILE = re.compile(
    r"^(?P<interface>tel_toltec|tel_toltec2|wyatt)_"
    r"(?P<file_timestamp>\d{4}-\d{2}-\d{2})"
    r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
    r"(?:_(?P<file_suffix>[^\/.]+))?"
    r"(?P<file_ext>\..+)$",
)

_re_named_field_type_dispatcher = {
    "obsnum": int,
    "subobsnum": int,
    "scannum": int,
}


def _guess_meta_from_source(source):
    m = {"source": source}
    m.update(
        dict_from_regex_match(
            RE_LMT_TEL_FILE,
            source.path_orig.name,
            type_dispatcher=_re_named_field_type_dispatcher,
        ),
    )
    m["file_timestamp"] = datetime.strptime(m["file_timestamp"], "%Y-%m-%d").replace(
        tzinfo=timezone.utc,
    )
    return m


class LmtTelMetadata(FileIOMetadataModelBase):
    """Metadata model for LMT telescope file IO."""

    instru: Literal["lmt"] = "lmt"
    instru_component: Literal["tcs"] = "tcs"

    # filename componnts
    interface: None | Annotated[str, StringConstraints(to_lower=True)] = None
    obsnum: None | int = None
    subobsnum: None | int = None
    scannum: None | int = None
    file_timestamp: None | datetime = None
    file_suffix: None | str = None
    file_ext: None | Annotated[str, StringConstraints(to_lower=True)] = None

    # header info
    obs_goal: ObsGoalType = None
    t0: None | TimeField = None
    target: None | SkyCoordField = None
    target_off: None | SkyCoordField = None

    # data info


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


class LmtTelFileIO(NcFileIO[LmtTelFileIOData, LmtTelMetadata]):
    """The LMT telescope file IO."""

    _node_mapper_defs: ClassVar[dict[str, dict]] = {
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
        "data": {
            "time": "Data.TelescopeBackend.TelTime",
            "az": "Data.TelescopeBackend.TelAzAct",
            "alt": "Data.TelescopeBackend.TelElAct",
            "ra": "Data.TelescopeBackend.SourceRaAct",
            "dec": "Data.TelescopeBackend.SourceDecAct",
            "holdflag": "Data.TelescopeBackend.Hold",
        },
    }

    def _update_meta_from_io_data(self):
        self.meta.__dict__.update(_guess_meta_from_source(self.io_data.source))
        if self.is_open():
            nm = self.node_mappers["meta"]
            m = {}
            m["mapping_type"] = nm.get_str("mapping_type")
            if nm.has_var("obs_goal"):
                m["obs_goal"] = nm.get_str("obs_goal").lower()
            else:
                m["obs_goal"] = None
            m["master"] = 0
            m["master_name"] = "tcs"
            m["interface"] = "lmt"

            # target info
            t_ra, t_ra_off = nm.get_var("target_ra")[:]
            t_dec, t_dec_off = nm.get_var("target_dec")[:]
            if nm.has_var("target_az"):
                t_az, t_az_off = nm.get_var("target_az")[:]
                t_alt, t_alt_off = nm.get_var("target_alt")[:]
            else:
                # TODO: fix this in simu tel
                pass

            t0 = Time(nm.get_var("time")[0], format="unix")
            t0.format = "isot"
            m["t0"] = t0
            m["target"] = SkyCoord(t_ra << u.rad, t_dec << u.rad, frame="icrs")
            m["target_off"] = SkyCoord(
                t_ra_off << u.rad,
                t_dec_off << u.rad,
                frame="icrs",
            )
            self.meta.__dict__.update(m)

    def read(self):
        """Return LMT telescope data object."""
        if not self.is_open():
            self.open()
        meta = self.meta
        nm = self.node_mappers["data"]
        t = nm.get_var("time")[:]
        t = (t - t[0]) << u.s
        ra = nm.get_var("ra")[:] << u.rad
        dec = nm.get_var("dec")[:] << u.rad
        az = nm.get_var("az")[:] << u.rad
        alt = nm.get_var("alt")[:] << u.rad
        holdflag = nm.get_var("holdflag")[:].astype(int)
        return LmtTelData(
            time=t,
            ra=ra,
            dec=dec,
            az=az,
            alt=alt,
            t0=meta.t0,
            meta=meta.model_dump(),
            holdflag=holdflag,
            target=meta.target,
            ref_frame="icrs",
        )
