"""Lazy xarray access to LMT telescope netCDF files."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import astropy.units as u
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.time import Time
from tollan.accessor import NameMapping, Schema
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper
from tollan.utils.fileloc import fileloc
from tollan.utils.log import logger

from ..toltec.toltecdb import ToltecDBRawObsMaster
from .metadata import LmtTelMetadata

if TYPE_CHECKING:
    from pathlib import Path

    from tollan.accessor import Mapping
    from tollan.utils.fileloc import FileLoc

__all__ = ["LmtTelAccessor", "LmtTelIOMapper", "LmtTelIOSchema", "open_lmt_tel"]


@dataclass
class LmtTelIOSchema(Schema):
    """Map telescope metadata and pointing samples to netCDF fields."""

    obsnum: Mapping = NameMapping("Header.Dcs.ObsNum")
    subobsnum: Mapping = NameMapping("Header.Dcs.SubObsNum")
    scannum: Mapping = NameMapping("Header.Dcs.ScanNum")
    master: Mapping = NameMapping("Header.TelescopeBackend.Master")
    obs_pgm: Mapping = NameMapping("Header.Dcs.ObsPgm")
    obs_goal: Mapping = NameMapping("Header.Dcs.ObsGoal")
    t_exp: Mapping = NameMapping("Header.Dcs.IntegrationTime")
    filepath_orig: Mapping = NameMapping("Header.File.Name")

    # Header coordinates contain the main and reference positions, in that order.
    ra: Mapping = NameMapping.var("Header.Source.Ra")
    dec: Mapping = NameMapping.var("Header.Source.Dec")
    tel_time: Mapping = NameMapping.var("Data.TelescopeBackend.TelTime")

    # Actual and desired horizontal pointing; native El is canonicalized to alt.
    az_act: Mapping = NameMapping.var("Data.TelescopeBackend.TelAzAct")
    alt_act: Mapping = NameMapping.var("Data.TelescopeBackend.TelElAct")
    az_des: Mapping = NameMapping.var("Data.TelescopeBackend.TelAzDes")
    alt_des: Mapping = NameMapping.var("Data.TelescopeBackend.TelElDes")

    # Map-coordinate streams, distinct from actual/desired telescope pointing.
    az_map: Mapping = NameMapping.var("Data.TelescopeBackend.TelAzMap")
    alt_map: Mapping = NameMapping.var("Data.TelescopeBackend.TelElMap")

    # Actual equatorial samples, not the two-position header coordinates above.
    source_ra_act: Mapping = NameMapping.var("Data.TelescopeBackend.SourceRaAct")
    source_dec_act: Mapping = NameMapping.var("Data.TelescopeBackend.SourceDecAct")
    hold_flag: Mapping = NameMapping.var("Data.TelescopeBackend.Hold")


class LmtTelIOMapper(XarrayMapper[LmtTelIOSchema]):
    """Read telescope headers and pointing arrays using LmtTelIOSchema."""

    interface: ClassVar[str] = "tel"
    """Readout interface represented by this mapper."""

    def get_metadata(
        self,
        data_source: xr.Dataset | xr.DataArray,
        *,
        check_filepath_orig: bool = False,
    ) -> LmtTelMetadata:
        """Read observation headers and the first/last telescope timestamps.

        Pointing arrays are not loaded. The integration timer is retained even
        when zero; it need not equal the elapsed span between timestamps.
        The required two entries of each target-coordinate header are the main
        and reference positions, respectively, in radians. Header identifiers
        are authoritative; a missing master is represented by None.

        Set ``check_filepath_orig=True`` to compare observation identifiers and
        available master information with the recorded original path. A mismatch
        raises ValueError. The current source path is not used for metadata.
        """
        ds = data_source
        schema = self.schema
        tel_time = self.get_arr(ds, schema.tel_time)
        if tel_time.ndim != 1 or tel_time.size == 0:
            msg = "Telescope time must be a nonempty one-dimensional array"
            raise ValueError(msg)
        t0 = Time(tel_time.isel({tel_time.dims[0]: 0}).to_numpy().item(), format="unix")
        t1 = Time(
            tel_time.isel({tel_time.dims[0]: -1}).to_numpy().item(), format="unix"
        )
        t0.format = "isot"
        t1.format = "isot"

        ra = self.get_arr(ds, schema.ra)
        dec = self.get_arr(ds, schema.dec)
        if ra.shape != (2,) or dec.shape != (2,):
            msg = "Target RA and Dec must each contain two main/reference entries"
            raise ValueError(msg)
        targets = SkyCoord(
            ra.to_numpy() << u.rad,
            dec.to_numpy() << u.rad,
            frame="icrs",
        )
        meta = LmtTelMetadata(
            interface=self.interface,
            master=(
                ToltecDBRawObsMaster(self.get_scalar(ds, schema.master))
                if schema.master in self
                else None
            ),
            obsnum=self.get_scalar(ds, schema.obsnum),
            subobsnum=self.get_scalar(ds, schema.subobsnum),
            scannum=self.get_scalar(ds, schema.scannum),
            filepath_orig=self.get_str(ds, schema.filepath_orig),
            obs_pgm=self.get_str(ds, schema.obs_pgm),
            obs_goal=self.get_str(ds, schema.obs_goal).lower(),
            t0=t0,
            t1=t1,
            # Header integration time is distinct from the TelTime span (t_elapsed).
            t_exp=self.get_scalar(ds, schema.t_exp) * u.s,
            target=targets[0],
            target_ref=targets[1],
        )
        logger.debug(
            "Read telescope metadata: interface={}, master={}, obs={}-{}-{}, "
            "obs_pgm={}, filepath_orig={}",
            meta.interface,
            meta.master,
            meta.obsnum,
            meta.subobsnum,
            meta.scannum,
            meta.obs_pgm,
            meta.filepath_orig,
        )
        if check_filepath_orig:
            filepath_info_orig = meta.filepath_info_orig
            for name in ("obsnum", "subobsnum", "scannum", "master"):
                attribute = "master_str" if name == "master" else name
                value = getattr(meta, attribute)
                value_orig = getattr(filepath_info_orig, attribute)
                if value is not None and value_orig is not None and value != value_orig:
                    msg = (
                        f"Conflicting {name}: "
                        f"{self.get_name(getattr(schema, name))}="
                        f"{getattr(meta, name)!r}, "
                        f"filepath_orig={meta.filepath_orig!r} "
                        f"identifies {value_orig!r}"
                    )
                    raise ValueError(msg)
            logger.debug("Checked telescope identifiers against {}", meta.filepath_orig)
        return meta


@xr.register_dataset_accessor("lmt_tel")
@xr.register_datatree_accessor("lmt_tel")
class LmtTelAccessor(XarrayAccessorBase[LmtTelIOMapper]):
    """Access typed telescope metadata and lazy pointing arrays.

    Examples
    --------
    >>> with open_lmt_tel("tel_toltec_2025-10-31_145609_00_0002.nc") as ds:
    ...     meta = ds.lmt_tel.meta
    ...     az = ds.lmt_tel.az.isel(time=slice(0, 10)).load()

    Keep the dataset open until the desired array slices have been loaded.
    Metadata is cached on this accessor; it contains no open file handles.
    """

    @functools.cached_property
    def meta(self) -> LmtTelMetadata:
        """Observation metadata, including main and reference sky positions."""
        return self.mapper.get_metadata(self.data_source)

    @functools.cached_property
    def t(self) -> xr.DataArray:
        """Elapsed seconds from the first sample; computes on non-Dask inputs."""
        tel_time = self.tel_time
        result = tel_time - tel_time.isel({tel_time.dims[0]: 0})
        result.attrs.update(units="s", long_name="Elapsed telescope time")
        return result

    @property
    def tel_time(self) -> xr.DataArray:
        """Native Unix timestamps, available for slicing before loading."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.tel_time)

    @property
    def az(self) -> xr.DataArray:
        """Actual telescope azimuth in radians."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.az_act)

    @property
    def alt(self) -> xr.DataArray:
        """Actual telescope altitude in radians."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.alt_act)

    @property
    def az_des(self) -> xr.DataArray:
        """Desired telescope azimuth in radians."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.az_des)

    @property
    def alt_des(self) -> xr.DataArray:
        """Desired telescope altitude in radians."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.alt_des)

    @property
    def az_map(self) -> xr.DataArray:
        """Telescope map azimuth coordinate in radians."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.az_map)

    @property
    def alt_map(self) -> xr.DataArray:
        """Telescope map altitude coordinate in radians."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.alt_map)

    @property
    def ra(self) -> xr.DataArray:
        """Actual source right ascension in radians."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.source_ra_act)

    @property
    def dec(self) -> xr.DataArray:
        """Actual source declination in radians."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.source_dec_act)

    @functools.cached_property
    def hold_flag(self) -> xr.DataArray:
        """Integer hold flags; computes on non-Dask inputs."""
        result = self.mapper.get_arr(self.data_source, self.mapper.schema.hold_flag)
        # The stored 'boolean' label is categorical, not a physical unit.
        return result.astype(int).u.unset()


def open_lmt_tel(filepath: str | Path | FileLoc, **kwargs) -> xr.Dataset:
    """Open telescope netCDF data lazily with the lmt_tel accessor registered.

    Additional arguments are passed to xarray.open_dataset. Use the returned
    dataset as a context manager, or close it after loading the needed arrays.
    The location must be local; remote files must be staged before opening.
    """
    source = fileloc(filepath)
    if source.is_remote():
        msg = "Remote telescope files must be staged locally before opening"
        raise ValueError(msg)
    return xr.open_dataset(source.path, **kwargs)
