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

from .metadata import LmtTelMetadata
from .types import LmtInfo, LmtObsGoal, LmtObsPgm, LmtRawObsMasterCode

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
    master_code: Mapping = NameMapping("Header.TelescopeBackend.Master")
    obs_pgm: Mapping = NameMapping("Header.Dcs.ObsPgm")
    obs_goal: Mapping = NameMapping("Header.Dcs.ObsGoal")
    t_int: Mapping = NameMapping("Header.Dcs.IntegrationTime")
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
    ) -> LmtTelMetadata:
        """Read telescope observation metadata.

        Parameters
        ----------
        data_source : xarray.Dataset or xarray.DataArray
            Telescope data containing the mapped header and timestamp fields.

        Returns
        -------
        LmtTelMetadata
            Observation metadata. Start and end times use the first and last
            telescope timestamps; integration time uses the header value.

        Raises
        ------
        ValueError
            If timestamps are empty or not one-dimensional, or target
            coordinates do not contain main and reference entries.
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
        master_code = LmtRawObsMasterCode(self.get_scalar(ds, schema.master_code))
        meta = LmtTelMetadata(
            interface=self.interface,
            master_code=master_code.value,
            master=LmtInfo.master_code_master[master_code],
            obsnum=self.get_scalar(ds, schema.obsnum),
            subobsnum=self.get_scalar(ds, schema.subobsnum),
            scannum=self.get_scalar(ds, schema.scannum),
            filepath_orig=self.get_str(ds, schema.filepath_orig),
            obs_pgm=LmtObsPgm(self.get_str(ds, schema.obs_pgm).lower()),
            obs_goal=LmtObsGoal(self.get_str(ds, schema.obs_goal).lower()),
            t0=t0,
            t1=t1,
            # Header integration time is distinct from the TelTime span (t_elapsed).
            t_int=self.get_scalar(ds, schema.t_int) * u.s,
            target=targets[0],
            target_ref=targets[1],
        )
        logger.debug(
            "Read telescope metadata: uid_raw_obs_file={}, "
            "obs_pgm={}, obs_goal={}, filepath_orig={}",
            meta.uid_raw_obs_file,
            meta.obs_pgm,
            meta.obs_goal,
            meta.filepath_orig,
        )
        return meta


@xr.register_dataset_accessor("lmt_tel")
@xr.register_datatree_accessor("lmt_tel")
class LmtTelAccessor(XarrayAccessorBase[LmtTelIOMapper]):
    """Access typed telescope metadata and lazy pointing arrays.

    Parameters
    ----------
    data_source : xarray.Dataset or xarray.DataTree
        Telescope dataset, or a tree with telescope data at its root. Keep it
        open until the desired array slices have been loaded.
    mapper : LmtTelIOMapper, optional
        Field mapper. If omitted, resolve fields against the dataset.

    Examples
    --------
    >>> with open_lmt_tel(filepath) as ds:
    ...     meta = ds.lmt_tel.meta
    ...     az = ds.lmt_tel.az.isel(time=slice(0, 10)).load()
    """

    def _validate(self) -> None:
        """Reject sources without a mapped telescope timestamp variable."""
        if not self.mapper.has_arr(self.data_source, self.mapper.schema.tel_time):
            msg = "Not an LMT telescope dataset: missing tel_time variable"
            raise ValueError(msg)

    @functools.cached_property
    def meta(self) -> LmtTelMetadata:
        """Observation metadata."""
        return self.mapper.get_metadata(self.data_source)

    @functools.cached_property
    def t(self) -> xr.DataArray:
        """Elapsed seconds from the first sample."""
        tel_time = self.tel_time
        result = tel_time - tel_time.isel({tel_time.dims[0]: 0})
        result.attrs.update(units="s", long_name="Elapsed telescope time")
        return result

    @property
    def tel_time(self) -> xr.DataArray:
        """Telescope sample timestamps in Unix seconds."""
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
        """Telescope hold flags."""
        result = self.mapper.get_arr(self.data_source, self.mapper.schema.hold_flag)
        # The stored 'boolean' label is categorical, not a physical unit.
        return result.astype(int).u.unset()


def open_lmt_tel(filepath: str | Path | FileLoc, **kwargs) -> xr.Dataset:
    """Open an LMT telescope netCDF file.

    Parameters
    ----------
    filepath : str, pathlib.Path or FileLoc
        Local telescope file. Remote files must be staged before opening.
    **kwargs
        Additional arguments passed to `xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset
        Open dataset with the ``lmt_tel`` accessor. Use it as a context manager,
        or close it after loading the required arrays.

    Raises
    ------
    ValueError
        If the location is remote or the dataset lacks telescope timestamps.
    """
    source = fileloc(filepath)
    if source.is_remote():
        msg = "Remote telescope files must be staged locally before opening"
        raise ValueError(msg)
    ds = xr.open_dataset(source.path, **kwargs)
    try:
        # Accessor initialization resolves the schema and validates structure only.
        _ = ds.lmt_tel
    except BaseException:
        ds.close()
        raise
    return ds
