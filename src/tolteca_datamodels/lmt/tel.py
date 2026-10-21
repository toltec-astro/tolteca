"""Lazy xarray access to LMT telescope netCDF files."""

from __future__ import annotations

import functools
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import astropy.units as u
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.time import Time
from tollan.accessor import NameMapping, Schema
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper, unwrap_data_source
from tollan.utils.log import logger
from tollan.utils.typing import get_literal_value

from ..core.source.opener import _open_netcdf
from .metadata import LmtTelInterfaceKindT, LmtTelInterfaceT, LmtTelMetadata
from .types import (
    LmtDataModel,
    LmtInfo,
    LmtInterfaceInfo,
    LmtObsGoal,
    LmtObsPgm,
    LmtRawObsMasterCode,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tollan.accessor import Mapping
    from tollan.utils.fileloc import FileLoc

    class LmtTelDataset(xr.Dataset):
        """Dataset with a typed telescope accessor."""

        lmt_tel: LmtTelAccessor

    class LmtTelDataTree(xr.DataTree):
        """DataTree with a typed telescope accessor."""

        lmt_tel: LmtTelAccessor


__all__ = [
    "LmtTelAccessor",
    "LmtTelIOMapper",
    "LmtTelIOSchema",
    "open_lmt_tel",
]


@dataclass
class LmtTelIOSchema(Schema):
    """Map telescope metadata and pointing samples to netCDF fields."""

    obsnum: Mapping = NameMapping("Header.Dcs.ObsNum")
    subobsnum: Mapping = NameMapping("Header.Dcs.SubObsNum")
    scannum: Mapping = NameMapping("Header.Dcs.ScanNum")
    master_code: Mapping = NameMapping("Header.TelescopeBackend.Master")
    receiver: Mapping = NameMapping("Header.Dcs.Receiver")
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
            instru=LmtInfo.instru,
            interface=get_literal_value(LmtTelInterfaceT),
            interface_kind=get_literal_value(LmtTelInterfaceKindT),
            master_code=master_code,
            master=LmtInfo.master_code_master[master_code],
            data_models=[
                LmtDataModel.name,
                self.get_str(ds, schema.receiver).casefold(),
            ],
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
        Telescope dataset, or a tree with telescope data at its current node. Keep it
        open until the desired array slices have been loaded.
    mapper : LmtTelIOMapper, optional
        Field mapper. If omitted, resolve fields against the dataset.

    Raises
    ------
    ValueError
        If the dataset lacks the telescope timestamp variable.

    Examples
    --------
    With the reference observation under ``data_lmt`` in the working directory:

    >>> from tolteca_datamodels.lmt import open_lmt_tel
    >>> filepath = "data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    >>> with open_lmt_tel(filepath) as ds:
    ...     meta = ds.lmt_tel.meta
    ...     az = ds.lmt_tel.az.isel(time=slice(0, 10)).load()
    >>> meta.uid_raw_obs_file
    'tcs-145609-0-2-tel'
    """

    def _validate(self) -> None:
        """Validate telescope structure."""
        mapper = self.mapper
        ds = self.data_source
        if not mapper.has_arr(ds, mapper.schema.tel_time):
            msg = "Not an LMT telescope dataset: missing tel_time variable"
            raise ValueError(msg)

    @functools.cached_property
    def meta(self) -> LmtTelMetadata:
        """Cached observation metadata."""
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


def _get_tel_reader_record(data_source: object) -> LmtInterfaceInfo | None:
    if not isinstance(data_source, (xr.Dataset, xr.DataTree)):
        return None
    ds = unwrap_data_source(data_source)
    mapper = LmtTelIOMapper.from_data_source(ds)
    if not mapper.has_arr(ds, mapper.schema.tel_time):
        return None
    memberships = [LmtDataModel.name]
    if mapper.schema.receiver in mapper:
        memberships.append(mapper.get_str(ds, mapper.schema.receiver).casefold())
    return LmtInterfaceInfo.model_validate(
        LmtInfo.interface_interface_info["tel"].model_dump(round_trip=True)
        | {"data_models": memberships}
    )


def open_lmt_tel(source: str | Path | FileLoc, **kwargs) -> LmtTelDataset:
    """Open an LMT telescope netCDF file.

    Parameters
    ----------
    source : str, pathlib.Path or FileLoc
        Local telescope file. Remote files must be staged before opening.
    **kwargs
        Additional arguments passed to `xarray.open_dataset`.

    Returns
    -------
    LmtTelDataset
        Open dataset with an initialized ``lmt_tel`` accessor. Headers and sample
        arrays are not read until requested. Use the dataset as a context manager,
        or close it after loading the required arrays. If accessor initialization
        fails, the dataset is closed before the exception is raised.

    Raises
    ------
    ValueError
        If the location is remote or the dataset lacks telescope timestamps.
    """
    with ExitStack() as stack:
        ds = cast("LmtTelDataset", stack.enter_context(_open_netcdf(source, **kwargs)))
        _ = ds.lmt_tel
        stack.pop_all()
    return ds
