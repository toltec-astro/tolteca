"""Lazy access to TolTEC half-wave plate readout files."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast

import astropy.units as u
import xarray as xr
from astropy.time import Time
from pydantic import Field
from tollan.accessor import NameMapping, Schema
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper
from tollan.config.types import FrequencyQuantityField
from tollan.utils.log import logger
from tollan.utils.typing import get_literal_value

from ..core.source.opener import _open_netcdf
from ..lmt_base.filepath import FilepathOrigMetadataMixin
from ..lmt_base.metadata import LmtRawObsMetadata
from .toltec_db import ToltecDBRawObsMasterCode
from .types import ToltecDataKind, ToltecDataModel, ToltecInfo

if TYPE_CHECKING:
    from pathlib import Path

    from tollan.accessor import Mapping
    from tollan.utils.fileloc import FileLoc

    class ToltecHwprDataset(xr.Dataset):
        """Dataset with a typed half-wave plate readout accessor."""

        toltec_hwpr: ToltecHwprAccessor

    class ToltecHwprDataTree(xr.DataTree):
        """DataTree with a typed half-wave plate readout accessor."""

        toltec_hwpr: ToltecHwprAccessor


__all__ = [
    "ToltecHwprAccessor",
    "ToltecHwprIOMapper",
    "ToltecHwprIOSchema",
    "ToltecHwprInterfaceKindT",
    "ToltecHwprInterfaceT",
    "ToltecHwprMetadata",
    "open_toltec_hwpr",
]

type ToltecHwprInterfaceT = Literal["hwpr"]
type ToltecHwprInterfaceKindT = Literal["hwpr"]


class ToltecHwprMetadata(LmtRawObsMetadata, FilepathOrigMetadataMixin):
    """Half-wave plate acquisition metadata."""

    instru: ToltecInfo.InstruT = Field(description="TolTEC instrument")
    interface: ToltecHwprInterfaceT = Field(description="Readout interface")
    interface_kind: ToltecHwprInterfaceKindT = Field(
        description="Half-wave plate readout"
    )
    master: ToltecInfo.MasterT = Field(description="Acquisition master label")
    master_code: ToltecDBRawObsMasterCode = Field(description="Recorded master code")
    readout_code: int = Field(description="Recorded readout code")
    obs_type: ToltecInfo.ObsTypeT = Field(description="Acquisition type")
    obs_type_code: ToltecInfo.ObsTypeCodeT = Field(
        description="Recorded acquisition type code"
    )
    repeat: int = Field(description="Recorded acquisition repeat level")
    hwp_installed: int = Field(description="Recorded HWP installation flag")
    n_times: int = Field(ge=0, description="Number of readout records")
    f_smp: FrequencyQuantityField = Field(description="Readout record rate")


@dataclass
class ToltecHwprIOSchema(Schema):
    """Half-wave plate acquisition headers and native readout arrays."""

    master_code: Mapping = NameMapping("Header.Toltec.Master")
    readout_code: Mapping = NameMapping("Header.Toltec.RoachIndex")
    obsnum: Mapping = NameMapping("Header.Toltec.ObsNum")
    subobsnum: Mapping = NameMapping("Header.Toltec.SubObsNum")
    scannum: Mapping = NameMapping("Header.Toltec.ScanNum")
    obs_type_code: Mapping = NameMapping("Header.Toltec.ObsType")
    repeat: Mapping = NameMapping("Header.Toltec.RepeatLevel")
    hwp_installed: Mapping = NameMapping("Header.Toltec.HwpInstalled")
    filepath_orig: Mapping = NameMapping("Header.Toltec.Filename")
    t0: Mapping = NameMapping("Header.Toltec.ObsStartTime")
    t1: Mapping = NameMapping("Header.Toltec.ObsEndTime")
    f_smp: Mapping = NameMapping("Header.Toltec.SampleFreq")
    n_times: Mapping = NameMapping.dim("time")

    # These are separate packet arrays, not a shared flattened sample axis.
    enc_ts: Mapping = NameMapping.var("Data.Toltec.HwprEncTs")
    enc_val: Mapping = NameMapping.var("Data.Toltec.HwprEncVal")
    enc_uts: Mapping = NameMapping.var("Data.Toltec.HwprEncUts")
    zeropt_ts: Mapping = NameMapping.var("Data.Toltec.HwprZeroptTs")
    zeropt_val: Mapping = NameMapping.var("Data.Toltec.HwprZeroptVal")
    zeropt_uts: Mapping = NameMapping.var("Data.Toltec.HwprZeroptUts")
    pps_ts: Mapping = NameMapping.var("Data.Toltec.HwprPpsTs")
    pps_val: Mapping = NameMapping.var("Data.Toltec.HwprPpsVal")
    pps_uts: Mapping = NameMapping.var("Data.Toltec.HwprPpsUts")
    sensor_chan: Mapping = NameMapping.var("Data.Toltec.HwprSensorChan")
    sensor_volt: Mapping = NameMapping.var("Data.Toltec.HwprSensorVolt")
    sensor_uts: Mapping = NameMapping.var("Data.Toltec.HwprSensorUts")
    ts: Mapping = NameMapping.var("Data.Toltec.Ts")
    recv_time: Mapping = NameMapping.var("Data.Toltec.RecvTime")


class ToltecHwprIOMapper(XarrayMapper[ToltecHwprIOSchema]):
    """Read half-wave plate acquisition headers and arrays."""

    def get_metadata(self, data_source: xr.Dataset) -> ToltecHwprMetadata:
        """Read acquisition metadata without loading readout arrays.

        Parameters
        ----------
        data_source : xarray.Dataset
            Open half-wave plate readout dataset.

        Returns
        -------
        ToltecHwprMetadata
            Recorded acquisition settings. Start and end times are from the
            headers; integration time is the record count divided by the
            recorded sample rate, not the number of encoder samples.

        Raises
        ------
        ValueError
            If the sample rate is not positive.
        """
        ds = data_source
        schema = self.schema
        master_code = ToltecDBRawObsMasterCode(self.get_scalar(ds, schema.master_code))
        obs_type_code = self.get_scalar(ds, schema.obs_type_code)
        n_times = self.get_dim(ds, schema.n_times)
        f_smp = self.get_scalar(ds, schema.f_smp) * u.Hz
        if not f_smp > 0 * u.Hz:
            msg = "Sample frequency must be positive"
            raise ValueError(msg)
        meta = ToltecHwprMetadata(
            instru=ToltecInfo.instru,
            data_models=frozenset({ToltecDataModel.name}),
            interface=get_literal_value(ToltecHwprInterfaceT),
            interface_kind=get_literal_value(ToltecHwprInterfaceKindT),
            master_code=master_code,
            master=ToltecInfo.master_code_master[master_code],
            readout_code=self.get_scalar(ds, schema.readout_code),
            obsnum=self.get_scalar(ds, schema.obsnum),
            subobsnum=self.get_scalar(ds, schema.subobsnum),
            scannum=self.get_scalar(ds, schema.scannum),
            obs_type_code=obs_type_code,
            obs_type=ToltecInfo.obs_type_code_obs_type[obs_type_code],
            repeat=self.get_scalar(ds, schema.repeat),
            hwp_installed=self.get_scalar(ds, schema.hwp_installed),
            filepath_orig=self.get_str(ds, schema.filepath_orig),
            t0=Time(self.get_scalar(ds, schema.t0), format="unix"),
            t1=Time(self.get_scalar(ds, schema.t1), format="unix"),
            t_int=(n_times / f_smp).to(u.s),
            n_times=n_times,
            f_smp=f_smp,
        )
        logger.debug(
            "Read HWPR metadata: uid_raw_obs_file={}, obs_type={}, readout_code={}",
            meta.uid_raw_obs_file,
            meta.obs_type,
            meta.readout_code,
        )
        return meta


@xr.register_dataset_accessor("toltec_hwpr")
@xr.register_datatree_accessor("toltec_hwpr")
class ToltecHwprAccessor(XarrayAccessorBase[ToltecHwprIOMapper]):
    """Access half-wave plate metadata and native readout arrays.

    Parameters
    ----------
    data_source : xarray.Dataset or xarray.DataTree
        HWPR data at the current node. Keep the source open until the desired
        slices have been loaded.
    mapper : ToltecHwprIOMapper, optional
        Field mapper. If omitted, resolve fields against the dataset.

    Raises
    ------
    ValueError
        If the encoder value and timestamp arrays are missing.

    Notes
    -----
    Arrays retain their recorded values, dimensions and attributes. No angle
    conversion, flattening, time alignment or interpolation is performed.
    """

    def _validate(self) -> None:
        schema = self.mapper.schema
        if not isinstance(self.data_source, xr.Dataset) or not all(
            self.mapper.has_arr(self.data_source, field)
            for field in (schema.enc_ts, schema.enc_val, schema.enc_uts)
        ):
            msg = "Not a TolTEC HWPR dataset: missing encoder arrays"
            raise ValueError(msg)

    @cached_property
    def meta(self) -> ToltecHwprMetadata:
        """Cached acquisition metadata."""
        return self.mapper.get_metadata(cast("xr.Dataset", self.data_source))

    @property
    def data_kind(self) -> ToltecDataKind:
        """Half-wave plate readout data kind."""
        return ToltecDataKind.hwpr

    @property
    def enc_ts(self) -> xr.DataArray:
        """Native encoder timestamps."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.enc_ts)

    @property
    def enc_val(self) -> xr.DataArray:
        """Native encoder values."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.enc_val)

    @property
    def enc_uts(self) -> xr.DataArray:
        """Recorded encoder Unix timestamps."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.enc_uts)

    @property
    def zeropt_ts(self) -> xr.DataArray:
        """Native zero-point timestamps."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.zeropt_ts)

    @property
    def zeropt_val(self) -> xr.DataArray:
        """Native zero-point values."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.zeropt_val)

    @property
    def zeropt_uts(self) -> xr.DataArray:
        """Recorded zero-point Unix timestamps."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.zeropt_uts)

    @property
    def pps_ts(self) -> xr.DataArray:
        """Native PPS timestamps."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.pps_ts)

    @property
    def pps_val(self) -> xr.DataArray:
        """Native PPS values."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.pps_val)

    @property
    def pps_uts(self) -> xr.DataArray:
        """Recorded PPS Unix timestamps."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.pps_uts)

    @property
    def sensor_chan(self) -> xr.DataArray:
        """Recorded sensor channel identifiers."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.sensor_chan)

    @property
    def sensor_volt(self) -> xr.DataArray:
        """Native sensor voltage readings."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.sensor_volt)

    @property
    def sensor_uts(self) -> xr.DataArray:
        """Recorded sensor Unix timestamps."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.sensor_uts)

    @property
    def ts(self) -> xr.DataArray:
        """Native packet clock and counter fields."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.ts)

    @property
    def recv_time(self) -> xr.DataArray:
        """Recorded packet reception times in Unix seconds."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.recv_time)


def open_toltec_hwpr(source: str | Path | FileLoc, **kwargs) -> ToltecHwprDataset:
    """Open a TolTEC half-wave plate netCDF file.

    Parameters
    ----------
    source : str, pathlib.Path or FileLoc
        Local HWPR file. Remote files must be staged locally before opening.
    **kwargs
        Additional arguments passed to `xarray.open_dataset`.

    Returns
    -------
    ToltecHwprDataset
        Open dataset with a typed ``toltec_hwpr`` accessor. Headers and arrays
        remain lazy until requested. Use a context manager or close the dataset
        after loading the required arrays.

    Raises
    ------
    ValueError
        If the location is remote or the encoder arrays are missing. A newly
        opened dataset is closed if accessor initialization fails.
    """
    with ExitStack() as stack:
        ds = cast(
            "ToltecHwprDataset", stack.enter_context(_open_netcdf(source, **kwargs))
        )
        _ = ds.toltec_hwpr
        stack.pop_all()
    return ds
