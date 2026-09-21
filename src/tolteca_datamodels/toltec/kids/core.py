"""TolTEC KIDs data I/O schema, mapper, and accessor.

This module defines:
1. Raw I/O schema (ToltecKidsIOSchema) - Maps to raw netCDF variable names
2. Mapper (ToltecKidsIOMapper) - Provides access to raw data and metadata
3. Main accessor (ToltecKidsAccessor) - xarray accessor for TolTEC data

ToltecKidsIOMapper uses ToltecKidsIOSchema and provides get_metadata() to build
ToltecSweepMetadata or ToltecTimeStreamMetadata from a dataset.
For reduced data variables (Data.Kids.*), see sweep.py and timestream.py.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time
from tollan.accessor import FieldLocation, NameMapping, Schema
from tollan.accessor.xarray import XarrayMapper

from ..metadata import ToltecSweepMetadata, ToltecTimeStreamMetadata
from ..toltecdb import ToltecDBRawObsMaster, ToltecDBRawObsType
from ..types import ToltecDataKind, ToltecInfo

if TYPE_CHECKING:
    from typing import Any

    from tollan.accessor import Mapping

__all__ = [
    "ToltecKidsAccessor",
    "ToltecKidsIOMapper",
    "ToltecKidsIOSchema",
]


@dataclass
class ToltecKidsIOSchema(Schema):
    """Schema for raw TolTEC KIDs data I/O.

    Maps to raw netCDF variable and dimension names. ToltecKidsAccessor uses
    this schema to provide access to raw data and metadata without processing.

    **Data Variables (Raw):**
    - I, Q: Raw in-phase and quadrature data (Data.Toltec.Is/Qs)

    **Dimension Counts:**
    - n_times: Number of acquired samples (time / ntimes)
    - n_chans: Number of readout channels (iqlen / n_chans)
    - n_kids_design: Number of designed detector positions (loclen)
    - n_blocks: Number of explicitly indexed blocks (block)
    - n_blocks_max: Allocated sweep capacity (numSweeps)

    **Tone/LO Metadata:**
    - f_tones: ROACH tone frequencies (Header.Toltec.ToneFreq)
    - f_los: LO frequency data (Data.Toltec.LoFreq)
    - f_lo_center: Center LO frequency (Header.Toltec.LoCenterFreq)
    - mask_tones: Tone enable/disable mask (Header.Toltec.ToneMask)
    - amp_tones: Tone amplitude values (Header.Toltec.ToneAmp)
    - phase_tones: Tone phase values (Header.Toltec.TonePhase)

    **Observation Metadata:**
    - roach: ROACH board index (Header.Toltec.RoachIndex)
    - obsnum: Observation number (Header.Toltec.ObsNum)
    - subobsnum: Sub-observation number (Header.Toltec.SubObsNum)
    - scannum: Scan number (Header.Toltec.ScanNum)
    - f_smp: Sample frequency (Header.Toltec.SampleFreq)
    - obs_type: Observation type (Header.Toltec.ObsType)
    - obs_start_time: Observation start time scalar (Header.Toltec.ObsStartTime)
    - kind_str: Data kind string (Header.Kids.kind)
    - n_sweepsteps: Number of sweep steps (Header.Toltec.NumSweepSteps)
    - n_sweepreps: Samples per sweep step
      (Header.Toltec.NumSamplesPerSweepStep)

    **Instrument Settings:**
    - atten_drive: Drive attenuation (Header.Toltec.DriveAtten)
    - atten_sense: Sense attenuation (Header.Toltec.SenseAtten)

    Notes
    -----
    This schema maps directly to the raw netCDF structure and does NOT
    include namespacing since it represents the actual file format.
    """

    # Raw I/Q data fields
    I: Mapping = NameMapping.var(("Data.Toltec.Is", "I"))
    Q: Mapping = NameMapping.var(("Data.Toltec.Qs", "Q"))

    # Dimension counts; these do not require coordinate variables.
    n_times: Mapping = NameMapping.dim(("time", "ntimes"))
    n_chans: Mapping = NameMapping.dim(("iqlen", "n_chans", "chan", "channel"))
    n_blocks: Mapping = NameMapping.dim("block")
    n_blocks_max: Mapping = NameMapping.dim("numSweeps")
    n_kids_design: Mapping = NameMapping.dim("loclen")

    # Tone/channel axis data
    f_tones: Mapping = NameMapping.var("Header.Toltec.ToneFreq")
    f_los: Mapping = NameMapping.var("Data.Toltec.LoFreq")
    f_lo_center: Mapping = NameMapping("Header.Toltec.LoCenterFreq")
    mask_tones: Mapping = NameMapping.var("Header.Toltec.ToneMask")
    amp_tones: Mapping = NameMapping.var(
        ("Header.Toltec.ToneAmp", "Header.Toltec.ToneAmps"),
    )
    phase_tones: Mapping = NameMapping.var("Header.Toltec.TonePhase")

    # Observation metadata
    master: Mapping = NameMapping("Header.Toltec.Master")
    roach: Mapping = NameMapping("Header.Toltec.RoachIndex")
    obsnum: Mapping = NameMapping("Header.Toltec.ObsNum")
    subobsnum: Mapping = NameMapping("Header.Toltec.SubObsNum")
    scannum: Mapping = NameMapping("Header.Toltec.ScanNum")
    f_smp: Mapping = NameMapping("Header.Toltec.SampleFreq")
    obs_type: Mapping = NameMapping("Header.Toltec.ObsType")
    kind_str: Mapping = NameMapping(("Header.Kids.kind", "kind_str"))
    obs_start_time: Mapping = NameMapping("Header.Toltec.ObsStartTime")
    obs_end_time: Mapping = NameMapping("Header.Toltec.ObsEndTime")
    cal_roach: Mapping = NameMapping("Header.Toltec.RoachIndex")
    cal_obsnum: Mapping = NameMapping("Header.Toltec.TargSweepObsNum")
    cal_subobsnum: Mapping = NameMapping("Header.Toltec.TargSweepSubObsNum")
    cal_scannum: Mapping = NameMapping("Header.Toltec.TargSweepScanNum")
    n_chans_max: Mapping = NameMapping("Header.Toltec.MaxNumTones")
    filename_orig: Mapping = NameMapping("Header.Toltec.Filename")
    repeatvar: Mapping = NameMapping("Header.Toltec.RepeatLevel")
    adc_snap: Mapping = NameMapping.var("Header.Toltec.AdcSnapData")
    n_sweepsteps: Mapping = NameMapping(
        [
            ("Header.Toltec.NumSweepSteps", FieldLocation.var),
            ("Header.Toltec.NumSweepSteps", FieldLocation.attr),
            ("sweep", FieldLocation.dim),
            ("nsweeps", FieldLocation.dim),
            ("sweeps", FieldLocation.dim),
        ]
    )
    n_sweepreps: Mapping = NameMapping(
        "Header.Toltec.NumSamplesPerSweepStep",
    )

    # Instrument settings
    atten_drive: Mapping = NameMapping("Header.Toltec.DriveAtten")
    atten_sense: Mapping = NameMapping("Header.Toltec.SenseAtten")
    atten_in: Mapping = NameMapping(("Header.Toltec.AttenIn", "atten_in"))
    atten_out: Mapping = NameMapping(("Header.Toltec.AttenOut", "atten_out"))


class ToltecKidsIOMapper(XarrayMapper[ToltecKidsIOSchema]):
    """Mapper for TolTEC KIDs data.

    Provides access to raw TolTEC netCDF files using ToltecKidsIOSchema
    and constructs TolTEC metadata dataclasses from a dataset.

    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.open_dataset("toltec_raw.nc")
    >>> mapper = ToltecKidsIOMapper.from_data_source(ds)
    >>> roach = mapper.get_scalar(ds, mapper.schema.roach)
    >>> f_tone = mapper.get_arr(ds, mapper.schema.f_tones)
    >>> meta = mapper.get_metadata(ds)
    """

    def get_metadata(
        self,
        data_source: xr.Dataset,
        block: int | None = None,
    ) -> ToltecSweepMetadata | ToltecTimeStreamMetadata | list[ToltecSweepMetadata]:
        """Read typed observation metadata without loading detector arrays.

        Required identifiers, acquisition times, and sample rate must be present.
        Optional header fields remain None when absent. Acquisition times come
        from the file headers; integration time is sample count divided by rate.

        A dataset with an explicit block dimension returns one metadata object
        per block. A flat acquisition returns one file-level object, including
        the number of sweeps it contains. Pass block to select one sweep.
        Flat tune block boundaries require reading the one-dimensional LO
        frequency array; detector I/Q arrays are never read.

        Parameters
        ----------
        data_source : xr.Dataset
            Open dataset payload to read from.
        block : int or None, optional
            Select one recorded sweep block. None reads the whole acquisition.

        Returns
        -------
        ToltecSweepMetadata or ToltecTimeStreamMetadata or list
            One metadata instance, or a list for an explicitly block-indexed
            sweep dataset when block is None.
        """
        ds = data_source
        schema = self.schema
        is_multi_block = self.has_dim(ds, schema.n_blocks)
        n_blocks = self.get_dim(ds, schema.n_blocks) if is_multi_block else 1
        data_kind = self._get_data_kind(ds)
        if (
            (is_multi_block or data_kind != ToltecDataKind.Tune)
            and block is not None
            and not 0 <= block < n_blocks
        ):
            msg = f"Block {block} out of range [0, {n_blocks})"
            raise ValueError(msg)

        roach = self.get_scalar(ds, schema.roach)
        interface = ToltecInfo.roach_interface[roach]
        master = self.get_scalar(ds, schema.master)
        t0 = Time(self.get_scalar(ds, schema.obs_start_time), format="unix")
        t1 = Time(self.get_scalar(ds, schema.obs_end_time), format="unix")
        f_smp = self._get_frequency(ds, schema.f_smp)
        if f_smp <= 0 * u.Hz:
            msg = "Sample frequency must be positive"
            raise ValueError(msg)
        base_fields: dict[str, Any] = {
            "roach": roach,
            "nw": roach,
            "interface": interface,
            "array_name": ToltecInfo.interface_array_name[interface],
            "master": ToltecDBRawObsMaster.get_master_type(
                ToltecDBRawObsMaster(master)
            ),
            "mastervar": str(master),
            "obsnum": self.get_scalar(ds, schema.obsnum),
            "subobsnum": self.get_scalar(ds, schema.subobsnum),
            "scannum": self.get_scalar(ds, schema.scannum),
            "t0": t0,
            "t1": t1,
            "n_chans": self.get_dim(ds, schema.n_chans),
            "f_smp": f_smp,
        }
        for key in (
            "obs_type",
            "atten_drive",
            "atten_sense",
            "cal_roach",
            "cal_obsnum",
            "cal_subobsnum",
            "cal_scannum",
            "n_chans_max",
        ):
            field = getattr(schema, key)
            if field in self:
                base_fields[key] = self.get_scalar(ds, field)
        if schema.n_kids_design in self:
            base_fields["n_kids_design"] = self.get_dim(ds, schema.n_kids_design)
        if schema.filename_orig in self:
            base_fields["filename_orig"] = self.get_str(ds, schema.filename_orig)
        if schema.repeatvar in self:
            base_fields["repeatvar"] = str(self.get_scalar(ds, schema.repeatvar))
        if schema.adc_snap in self:
            # Keep the array reference, not the ADC snapshot payload.
            base_fields["adc_snap"] = self.get_name(schema.adc_snap)

        if data_kind & ToltecDataKind.Sweep:
            return self._get_sweep_metadata(
                ds, base_fields, is_multi_block, n_blocks, block
            )
        return self._get_timestream_metadata(ds, base_fields)

    def _get_data_kind(self, data_source: xr.Dataset) -> ToltecDataKind:
        """Determine data kind from dataset."""
        # Check for kind_str (reduced/processed data)
        if self.schema.kind_str in self:
            kind_str = self.get_str(data_source, self.schema.kind_str)
            kind_map = {
                "d21": ToltecDataKind.D21,
                "processed_sweep": ToltecDataKind.ReducedSweep,
                "processed_timestream": ToltecDataKind.SolvedTimeStream,
                "SolvedTimeStream": ToltecDataKind.SolvedTimeStream,
            }
            return kind_map.get(kind_str, ToltecDataKind.Unknown)

        # Check for obs_type (raw data)
        if self.schema.obs_type in self:
            obs_type = self.get_scalar(data_source, self.schema.obs_type)
            try:
                raw_obs_type = ToltecDBRawObsType(obs_type)
            except (TypeError, ValueError):
                return ToltecDataKind.Unknown
            return ToltecDBRawObsType.get_data_kind(raw_obs_type)

        # Infer from structure
        if self.schema.n_sweepsteps in self:
            return ToltecDataKind.RawSweep
        if self.schema.n_times in self:
            return ToltecDataKind.RawTimeStream

        return ToltecDataKind.Unknown

    def _get_sweep_metadata(
        self,
        data_source,
        base_fields,
        is_multi_block,
        n_blocks,
        block,
    ):
        """Construct sweep metadata instance(s)."""
        ds = data_source
        schema = self.schema
        sweep_fields = base_fields.copy()
        sweep_fields["n_sweepsteps"] = self.get_scalar(ds, schema.n_sweepsteps)
        if schema.n_sweepreps in self:
            sweep_fields["n_sweepreps"] = self.get_scalar(ds, schema.n_sweepreps)

        slices = [slice(None)]
        if not is_multi_block and self._get_data_kind(ds) == ToltecDataKind.Tune:
            # numSweeps is allocated capacity; LO resets count recorded sweeps.
            f_los = self.get_arr(ds, schema.f_los).to_numpy()
            if f_los.ndim != 1 or not np.all(np.isfinite(f_los) & (f_los > 0)):
                msg = "Tune LO frequencies must be a positive finite 1-D array"
                raise ValueError(msg)
            edges = [0, *(np.flatnonzero(np.diff(f_los) < 0) + 1), len(f_los)]
            slices = [slice(a, b) for a, b in pairwise(edges)]
            n_blocks = len(slices)
        if not is_multi_block and block is not None and not 0 <= block < n_blocks:
            msg = f"Block {block} out of range [0, {n_blocks})"
            raise ValueError(msg)

        sweep_fields["n_blocks"] = n_blocks
        sweep_fields["n_blocks_max"] = (
            self.get_dim(ds, schema.n_blocks_max)
            if schema.n_blocks_max in self
            else n_blocks
        )
        if is_multi_block:
            if block is not None:
                return self._get_single_block_sweep_metadata(sweep_fields, ds, block)
            return [
                self._get_single_block_sweep_metadata(sweep_fields, ds, i)
                for i in range(n_blocks)
            ]

        if block is not None and schema.n_times in self:
            ds = ds.isel({self.get_name(schema.n_times): slices[block]})
        sweep_fields["n_sweeps"] = n_blocks if block is None else 1
        # Reduced sweeps have one sample per sweep step, not a time dimension.
        sweep_fields["n_times"] = (
            self.get_dim(ds, schema.n_times)
            if schema.n_times in self
            else sweep_fields["n_sweepsteps"]
        )
        sweep_fields["t_exp"] = (sweep_fields["n_times"] / sweep_fields["f_smp"]).to(
            u.s
        )
        if schema.f_lo_center in self:
            sweep_fields["f_lo_center"] = self._get_frequency(
                ds, schema.f_lo_center
            ).to_value(u.Hz)
        return ToltecSweepMetadata(**sweep_fields)

    def _get_single_block_sweep_metadata(self, base_fields, ds, block_idx):
        """Get sweep metadata for a specific explicitly indexed block."""
        block_fields = base_fields.copy()
        ds = ds.isel({self.get_name(self.schema.n_blocks): block_idx})
        block_fields["n_sweeps"] = 1
        block_fields["n_times"] = (
            self.get_dim(ds, self.schema.n_times)
            if self.schema.n_times in self
            else block_fields["n_sweepsteps"]
        )
        block_fields["t_exp"] = (block_fields["n_times"] / block_fields["f_smp"]).to(
            u.s
        )
        if self.schema.f_lo_center in self:
            block_fields["f_lo_center"] = self._get_frequency(
                ds, self.schema.f_lo_center
            ).to_value(u.Hz)
        return ToltecSweepMetadata(**block_fields)

    def _get_timestream_metadata(self, data_source, base_fields):
        """Construct timestream metadata instance."""
        ts_fields = base_fields.copy()
        ts_fields["n_times"] = self.get_dim(data_source, self.schema.n_times)
        ts_fields["t_exp"] = (ts_fields["n_times"] / ts_fields["f_smp"]).to(u.s)
        return ToltecTimeStreamMetadata(**ts_fields)

    def _get_frequency(self, ds, field):
        """Read a scalar frequency, using Hz when the header has no unit."""
        value = self.get_scalar(ds, field)
        unit = (
            self.get_arr(ds, field).u.unit or u.Hz
            if self.has_arr(ds, field)
            else u.Hz
        )
        return u.Quantity(value, unit).to(u.Hz)

    def _get_f_lo_center(self, data_source: xr.Dataset) -> float | None:
        """Resolve LO center frequency from dataset.

        Checks the f_lo_center field first (as attr or data variable), then
        falls back to the mean of f_los if available.

        Parameters
        ----------
        data_source : xr.Dataset
            Dataset to read from

        Returns
        -------
        float or None
            LO center frequency in Hz, or None if not determinable
        """
        if self.schema.f_lo_center in self:
            if self.has_attr(data_source, self.schema.f_lo_center):
                return float(self.get_scalar(data_source, self.schema.f_lo_center))
            if self.has_arr(data_source, self.schema.f_lo_center):
                val = self.get_arr(data_source, self.schema.f_lo_center)
                if hasattr(val, "values"):
                    return (
                        float(val.mean().values) if val.size > 1 else float(val.values)
                    )
                return float(val)
        if self.schema.f_los in self:
            f_los = self.get_arr(data_source, self.schema.f_los)
            return float(f_los.mean().values)
        return None

    def get_chan_axis_data(self, data_source: xr.Dataset) -> QTable:
        """Get channel axis data as an astropy QTable.

        Parameters
        ----------
        data_source : xr.Dataset
            Dataset to read from

        Returns
        -------
        QTable
            Table with channel metadata including:
            - channel: Channel index
            - f_tone: Tone frequency
            - f_chan: Channel frequency (tone + LO center)
            - tone_mask: Enabled/disabled mask
            - tone_amp: Tone amplitude (if available)
            - tone_phase: Tone phase (if available)
        """
        n_chans = self.get_dim(data_source, self.schema.n_chans)

        table = QTable()
        table["channel"] = np.arange(n_chans)

        if self.schema.f_tones in self:
            f_tones = self.get_arr(data_source, self.schema.f_tones)
            table["f_tone"] = f_tones.values * u.Hz

        f_lo_center = self._get_f_lo_center(data_source)
        if "f_tone" in table.colnames and f_lo_center is not None:
            table["f_chan"] = table["f_tone"] + f_lo_center * u.Hz

        if self.schema.mask_tones in self:
            table["tone_mask"] = self.get_arr(
                data_source, self.schema.mask_tones
            ).values
        if self.schema.amp_tones in self:
            table["tone_amp"] = self.get_arr(data_source, self.schema.amp_tones).values
        if self.schema.phase_tones in self:
            table["tone_phase"] = (
                self.get_arr(data_source, self.schema.phase_tones).values * u.rad
            )

        meta = self.get_metadata(data_source)
        table.meta["roach"] = getattr(meta, "roach", None)
        table.meta["array_name"] = getattr(meta, "array_name", None)

        return table

    def get_sweep_axis_data(self, data_source: xr.Dataset) -> QTable:
        """Get sweep axis data as an astropy QTable.

        Parameters
        ----------
        data_source : xr.Dataset
            Dataset to read from

        Returns
        -------
        QTable
            Table with sweep step metadata including:
            - sweep_id: Sweep step index
            - f_sweep: Sweep frequency offset from LO center
            - f_lo: LO frequency at this sweep step

        Raises
        ------
        ValueError
            If dataset has no frequency dimension
        """
        ds = data_source
        is_sweep = (
            "sweep" in ds.dims or "nsweeps" in ds.dims or "frequency" in ds.coords
        )
        if not is_sweep:
            msg = "Dataset is not sweep data (no frequency coordinate)"
            raise ValueError(msg)

        freq_coord = None
        for coord_name in ["sweep", "nsweeps", "frequency"]:
            if coord_name in ds.coords:
                freq_coord = ds.coords[coord_name]
                break
            if coord_name in ds.dims:
                freq_coord = ds[coord_name] if coord_name in ds else None
                if freq_coord is not None:
                    break

        if freq_coord is None:
            msg = "Could not find frequency coordinate"
            raise ValueError(msg)

        table = QTable()
        table["sweep_id"] = np.arange(len(freq_coord))

        f_lo_center = self._get_f_lo_center(data_source)
        if f_lo_center is not None:
            freq_vals = freq_coord.values
            if np.abs(freq_vals).max() < 10e6:  # noqa: PLR2004
                f_sweep = freq_vals
                f_lo = f_sweep + f_lo_center
            else:
                f_lo = freq_vals
                f_sweep = f_lo - f_lo_center
            table["f_sweep"] = f_sweep * u.Hz
            table["f_lo"] = f_lo * u.Hz
        else:
            table["f_lo"] = freq_coord.values * u.Hz

        meta = self.get_metadata(data_source)
        table.meta["roach"] = getattr(meta, "roach", None)
        table.meta["array_name"] = getattr(meta, "array_name", None)
        table.meta["f_lo_center"] = f_lo_center

        return table


# Main Accessor
# =============


@xr.register_dataset_accessor("toltec_kids")
@xr.register_datatree_accessor("toltec_kids")
class ToltecKidsAccessor:
    """TolTEC-specific accessor for KIDs data.

    Provides access to TolTEC instrument-specific metadata, tone properties,
    and reduced data views. Registered for both ``xr.Dataset`` and
    ``xr.DataTree`` so that it works on raw datasets and on the DataTree
    returned by ``SweepReducer`` / ``TimestreamReducer``.

    Properties
    ----------
    meta : ToltecSweepMetadata or ToltecTimeStreamMetadata
        Complete metadata dataclass with all observation info
    data_kind : ToltecDataKind
        Data kind flag (VnaSweep, TargetSweep, RawTimeStream, etc.)
    f_tone : xr.DataArray
        ROACH tone frequencies
    f_lo : xr.DataArray or float
        LO frequency (array for sweeps, scalar for fixed)
    f_chan : xr.DataArray
        Channel frequencies (f_tone + f_lo_center)
    tone_mask : xr.DataArray
        Tone enable/disable mask
    tone_amp : xr.DataArray
        Tone amplitude values
    tone_phase : xr.DataArray
        Tone phase values
    sweep : ReducedSweepView
        View of the reduced sweep child node (when DataTree is available)
    timestream : ReducedTimestreamView
        View of the reduced timestream/PSD child node (when DataTree is available)

    Examples
    --------
    Access metadata:
        >>> meta = ds.toltec_kids.meta
        >>> meta.roach
        0
        >>> meta.obsnum
        12345
        >>> meta.f_smp
        <Quantity 488.28125 Hz>

    Access tone properties:
        >>> ds.toltec_kids.f_tone
        <xr.DataArray 'f_tone' (channel: 256)>
        >>> ds.toltec_kids.f_chan
        <xr.DataArray 'f_chan' (channel: 256)>

    Access reduced data via DataTree:
        >>> dt = SweepReducer()(ds)
        >>> view = dt.toltec_kids.sweep
        >>> view.sweep         # averaged sweep Dataset
        >>> view.has_blocks    # True if multi-block
    """

    def __init__(self, xarray_obj: xr.Dataset | xr.DataTree) -> None:
        self._obj = xarray_obj
        # Mapper operates on a plain Dataset — use root dataset for DataTree
        self._root_ds: xr.Dataset = (
            xarray_obj.dataset if isinstance(xarray_obj, xr.DataTree) else xarray_obj
        )
        self.mapper = ToltecKidsIOMapper.from_data_source(self._root_ds)

    # Metadata properties
    # ===================

    @functools.cached_property
    def meta(self):
        """Get metadata as dataclass instance(s).

        Constructs appropriate metadata dataclass based on data kind.
        For multi-block data, returns list of metadata instances (one per block).

        Returns
        -------
        ToltecSweepMetadata or ToltecTimeStreamMetadata or list
            Metadata instance for single-block data, or list of metadata
            instances for multi-block data (one per block)

        Examples
        --------
        Single-block sweep data:
            >>> meta = ds.toltec_kids.meta
            >>> meta.roach
            0
            >>> meta.obsnum
            12345
            >>> meta.n_sweepsteps
            491
            >>> meta.f_smp
            <Quantity 488.28125 Hz>

        Multi-block sweep data:
            >>> meta_list = ds.toltec_kids.meta
            >>> len(meta_list)
            2
            >>> meta_list[0].n_sweeps
            1
            >>> meta_list[0].f_lo_center
            4500000000.0

        Convenience fields:
            >>> meta = ds.toltec_kids.meta
            >>> meta.array_name
            'a1100'
            >>> meta.interface
            'toltec0'
        """
        return self.mapper.get_metadata(self._root_ds)

    # Tone/Channel properties
    # =======================

    @functools.cached_property
    def f_tone(self) -> xr.DataArray | None:
        """ROACH tone frequencies.

        Returns
        -------
        xr.DataArray or None
            Tone frequencies in Hz, or None if not present
        """
        if self.mapper.schema.f_tones not in self.mapper:
            return None
        return self.mapper.get_arr(self._root_ds, self.mapper.schema.f_tones)

    @functools.cached_property
    def f_lo(self) -> xr.DataArray | float | None:
        """LO frequency.

        Returns
        -------
        xr.DataArray, float, or None
            LO frequency in Hz (array for sweeps, scalar for fixed),
            or None if not present
        """
        if self.mapper.schema.f_los not in self.mapper:
            return None

        name = self.mapper.get_name(self.mapper.schema.f_los)
        # Check if it's a coordinate/variable (array) or attr (scalar)
        if name in self._root_ds.coords:
            return self.mapper.get_arr(self._root_ds, self.mapper.schema.f_los)
        if name in self._root_ds:
            return self.mapper.get_arr(self._root_ds, self.mapper.schema.f_los)
        if name in self._root_ds.attrs:
            return self._root_ds.attrs[name]
        return None

    @functools.cached_property
    def f_chan(self) -> xr.DataArray | None:
        """Channel frequencies (f_tone + f_lo_center).

        Returns
        -------
        xr.DataArray or None
            Channel frequencies in Hz, or None if not present
        """
        f_tone = self.f_tone
        if f_tone is None:
            return None
        f_lo_center = self.mapper._get_f_lo_center(self._root_ds)
        if f_lo_center is None:
            return None
        return f_tone + f_lo_center

    @functools.cached_property
    def tone_mask(self) -> xr.DataArray | None:
        """Tone enable/disable mask.

        Returns
        -------
        xr.DataArray or None
            Boolean mask where True indicates enabled tone.
            For multi-block data, has 'block' dimension.
            Returns None if not present.
        """
        if self.mapper.schema.mask_tones not in self.mapper:
            return None
        arr = self.mapper.get_arr(self._root_ds, self.mapper.schema.mask_tones)
        # Note: For multi-block data after reduce_raw_sweep with concat,
        # the block dimension is automatically preserved
        return arr

    @functools.cached_property
    def tone_amp(self) -> xr.DataArray | None:
        """Tone amplitude values.

        Returns
        -------
        xr.DataArray or None
            Tone amplitudes.
            For multi-block data, has 'block' dimension.
            Returns None if not present.
        """
        if self.mapper.schema.amp_tones not in self.mapper:
            return None
        arr = self.mapper.get_arr(self._root_ds, self.mapper.schema.amp_tones)
        return arr

    @functools.cached_property
    def tone_phase(self) -> xr.DataArray | None:
        """Tone phase values.

        Returns
        -------
        xr.DataArray or None
            Tone phases in radians.
            For multi-block data, has 'block' dimension.
            Returns None if not present.
        """
        if self.mapper.schema.phase_tones not in self.mapper:
            return None
        arr = self.mapper.get_arr(self._root_ds, self.mapper.schema.phase_tones)
        return arr

    # Data Kind Identification
    # ========================

    @functools.cached_property
    def data_kind(self) -> ToltecDataKind:
        """Identify the data kind from dataset attributes.

        Returns
        -------
        ToltecDataKind
            Data kind flag indicating the type of KIDs data
        """
        return self.mapper._get_data_kind(self._root_ds)

    # Reduced Data Views
    # ==================

    @functools.cached_property
    def sweep(self):
        """View of the reduced sweep child node.

        Returns a :class:`ReducedSweepView` backed by the current object.
        When the object is a DataTree returned by :class:`SweepReducer`, the
        view transparently resolves the reduced sweep child node.

        Returns
        -------
        ReducedSweepView
            View providing access to averaged sweep data and uncertainties.

        Examples
        --------
        >>> dt = SweepReducer()(ds_raw)
        >>> view = dt.toltec_kids.sweep
        >>> view.sweep       # averaged sweep Dataset
        >>> view.has_blocks  # True if multi-block data
        """
        from .sweep import ReducedSweepView  # noqa: PLC0415

        return ReducedSweepView(self._obj)

    @functools.cached_property
    def timestream(self):
        """View of the reduced timestream / PSD child node.

        Returns a :class:`ReducedTimestreamView` backed by the current object.
        When the object is a DataTree returned by :class:`TimestreamReducer`,
        the view transparently resolves the reduced timestream child node.

        Returns
        -------
        ReducedTimestreamView
            View providing access to PSD data and summary statistics.

        Examples
        --------
        >>> dt = TimestreamReducer()(ds_ts)
        >>> view = dt.toltec_kids.timestream
        >>> view.f_psd           # PSD frequency axis
        >>> view.I_psd           # I-channel PSD
        >>> view.I_psd_median    # median PSD value in stat freq range
        """
        from .timestream import ReducedTimestreamView  # noqa: PLC0415

        return ReducedTimestreamView(self._obj)

    # Axis Data Methods
    # =================

    def get_chan_axis_data(self) -> QTable:
        """Get channel axis data as an astropy QTable.

        Returns
        -------
        QTable
            Table with channel metadata including:
            - channel: Channel index
            - f_tone: Tone frequency
            - f_chan: Channel frequency (tone + LO center)
            - tone_mask: Enabled/disabled mask
            - tone_amp: Tone amplitude (if available)
            - tone_phase: Tone phase (if available)

        Examples
        --------
        >>> chan_data = ds.toltec_kids.get_chan_axis_data()
        >>> chan_data['f_chan']  # Access channel frequencies
        >>> enabled = chan_data[chan_data['tone_mask']]  # Filter enabled
        """
        return self.mapper.get_chan_axis_data(self._root_ds)

    def get_sweep_axis_data(self) -> QTable:
        """Get sweep axis data as an astropy QTable.

        Returns
        -------
        QTable
            Table with sweep step metadata including:
            - sweep_id: Sweep step index
            - f_sweep: Sweep frequency offset from LO center
            - f_lo: LO frequency at this sweep step

        Raises
        ------
        ValueError
            If dataset has no frequency dimension

        Examples
        --------
        >>> sweep_data = ds.toltec_kids.get_sweep_axis_data()
        >>> sweep_data['f_sweep']  # Access sweep frequencies
        """
        return self.mapper.get_sweep_axis_data(self._root_ds)

    def select_channels(self, query: str | slice | list | np.ndarray) -> xr.Dataset:
        """Select channels by various criteria.

        Parameters
        ----------
        query : str, slice, list, or ndarray
            Channel selection criteria:
            - str: Query expression (e.g., "tone_mask == True")
            - slice: Index slice (e.g., slice(0, 10))
            - list/array: List of channel indices or boolean mask

        Returns
        -------
        xr.Dataset
            Dataset with selected channels

        Examples
        --------
        Select enabled channels:
            >>> ds_enabled = ds.toltec_kids.select_channels("tone_mask")

        Select first 10 channels:
            >>> ds_subset = ds.toltec_kids.select_channels(slice(0, 10))

        Select by index list:
            >>> ds_subset = ds.toltec_kids.select_channels([0, 5, 10, 15])
        """
        ds = self._root_ds

        # Get channel dimension name
        i_data = self.mapper.get_arr(self._root_ds, self.mapper.schema.I)
        chan_dim = i_data.dims[0]

        if isinstance(query, str):
            # Query expression using channel data
            chan_table = self.get_chan_axis_data()
            # Simple query parsing (e.g., "tone_mask")
            if query in chan_table.colnames:
                mask = chan_table[query]
                indices = np.where(mask)[0]
            else:
                msg = f"Query '{query}' not recognized. Available columns: {chan_table.colnames}"
                raise ValueError(msg)
        elif isinstance(query, slice):
            indices = query
        else:
            # List or array
            indices = query

        return ds.isel({chan_dim: indices})
