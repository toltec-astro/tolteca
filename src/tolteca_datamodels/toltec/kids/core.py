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
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from astropy.table import QTable
from pydantic.dataclasses import dataclass
from tollan.accessor import Mapping, Schema
from tollan.accessor.xarray import XarrayMapper

if TYPE_CHECKING:
    from ..types import ToltecDataKind

__all__ = [
    "ToltecKidsAccessor",
    "ToltecKidsIOMapper",
    "ToltecKidsIOSchema",
]


@dataclass
class ToltecKidsIOSchema(Schema):
    """Schema for raw TolTEC KIDs data I/O.

    Maps to raw netCDF variable names as they appear in the LMT data files
    (tolteca_ref_data/data_lmt). This schema is used by ToltecKidsAccessor
    to provide access to raw data and metadata without any processing.

    **Data Variables (Raw):**
    - I, Q: Raw in-phase and quadrature data (Data.Toltec.Is/Qs)

    **Coordinates:**
    - time: Sample time axis
    - channel: Detector channel axis (iqlen, loclen)

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

    **Sweep Coordinate:**
    - frequency: Sweep frequency axis (sweep / nsweeps dim)

    **Instrument Settings:**
    - atten_drive: Drive attenuation (Header.Toltec.DriveAtten)
    - atten_sense: Sense attenuation (Header.Toltec.SenseAtten)

    Notes
    -----
    This schema maps directly to the raw netCDF structure and does NOT
    include namespacing since it represents the actual file format.
    """

    # Raw I/Q data fields
    I: Mapping = Mapping(("Data.Toltec.Is", "I"))
    Q: Mapping = Mapping(("Data.Toltec.Qs", "Q"))

    # Coordinate fields
    time: Mapping = Mapping(("time", "ntimes"))
    channel: Mapping = Mapping(("iqlen", "loclen", "chan", "channel"))
    frequency: Mapping = Mapping(("sweep", "nsweeps"))

    # Tone/channel axis data
    f_tones: Mapping = Mapping("Header.Toltec.ToneFreq")
    f_los: Mapping = Mapping("Data.Toltec.LoFreq")
    f_lo_center: Mapping = Mapping("Header.Toltec.LoCenterFreq")
    mask_tones: Mapping = Mapping("Header.Toltec.ToneMask")
    amp_tones: Mapping = Mapping(("Header.Toltec.ToneAmp", "Header.Toltec.ToneAmps"))
    phase_tones: Mapping = Mapping("Header.Toltec.TonePhase")

    # Observation metadata
    master: Mapping = Mapping("Header.Toltec.Master")
    roach: Mapping = Mapping("Header.Toltec.RoachIndex")
    obsnum: Mapping = Mapping("Header.Toltec.ObsNum")
    subobsnum: Mapping = Mapping("Header.Toltec.SubObsNum")
    scannum: Mapping = Mapping("Header.Toltec.ScanNum")
    f_smp: Mapping = Mapping("Header.Toltec.SampleFreq")
    obs_type: Mapping = Mapping("Header.Toltec.ObsType")
    kind_str: Mapping = Mapping(("Header.Kids.kind", "kind_str"))
    obs_start_time: Mapping = Mapping("Header.Toltec.ObsStartTime")

    # Instrument settings
    atten_drive: Mapping = Mapping("Header.Toltec.DriveAtten")
    atten_sense: Mapping = Mapping("Header.Toltec.SenseAtten")
    atten_in: Mapping = Mapping(("Header.Toltec.AttenIn", "atten_in"))
    atten_out: Mapping = Mapping(("Header.Toltec.AttenOut", "atten_out"))


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

    def get_metadata(self, data_source: xr.Dataset, block: int | None = None):
        """Get metadata as dataclass instance(s).

        Constructs appropriate metadata dataclass based on data kind.
        For multi-block data, returns list of metadata instances (one per block).

        Parameters
        ----------
        data_source : xr.Dataset
            Dataset to read from
        block : int or None, optional
            If specified, return metadata for specific block only.
            If None, returns single metadata for single-block data or
            list of metadata for multi-block data.

        Returns
        -------
        ToltecSweepMetadata or ToltecTimeStreamMetadata or list
            Metadata instance(s) matching data kind and block structure
        """
        from ..types import ToltecDataKind

        ds = data_source

        # Check if multi-block
        is_multi_block = "block" in ds.dims or ds.attrs.get("is_multi_block", False)
        n_blocks = ds.sizes.get("block", 1) if is_multi_block else 1

        # Validate block parameter
        if block is not None:
            if block < 0 or block >= n_blocks:
                msg = f"Block {block} out of range [0, {n_blocks})"
                raise ValueError(msg)

        # Extract base metadata fields (common to all data kinds)
        base_fields = {}

        # Observation identifiers (use parent class field names)
        if self.schema.roach in self:
            roach_val = self.get_scalar(data_source, self.schema.roach)
            base_fields["roach"] = roach_val
            # nw is alias for roach
            base_fields["nw"] = roach_val
            # interface from roach
            base_fields["interface"] = f"toltec{roach_val}"
            # array_name from roach (TolTEC array mapping)
            if 0 <= roach_val <= 6:  # noqa: PLR2004
                base_fields["array_name"] = "a1100"
            elif 7 <= roach_val <= 10:  # noqa: PLR2004
                base_fields["array_name"] = "a1400"
            elif 11 <= roach_val <= 12:  # noqa: PLR2004
                base_fields["array_name"] = "a2000"

        if self.schema.obsnum in self:
            base_fields["obsnum"] = self.get_scalar(data_source, self.schema.obsnum)
        else:
            base_fields["obsnum"] = 0  # Default if not present

        if self.schema.subobsnum in self:
            base_fields["subobsnum"] = self.get_scalar(
                data_source,
                self.schema.subobsnum,
            )
        else:
            base_fields["subobsnum"] = 0  # Default if not present

        if self.schema.scannum in self:
            base_fields["scannum"] = self.get_scalar(data_source, self.schema.scannum)
        else:
            base_fields["scannum"] = 0  # Default if not present

        # Master controller (from v2 mastervar)
        # Parent class expects string, but v2 uses int - convert to string
        master_val = ds.attrs.get("Header.Toltec.Master", 1)
        if isinstance(master_val, int):
            # Map to master name
            master_map = {1: "tcs", 2: "ics", 3: "clip"}
            base_fields["master"] = master_map.get(master_val, "tcs")
        else:
            base_fields["master"] = str(master_val)

        # Timing fields (required by parent class, use defaults if not available)
        # For now, use placeholder values - these should come from file in real data
        from astropy import units as u
        from astropy.time import Time

        base_fields["t0"] = Time("2000-01-01T00:00:00", format="isot", scale="utc")
        base_fields["t1"] = Time("2000-01-01T00:01:00", format="isot", scale="utc")
        base_fields["t_exp"] = 60.0 * u.s

        # Determine data kind
        data_kind = self._get_data_kind(data_source)

        # Construct metadata based on data kind
        if data_kind in (
            ToltecDataKind.VnaSweep,
            ToltecDataKind.TargetSweep,
            ToltecDataKind.Tune,
            ToltecDataKind.RawSweep,
            ToltecDataKind.ReducedSweep,
        ):
            return self._get_sweep_metadata(
                data_source,
                base_fields,
                is_multi_block,
                n_blocks,
                block,
            )

        # Default to timestream metadata
        return self._get_timestream_metadata(data_source, base_fields)

    def _get_data_kind(self, data_source: xr.Dataset) -> ToltecDataKind:
        """Determine data kind from dataset."""
        from ..types import ToltecDataKind

        # Check for kind_str (reduced/processed data)
        if self.schema.kind_str in self:
            kind_str = self.get_scalar(data_source, self.schema.kind_str)
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
            obs_type_map = {
                0: ToltecDataKind.VnaSweep,
                1: ToltecDataKind.TargetSweep,
                2: ToltecDataKind.Tune,
                3: ToltecDataKind.RawTimeStream,
            }
            return obs_type_map.get(obs_type, ToltecDataKind.Unknown)

        # Infer from structure
        if self.schema.frequency in self:
            return ToltecDataKind.RawSweep
        if self.schema.time in self:
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
        from ..metadata import ToltecSweepMetadata

        ds = data_source

        # Common sweep fields
        sweep_fields = base_fields.copy()

        # Sample frequency (needs astropy unit)
        if self.schema.f_smp in self:
            from astropy import units as u

            f_smp_val = self.get_scalar(data_source, self.schema.f_smp)
            sweep_fields["f_smp"] = f_smp_val * u.Hz

        # Get n_chans from I data shape
        if self.schema.I in self:
            i_data = self.get_arr(data_source, self.schema.I)
            sweep_fields["n_chans"] = i_data.sizes[i_data.dims[0]]

        # Get n_times from data shape (time dimension)
        if self.schema.I in self:
            i_data = self.get_arr(data_source, self.schema.I)
            # Find time/sample dimension (last dimension typically)
            time_dim = i_data.dims[-1] if len(i_data.dims) > 0 else None
            if time_dim:
                sweep_fields["n_times"] = i_data.sizes[time_dim]

        # Get n_sweepsteps
        if self.schema.frequency in self:
            freq = self.get_arr(data_source, self.schema.frequency)
            sweep_fields["n_sweepsteps"] = freq.sizes[freq.dims[0]]
        elif "Header.Toltec.NumSweepSteps" in ds.attrs:
            sweep_fields["n_sweepsteps"] = ds.attrs["Header.Toltec.NumSweepSteps"]
        elif "nsweeps" in ds.dims:  # For reduced sweeps
            sweep_fields["n_sweepsteps"] = ds.sizes["nsweeps"]

        # Get n_sweepreps (for raw sweeps)
        if "Header.Toltec.NumSamplesPerSweepStep" in ds.attrs:
            sweep_fields["n_sweepreps"] = ds.attrs[
                "Header.Toltec.NumSamplesPerSweepStep"
            ]

        # For multi-block data
        if is_multi_block:
            # If block parameter specified, return single metadata
            if block is not None:
                return self._get_single_block_sweep_metadata(sweep_fields, ds, block)

            # Otherwise return list of metadata for all blocks
            return [
                self._get_single_block_sweep_metadata(sweep_fields, ds, i)
                for i in range(n_blocks)
            ]

        # Single-block data
        sweep_fields["n_sweeps"] = 1  # v2 convention: n_sweeps=1 for single block
        sweep_fields["n_blocks"] = 1

        # Get f_lo_center
        if "Header.Toltec.LoCenterFreq" in ds.attrs:
            sweep_fields["f_lo_center"] = ds.attrs["Header.Toltec.LoCenterFreq"]
        elif self.schema.f_los in self:
            f_los = self.get_arr(data_source, self.schema.f_los)
            sweep_fields["f_lo_center"] = float(f_los.mean().values)

        return ToltecSweepMetadata(**sweep_fields)

    def _get_single_block_sweep_metadata(self, base_fields, ds, block_idx):
        """Get sweep metadata for a specific block."""
        from ..metadata import ToltecSweepMetadata

        block_fields = base_fields.copy()

        # Block-specific fields
        block_fields["n_sweeps"] = 1  # Each block is one sweep in v2 convention
        block_fields["n_blocks"] = ds.sizes["block"]
        block_fields["n_blocks_max"] = ds.sizes.get(
            "numSweeps",
            block_fields["n_blocks"],
        )

        # Get f_lo_center for this block
        if self.schema.f_los in self:
            f_los = self.get_arr(ds, self.schema.f_los)
            if "block" in f_los.dims:
                block_f_los = f_los.isel(block=block_idx)
                block_fields["f_lo_center"] = float(block_f_los.mean().values)
        elif "Header.Toltec.LoCenterFreq" in ds.attrs:
            block_fields["f_lo_center"] = ds.attrs["Header.Toltec.LoCenterFreq"]

        return ToltecSweepMetadata(**block_fields)

    def _get_timestream_metadata(self, data_source, base_fields):
        """Construct timestream metadata instance."""
        from ..metadata import ToltecTimeStreamMetadata

        ds = data_source

        ts_fields = base_fields.copy()

        # Sample frequency (needs astropy unit)
        if self.schema.f_smp in self:
            from astropy import units as u

            f_smp_val = self.get_scalar(data_source, self.schema.f_smp)
            ts_fields["f_smp"] = f_smp_val * u.Hz

        # Get n_chans from I data shape
        if self.schema.I in self:
            i_data = self.get_arr(data_source, self.schema.I)
            ts_fields["n_chans"] = i_data.sizes[i_data.dims[0]]

        # Get n_times from time coordinate or data shape
        if self.schema.time in self:
            time = self.get_arr(data_source, self.schema.time)
            ts_fields["n_times"] = time.sizes[time.dims[0]]
        elif self.schema.I in self:
            i_data = self.get_arr(data_source, self.schema.I)
            # Last dimension is typically time
            if len(i_data.dims) > 1:
                ts_fields["n_times"] = i_data.sizes[i_data.dims[-1]]

        return ToltecTimeStreamMetadata(**ts_fields)

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
            name = self.get_name(self.schema.f_lo_center)
            if name in data_source.attrs:
                return float(data_source.attrs[name])
            if name in data_source:
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
        from astropy import units as u

        n_chans = 0
        if self.schema.I in self:
            i_data = self.get_arr(data_source, self.schema.I)
            n_chans = i_data.sizes[i_data.dims[0]]

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
            table["tone_amp"] = self.get_arr(
                data_source, self.schema.amp_tones
            ).values
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
        from astropy import units as u

        ds = data_source
        is_sweep = (
            "sweep" in ds.dims
            or "nsweeps" in ds.dims
            or "frequency" in ds.coords
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
            xarray_obj.dataset
            if isinstance(xarray_obj, xr.DataTree)
            else xarray_obj
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
        if not self.mapper.schema.f_tones in self.mapper:
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
        if not self.mapper.schema.f_los in self.mapper:
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
        if not self.mapper.schema.mask_tones in self.mapper:
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
        if not self.mapper.schema.amp_tones in self.mapper:
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
        if not self.mapper.schema.phase_tones in self.mapper:
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
