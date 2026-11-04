"""Native TolTEC KIDs acquisition metadata and array access."""

# ruff: noqa: E741 -- Preserve the physical I/Q schema names.

from __future__ import annotations

import functools
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import xarray as xr
from astropy import units as u
from astropy.time import Time
from tollan.accessor import FieldLocation, NameMapping, Schema
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper, unwrap_data_source

from ...core.source.opener import _open_netcdf
from ..axes import ToltecAxisView
from ..metadata import ToltecProcessedKidsMetadata, ToltecRawObsMetadata
from ..toltec_db import ToltecDBRawObsMasterCode
from ..types import ToltecDataKind, ToltecDataModel, ToltecInfo
from .blocks import ToltecBlockIndex, _find_lo_block_slices

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from astropy.table import QTable
    from tollan.accessor import Mapping
    from tollan.utils.fileloc import FileLoc

    class ToltecKidsDataset(xr.Dataset):
        """Dataset with a typed TolTEC KIDs accessor."""

        toltec_kids: ToltecKidsAccessor

    class ToltecKidsDataTree(xr.DataTree):
        """DataTree with a typed TolTEC KIDs accessor."""

        toltec_kids: ToltecKidsAccessor


__all__ = [
    "ToltecKidsAccessor",
    "ToltecKidsIOMapper",
    "ToltecKidsIOSchema",
    "open_toltec_kids",
]

_TIMESTREAM_LO_CHECK_SAMPLES = 64


@dataclass
class ToltecKidsIOSchema(Schema):
    """Schema for TolTEC KIDs acquisition data."""

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
    f_sweep: Mapping = NameMapping.var(("sweep", "nsweeps", "frequency"))
    f_lo_center: Mapping = NameMapping("Header.Toltec.LoCenterFreq")
    mask_tones: Mapping = NameMapping.var("Header.Toltec.ToneMask")
    amp_tones: Mapping = NameMapping.var(
        ("Header.Toltec.ToneAmp", "Header.Toltec.ToneAmps")
    )
    phase_tones: Mapping = NameMapping.var("Header.Toltec.TonePhase")

    # Recorded per-channel model parameters
    model_params: Mapping = NameMapping.var("Header.Toltec.ModelParams")
    model_params_header: Mapping = NameMapping.var("Header.Toltec.ModelParamsHeader")

    # Observation metadata
    master_code: Mapping = NameMapping("Header.Toltec.Master")
    readout_code: Mapping = NameMapping("Header.Toltec.RoachIndex")
    obsnum: Mapping = NameMapping("Header.Toltec.ObsNum")
    subobsnum: Mapping = NameMapping("Header.Toltec.SubObsNum")
    scannum: Mapping = NameMapping("Header.Toltec.ScanNum")
    f_smp: Mapping = NameMapping("Header.Toltec.SampleFreq")
    obs_type_code: Mapping = NameMapping("Header.Toltec.ObsType")
    kind_str: Mapping = NameMapping(("Header.Kids.kind", "kind_str"))
    processed_source: Mapping = NameMapping(("Header.Kids.source", "source"))
    reduction_contract: Mapping = NameMapping.attr("contract")
    reduction_kind: Mapping = NameMapping.attr("kind")
    obs_start_time: Mapping = NameMapping("Header.Toltec.ObsStartTime")
    obs_end_time: Mapping = NameMapping("Header.Toltec.ObsEndTime")
    cal_obsnum: Mapping = NameMapping("Header.Toltec.TargSweepObsNum")
    cal_subobsnum: Mapping = NameMapping("Header.Toltec.TargSweepSubObsNum")
    cal_scannum: Mapping = NameMapping("Header.Toltec.TargSweepScanNum")
    n_chans_max: Mapping = NameMapping("Header.Toltec.MaxNumTones")
    filepath_orig: Mapping = NameMapping("Header.Toltec.Filename")
    repeat: Mapping = NameMapping("Header.Toltec.RepeatLevel")
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


class ToltecKidsIOMapper(XarrayMapper[ToltecKidsIOSchema]):
    """Mapper for TolTEC KIDs acquisition data.

    Examples
    --------
    With the reference observation under ``data_lmt`` in the working directory:

    >>> import xarray as xr
    >>> from tolteca_datamodels.toltec.kids import ToltecKidsIOMapper
    >>> filepath = (
    ...     "data_lmt/toltec/tcs/toltec0/"
    ...     "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc"
    ... )
    >>> with xr.open_dataset(filepath) as ds:
    ...     mapper = ToltecKidsIOMapper.from_data_source(ds)
    ...     readout_code = mapper.get_scalar(ds, mapper.schema.readout_code)
    ...     f_tone = mapper.get_arr(ds, mapper.schema.f_tones).load()
    ...     meta = mapper.get_metadata(ds)
    >>> meta.obsnum, int(readout_code)
    (145606, 0)
    """

    def get_metadata(self, data_source: xr.Dataset) -> ToltecRawObsMetadata:
        """Read acquisition metadata without loading sample arrays.

        Parameters
        ----------
        data_source : xarray.Dataset
            Unpartitioned raw KIDs acquisition.

        Returns
        -------
        ToltecRawObsMetadata
            Recorded acquisition settings and dimension counts. Block discovery
            and axis data require an explicit `get_blocks` call.

        Raises
        ------
        ValueError
            If the input is processed or explicitly block-indexed data.
        """
        ds = data_source
        if self.has_dim(ds, self.schema.n_blocks):
            msg = "Acquisition metadata requires an unpartitioned acquisition"
            raise ValueError(msg)
        kind = self._get_data_kind(ds)
        if not kind & ToltecDataKind.raw_kids_data:
            msg = "Acquisition metadata requires a raw KIDs acquisition"
            raise ValueError(msg)
        values = self._get_metadata_fields(ds, is_raw=True)
        values.update(self._get_sample_metadata_fields(ds, values, is_raw=True))
        return ToltecRawObsMetadata(**values)

    def get_blocks(
        self,
        data_source: xr.Dataset,
        *,
        acquisition: ToltecRawObsMetadata | None = None,
    ) -> ToltecBlockIndex:
        """Read the block index and columnar channel/probe axes.

        Parameters
        ----------
        data_source : xarray.Dataset
            Native acquisition or supported indexed/processed KIDs dataset.
        acquisition : ToltecRawObsMetadata, optional
            Already-read acquisition headers for this unpartitioned raw source.

        Returns
        -------
        ToltecBlockIndex
            Recorded index, columnar axes and acquisition metadata enriched with
            block counts. No processing tree or detector-data products are created.

        Notes
        -----
        Current-file discovery preserves tune LO-reset boundaries. Mixed-file
        sweep/timestream boundary detection is deferred. Axis preparation reads
        tone headers and LO samples, but no detector I/Q or ADC arrays.
        Nominal/timestream acquisitions check bounded LO windows at each end
        and assume a constant interior when those samples agree. Disagreement
        falls back to a full LO scan.
        """
        from tolteca_kids.block_identify import BlockIdentify  # noqa: PLC0415

        return BlockIdentify._identify(self, data_source, acquisition=acquisition)

    @classmethod
    def _empty_axis(cls, dim):
        integer_names = (
            ("block", "chan")
            if dim == "chan_entry"
            else ("block", "step", "n_samples", "sample_start", "sample_end")
        )
        columns = {name: np.array([], dtype=np.int64) for name in integer_names}
        frequency_names = (
            ("f_tone", "f_chan") if dim == "chan_entry" else ("f_lo", "f_sweep")
        )
        columns.update({name: np.array([]) * u.Hz for name in frequency_names})
        if dim == "chan_entry":
            columns.update(
                tone_mask=np.array([], dtype=bool),
                tone_amp=np.array([]),
                tone_phase=np.array([]) * u.rad,
            )
        return cls._axis_dataset(columns, dim=dim)

    def _get_block_indexers(self, ds, kind, *, f_los=None):
        """Locate acquired blocks without reading detector arrays."""
        schema = self.schema
        if self.has_dim(ds, schema.n_blocks):
            name = self.get_name(schema.n_blocks)
            return [{name: index} for index in range(self.get_dim(ds, schema.n_blocks))]
        if kind != ToltecDataKind.tune:
            return [{}]
        los = (
            f_los if f_los is not None else self.get_arr(ds, schema.f_los)
        ).to_numpy()
        name = self.get_name(schema.n_times)
        return [{name: part} for part in _find_lo_block_slices(los, split_resets=True)]

    def _get_metadata_fields(self, ds, *, is_raw):
        """Read common observation and readout fields."""
        schema = self.schema
        readout_code = self.get_scalar(ds, schema.readout_code)
        # Current KIDs readout codes coincide with detector-board indices.
        roach = readout_code
        interface = ToltecInfo.roach_interface[roach]
        identity = ToltecInfo.interface_interface_info[interface]
        array_name = ToltecInfo.interface_array_name[interface]
        master_code = ToltecDBRawObsMasterCode(self.get_scalar(ds, schema.master_code))
        t0 = Time(self.get_scalar(ds, schema.obs_start_time), format="unix")
        t1 = Time(self.get_scalar(ds, schema.obs_end_time), format="unix")
        f_smp = self._get_frequency(ds, schema.f_smp)
        if f_smp <= 0 * u.Hz:
            msg = "Sample frequency must be positive"
            raise ValueError(msg)
        base_fields: dict[str, Any] = {
            "readout_code": readout_code,
            "roach": roach,
            "nw": roach,
            "cal_roach": roach,
            "interface": interface,
            "instru": identity.instru,
            "data_models": identity.data_models,
            "interface_kind": identity.interface_kind,
            "array_name": array_name,
            "array": ToltecInfo.array_name_array[array_name],
            "master_code": master_code,
            "master": ToltecInfo.master_code_master[master_code],
            "obsnum": self.get_scalar(ds, schema.obsnum),
            "subobsnum": self.get_scalar(ds, schema.subobsnum),
            "scannum": self.get_scalar(ds, schema.scannum),
            "t0": t0,
            "t1": t1,
            "n_chans": self.get_dim(ds, schema.n_chans),
            "f_smp": f_smp,
        }
        # A raw sweep's shape cannot distinguish tune from VNA/target. Its
        # recorded type determines how acquired blocks are counted.
        if is_raw or schema.obs_type_code in self:
            code = self.get_scalar(ds, schema.obs_type_code)
            base_fields["obs_type_code"] = code
            base_fields["obs_type"] = ToltecInfo.obs_type_code_obs_type[code]
        for key in (
            "atten_drive",
            "atten_sense",
            "cal_obsnum",
            "cal_subobsnum",
            "cal_scannum",
            "n_chans_max",
            "repeat",
        ):
            field = getattr(schema, key)
            if is_raw or field in self:
                base_fields[key] = self.get_scalar(ds, field)
        if is_raw or schema.n_kids_design in self:
            base_fields["n_kids_design"] = self.get_dim(ds, schema.n_kids_design)
        if is_raw or schema.filepath_orig in self:
            base_fields["filepath_orig"] = self.get_str(ds, schema.filepath_orig)

        if is_raw or schema.n_blocks_max in self:
            base_fields["n_blocks_max"] = self.get_dim(ds, schema.n_blocks_max)
        else:
            base_fields["n_blocks_max"] = (
                self.get_dim(ds, schema.n_blocks) if schema.n_blocks in self else 1
            )
        if (
            is_raw and base_fields.get("obs_type") in {"vna", "targ", "tune"}
        ) or schema.n_sweepreps in self:
            base_fields["n_sweepreps"] = self.get_scalar(ds, schema.n_sweepreps)
        return base_fields

    def _get_kidscpp_string(self, dataset: xr.Dataset, field: Mapping) -> str:
        # Legacy NetCDF headers contain C strings in fixed-size buffers. Bytes
        # after the first NUL can be uninitialized and cannot be decoded as UTF-8.
        if self.has_arr(dataset, field):
            value = self.get_arr(dataset, field)
            if value.dtype.kind == "S" and (
                value.ndim == 0 or (value.dtype == "S1" and value.ndim == 1)
            ):
                return value.to_numpy().tobytes().partition(b"\0")[0].decode().strip()
        return self.get_str(dataset, field)

    def get_processed_metadata(
        self, dataset: xr.Dataset
    ) -> ToltecProcessedKidsMetadata | None:
        """Read the recorded acquisition reference without accessing its target."""
        if self.schema.processed_source not in self:
            return None
        source = self._get_kidscpp_string(dataset, self.schema.processed_source)
        if not source:
            return None
        return ToltecProcessedKidsMetadata(
            filepath_orig=source,
            instru=ToltecInfo.instru,
            data_models=frozenset({ToltecDataModel.name}),
        )

    def _get_processed_kind(self, data_source: xr.Dataset) -> ToltecDataKind | None:
        """Return the processed classification when the writer declares a product."""
        if self.schema.kind_str in self:
            kind = self._get_kidscpp_string(data_source, self.schema.kind_str)
        elif self.schema.reduction_contract in self:
            if (
                self.get_str(data_source, self.schema.reduction_contract)
                != "kids.reduction/1"
            ):
                return ToltecDataKind.unknown
            if self.schema.reduction_kind not in self:
                return ToltecDataKind.unknown
            kind = self.get_str(data_source, self.schema.reduction_kind)
        else:
            return None
        return {
            "d21": ToltecDataKind.d21,
            "processed_sweep": ToltecDataKind.reduced_sweep,
            "ReducedSweep": ToltecDataKind.reduced_sweep,
            "processed_timestream": ToltecDataKind.solved_timestream,
            "SolvedTimeStream": ToltecDataKind.solved_timestream,
        }.get(kind, ToltecDataKind.unknown)

    def _get_data_kind(self, data_source: xr.Dataset) -> ToltecDataKind:
        """Classify the native acquisition or processed product."""
        if (kind := self._get_processed_kind(data_source)) is not None:
            return kind

        # Auxiliary readouts share acquisition headers, including ObsType.
        # A raw KIDs classification additionally requires the I/Q payload pair.
        if not (self.schema.I in self and self.schema.Q in self):
            return ToltecDataKind.unknown

        # Decode the recorded acquisition type before classifying raw KIDs data.
        if self.schema.obs_type_code in self:
            obs_type_code = self.get_scalar(data_source, self.schema.obs_type_code)
            obs_type = ToltecInfo.obs_type_code_obs_type.get(obs_type_code)
            return ToltecInfo.obs_type_data_kind.get(obs_type, ToltecDataKind.unknown)

        if self.schema.n_sweepsteps in self:
            return ToltecDataKind.raw_sweep
        if self.schema.n_times in self:
            return ToltecDataKind.raw_timestream

        return ToltecDataKind.unknown

    def _get_sample_metadata_fields(self, ds, base_fields, *, is_raw):
        """Read sample counts, integration time and the scalar LO center."""
        # Only processed sweeps may replace acquisition samples with sweep steps.
        n_times = (
            base_fields["n_sweepsteps"]
            if (
                not is_raw
                and self.schema.n_times not in self
                and "n_sweepsteps" in base_fields
            )
            else self.get_dim(ds, self.schema.n_times)
        )
        fields = {
            "n_times": n_times,
            "t_int": (n_times / base_fields["f_smp"]).to(u.s),
        }
        if is_raw or self.schema.f_lo_center in self:
            fields["f_lo_center"] = self._get_frequency(
                ds, self.schema.f_lo_center
            ).to_value(u.Hz)
        return fields

    def _get_frequency(self, ds, field):
        """Read a scalar frequency, using Hz when the header has no unit."""
        value = self.get_scalar(ds, field)
        unit = (
            self.get_arr(ds, field).u.unit or u.Hz if self.has_arr(ds, field) else u.Hz
        )
        return u.Quantity(value, unit).to(u.Hz)

    def _get_f_lo_center(self, data_source: xr.Dataset) -> float | None:
        """Read the LO center in Hz, falling back to the mean acquired LO."""
        field = self.schema.f_lo_center
        if field in self:
            if self.has_arr(data_source, field):
                arr = self.get_arr(data_source, field)
                return u.Quantity(arr.to_numpy().mean(), arr.u.unit or u.Hz).to_value(
                    u.Hz
                )
            return self._get_frequency(data_source, field).to_value(u.Hz)
        if self.schema.f_los in self:
            arr = self.get_arr(data_source, self.schema.f_los)
            return u.Quantity(arr.to_numpy().mean(), arr.u.unit or u.Hz).to_value(u.Hz)
        return None

    def get_model_params(
        self, data_source: xr.Dataset, *, block: int = 0
    ) -> xr.Dataset | None:
        """Project recorded model parameters into lazy channel columns.

        Parameters
        ----------
        data_source : xarray.Dataset
            Acquisition or already-selected native block data.
        block : int, optional
            Zero-based header row to select when the block dimension remains;
            default is 0. Ignored for an already-selected block.

        Returns
        -------
        xarray.Dataset or None
            Recorded parameter columns in header order, with original channel
            IDs as the ``chan`` coordinate. Only column names are loaded here.
            No units are inferred. None means both model headers are absent.

        Raises
        ------
        ValueError
            Headers are incomplete, malformed, or have inconsistent dimensions.
        """
        schema = self.schema
        present = [
            field in self for field in (schema.model_params, schema.model_params_header)
        ]
        if not any(present):
            return None
        if not all(present):
            msg = "Model parameters require both ModelParams and ModelParamsHeader"
            raise ValueError(msg)
        header = self.get_arr(data_source, schema.model_params_header)
        # xarray's netCDF decoder joins the native trailing character dimension.
        if header.ndim != 1 or header.dtype.kind not in "SU":
            msg = "ModelParamsHeader must be a one-dimensional string array"
            raise ValueError(msg)
        names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in header.values
        ]
        if any(not name or name == "chan" for name in names) or len(set(names)) != len(
            names
        ):
            msg = "ModelParamsHeader requires nonempty unique names other than 'chan'"
            raise ValueError(msg)
        params = self.get_arr(data_source, schema.model_params)
        if schema.n_blocks_max in self:
            dim = self.get_name(schema.n_blocks_max)
            if dim in params.dims:
                params = params.isel({dim: block})
        param_dim = header.dims[0]
        chan_dim = self.get_name(schema.n_chans)
        if (
            params.ndim != 2  # noqa: PLR2004 -- parameter and channel axes
            or params.dims[0] != param_dim
            or params.sizes[param_dim] != len(names)
            or params.shape[1] != data_source.sizes[chan_dim]
        ):
            msg = "ModelParams dimensions must match names and readout channels"
            raise ValueError(msg)
        columns = {
            name: xr.DataArray(params.isel({param_dim: i}).variable, dims="chan")
            for i, name in enumerate(names)
        }
        # Parameters have individually defined physical meanings. The native
        # array has no per-column units; preserve v2's unitless recorded values.
        for column in columns.values():
            column.attrs = {}
        return xr.Dataset(
            columns,
            coords={"chan": data_source[chan_dim].values},
            attrs={"parameter_names": names},
        )

    def _read_chan_axis(self, ds, values):
        """Read one tone row, respecting frequency and phase units."""
        schema = self.schema
        n_chans = values["n_chans"]
        table = {
            "block": np.full(n_chans, values["block"]),
            "chan": ds[self.get_name(schema.n_chans)].values,
        }

        def read(field, unit=None):
            arr = self.get_arr(ds, field)
            if schema.n_blocks_max in self:
                dim = self.get_name(schema.n_blocks_max)
                if dim in arr.dims:
                    arr = arr.isel({dim: values["block"]})
            data = arr.to_numpy().reshape(-1)
            if len(data) != n_chans:
                msg = f"Tone axis has {len(data)} channels; expected {n_chans}"
                raise ValueError(msg)
            if unit is not None:
                data = u.Quantity(data, arr.u.unit or unit).to(unit)
            return data

        if schema.f_tones in self:
            table["f_tone"] = read(schema.f_tones, u.Hz)
            if values.get("f_lo_center") is not None:
                table["f_chan"] = table["f_tone"] + values["f_lo_center"] * u.Hz
        table["tone_mask"] = (
            read(schema.mask_tones).astype(bool)
            if schema.mask_tones in self
            else np.ones(n_chans, dtype=bool)
        )
        table["tone_amp"] = read(schema.amp_tones)
        table["tone_phase"] = (
            read(schema.phase_tones, u.rad)
            if schema.phase_tones in self
            else np.zeros(n_chans) * u.rad
        )
        return self._axis_dataset(table, dim="chan_entry")

    @staticmethod
    def _axis_dataset(columns, *, dim):
        return xr.Dataset(
            {
                name: xr.DataArray(
                    np.asarray(value),
                    dims=dim,
                    attrs={"units": str(value.unit)}
                    if isinstance(value, u.Quantity)
                    else {},
                )
                for name, value in columns.items()
            }
        )

    @staticmethod
    def _get_constant_timestream_lo(f_los):
        """Return the sampled constant LO in Hz, or None to request a full scan."""
        n_samples = f_los.size
        if not n_samples:
            return None
        dim = f_los.dims[0]
        window = _TIMESTREAM_LO_CHECK_SAMPLES
        if n_samples <= 2 * window:
            samples = f_los.to_numpy()
        else:
            samples = np.concatenate(
                [
                    f_los.isel({dim: slice(0, window)}).to_numpy(),
                    f_los.isel({dim: slice(-window, None)}).to_numpy(),
                ]
            )
        # TODO: Revisit this acquisition-type assumption for mixed-block files;
        # matching endpoints do not establish that the interior is constant.
        if np.all(samples == samples[0]) and np.isfinite(samples[0]):
            return u.Quantity(samples[0], f_los.u.unit or u.Hz).to_value(u.Hz)
        return None

    def _read_probe_axis(self, ds, values, f_los, *, sample_start, is_sweep):
        """Describe acquired LO steps with exclusive acquisition sample bounds."""
        center = values.get("f_lo_center")
        constant_lo = None
        if f_los is not None and f_los.ndim != 1:
            msg = "Probe LO frequencies must be a 1-D array"
            raise ValueError(msg)
        if (
            f_los is not None
            and not is_sweep
            and values.get("obs_type") in {"nominal", "timestream"}
        ):
            constant_lo = self._get_constant_timestream_lo(f_los)
        if constant_lo is not None:
            frequencies = np.asarray([constant_lo])
            starts = np.asarray([sample_start])
            counts = np.asarray([values["n_times"]])
        elif f_los is not None:
            los = u.Quantity(f_los.to_numpy(), f_los.u.unit or u.Hz).to_value(u.Hz)
            starts = (
                np.r_[0, np.flatnonzero(np.diff(los) != 0) + 1]
                if los.size
                else np.array([], dtype=int)
            )
            frequencies = los[starts]
            counts = np.diff(np.r_[starts, len(los)])
            starts += sample_start
        elif not is_sweep:
            frequencies = np.asarray([center]) if center is not None else None
            starts = np.asarray([sample_start])
            counts = np.asarray([values["n_times"]])
        else:
            # Processed sweeps store one datum per frequency coordinate.
            if not self.has_arr(ds, self.schema.f_sweep):
                msg = "Could not find sweep LO samples or frequency coordinate"
                raise ValueError(msg)
            coord = self.get_arr(ds, self.schema.f_sweep)
            frequencies = u.Quantity(coord.to_numpy(), coord.u.unit or u.Hz).to_value(
                u.Hz
            )
            if (
                center is not None
                and frequencies.size
                and np.abs(frequencies).max() < 10e6  # noqa: PLR2004
            ):
                frequencies = frequencies + center
            starts = np.arange(len(frequencies)) + sample_start
            counts = np.ones(len(frequencies), dtype=int)
        table = {
            "block": np.full(len(counts), values["block"]),
            "step": np.arange(len(counts)),
            "n_samples": counts,
            "sample_start": starts,
            "sample_end": starts + counts,
        }
        if frequencies is not None:
            table["f_lo"] = frequencies * u.Hz
            if center is not None:
                table["f_sweep"] = (frequencies - center) * u.Hz
        return self._axis_dataset(table, dim="probe")

    def get_chan_axis_table(self, data_source: xr.Dataset, block: int = 0) -> QTable:
        """Read the channel axis for one acquired block.

        Parameters
        ----------
        data_source : xarray.Dataset
            Acquisition containing tone headers.
        block : int, optional
            Zero-based block index; default is 0.

        Returns
        -------
        astropy.table.QTable
            Channel indices, tone/channel frequencies in Hz, masks, amplitudes
            and phases in radians.
        """
        return _block_axis_table(self.get_blocks(data_source), "chan", block)

    def get_probe_axis_table(self, data_source: xr.Dataset, block: int = 0) -> QTable:
        """Read the probe axis for one acquired block.

        Parameters
        ----------
        data_source : xarray.Dataset
            Raw or processed KIDs acquisition.
        block : int, optional
            Zero-based block index; default is 0.

        Returns
        -------
        astropy.table.QTable
            LO frequencies and offsets in Hz, actual sample counts, and
            inclusive start/exclusive end indices in acquisition samples.

        Raises
        ------
        ValueError
            If the block is out of range.
        """
        return _block_axis_table(self.get_blocks(data_source), "probe", block)


def _block_axis_table(index, axis, block):
    if not 0 <= block < index.meta.n_blocks:
        msg = f"Block {block} out of range for this acquisition"
        raise ValueError(msg)
    meta = index.meta
    table = ToltecAxisView(getattr(index, axis)).select_block(block).axis_table
    table.meta.update(
        roach=meta.roach,
        array_name=meta.array_name,
        block=block,
        f_lo_center=meta.f_lo_center,
    )
    return table


def open_toltec_kids(
    source: str | Path | FileLoc,
    *,
    decode_coords: Literal["coordinates", "all"] | bool | None = "all",
    decode_times: bool = True,
    **kwargs,
) -> ToltecKidsDataset:
    """Open a recognized TolTEC KIDs netCDF file.

    Parameters
    ----------
    source : str, pathlib.Path or FileLoc
        Local netCDF file. Remote files must be staged before opening.
    decode_coords : {"coordinates", "all"}, bool or None, optional
        Coordinate decoding passed to `xarray.open_dataset`. Default is "all".
    decode_times : bool, optional
        Whether to decode time variables. Default is True.
    **kwargs
        Additional arguments passed to `xarray.open_dataset`.

    Returns
    -------
    ToltecKidsDataset
        Dataset with an initialized ``toltec_kids`` accessor and a recognized
        data kind. Opening reads classification information, not full metadata
        or signal arrays.
        Close the dataset after loading the arrays needed by the caller.

    Raises
    ------
    ValueError
        If the location is remote or the dataset is not recognized as KIDs data.
        The dataset is closed if accessor initialization or classification fails.

    Examples
    --------
    With the reference observation under ``data_lmt`` in the working directory:

    >>> from tolteca_datamodels.toltec import open_toltec_kids
    >>> filepath = (
    ...     "data_lmt/toltec/tcs/toltec0/"
    ...     "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc"
    ... )
    >>> with open_toltec_kids(filepath) as ds:
    ...     meta = ds.toltec_kids.meta
    ...     f_tone = ds.toltec_kids.f_tone.load()
    >>> meta.uid_raw_obs_file
    'tcs-145606-0-1-toltec0'
    """
    with ExitStack() as stack:
        dataset = cast(
            "ToltecKidsDataset",
            stack.enter_context(
                _open_netcdf(
                    source,
                    decode_coords=decode_coords,
                    decode_times=decode_times,
                    **kwargs,
                )
            ),
        )
        if dataset.toltec_kids.data_kind == ToltecDataKind.unknown:
            msg = "Not a recognized TolTEC KIDs dataset"
            raise ValueError(msg)
        stack.pop_all()
    return dataset


# Main Accessor
# =============


@xr.register_dataset_accessor("toltec_kids")
class ToltecKidsAccessor(XarrayAccessorBase[ToltecKidsIOMapper]):
    """Read native TolTEC KIDs headers, arrays and acquisition structure.

    Parameters
    ----------
    data_source : xarray.Dataset or xarray.DataTree
        Open native acquisition at the supplied dataset or current tree node.
    mapper : ToltecKidsIOMapper, optional
        Mapper for this data layout. If omitted, resolve one from the source.
        A supplied mapper can be shared with other sources of the same layout.

    Examples
    --------
    With the reference observation under ``data_lmt`` in the working directory:

    >>> from tolteca_datamodels.toltec import open_toltec_kids
    >>> filepath = (
    ...     "data_lmt/toltec/tcs/toltec0/"
    ...     "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc"
    ... )
    >>> with open_toltec_kids(filepath) as ds:
    ...     meta = ds.toltec_kids.meta
    ...     f_tone = ds.toltec_kids.f_tone.load()
    ...     index = ds.toltec_kids.block_index
    >>> meta.obsnum
    145606
    """

    @property
    def data_source(self) -> xr.Dataset:
        """Native acquisition dataset at the supplied node."""
        return cast("xr.Dataset", unwrap_data_source(self._data_source))

    @functools.cached_property
    def meta(self) -> ToltecRawObsMetadata | ToltecProcessedKidsMetadata | None:
        """Raw acquisition headers or the acquisition reference in a processed file."""
        if self.data_kind & ToltecDataKind.reduced_kids_data:
            return self.mapper.get_processed_metadata(self.data_source)
        return self.mapper.get_metadata(self.data_source)

    @functools.cached_property
    def block_index(self) -> ToltecBlockIndex:
        """Explicit block inspection, reading LO/tone axes but no detector arrays."""
        metadata = (
            None
            if self.mapper.has_dim(self.data_source, self.mapper.schema.n_blocks)
            else self.meta
        )
        acquisition = metadata if isinstance(metadata, ToltecRawObsMetadata) else None
        return self.mapper.get_blocks(self.data_source, acquisition=acquisition)

    # Tone/Channel properties
    # =======================

    @functools.cached_property
    def adc_snap(self) -> xr.DataArray:
        """Lazy ADC snapshot array."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.adc_snap)

    @functools.cached_property
    def f_tone(self) -> xr.DataArray | None:
        """ROACH tone frequencies in Hz."""
        if self.mapper.schema.f_tones not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.f_tones)

    @functools.cached_property
    def f_lo(self) -> xr.DataArray | float | None:
        """LO frequency in Hz."""
        if self.mapper.schema.f_los not in self.mapper:
            return None

        field = self.mapper.schema.f_los
        if self.mapper.has_arr(self.data_source, field):
            return self.mapper.get_arr(self.data_source, field)
        return self.mapper.get_scalar(self.data_source, field)

    @functools.cached_property
    def f_chan(self) -> xr.DataArray | None:
        """Channel frequencies in Hz."""
        f_tone = self.f_tone
        if f_tone is None:
            return None
        f_lo_center = self.mapper._get_f_lo_center(self.data_source)
        if f_lo_center is None:
            return None
        return f_tone + f_lo_center

    @functools.cached_property
    def tone_mask(self) -> xr.DataArray | None:
        """Tone mask, with True indicating an enabled tone."""
        if self.mapper.schema.mask_tones not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.mask_tones)

    @functools.cached_property
    def tone_amp(self) -> xr.DataArray | None:
        """Tone amplitudes."""
        if self.mapper.schema.amp_tones not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.amp_tones)

    @functools.cached_property
    def tone_phase(self) -> xr.DataArray | None:
        """Tone phases in radians."""
        if self.mapper.schema.phase_tones not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.phase_tones)

    # Data Kind Identification
    # ========================

    @functools.cached_property
    def data_kind(self) -> ToltecDataKind:
        """Native file classification from the recorded headers."""
        return self.mapper._get_data_kind(self.data_source)

    # Axis Data Methods
    # =================

    def get_chan_axis_table(self, block: int = 0) -> QTable:
        """Get channel axis data as an astropy QTable.

        Parameters
        ----------
        block : int, optional
            Zero-based block index; default is 0.

        Returns
        -------
        QTable
            Table with channel metadata including:
            - chan: Channel index
            - f_tone: Tone frequency
            - f_chan: Channel frequency (tone + LO center)
            - tone_mask: Enabled/disabled mask
            - tone_amp: Tone amplitude (if available)
            - tone_phase: Tone phase (if available)

        Examples
        --------
        With the reference observation under ``data_lmt`` in the working directory:

        >>> from tolteca_datamodels.toltec import open_toltec_kids
        >>> filepath = (
        ...     "data_lmt/toltec/tcs/toltec0/"
        ...     "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc"
        ... )
        >>> with open_toltec_kids(filepath) as ds:
        ...     chan_data = ds.toltec_kids.get_chan_axis_table()
        >>> frequencies = chan_data["f_chan"]
        >>> enabled = chan_data[chan_data["tone_mask"]]
        """
        return _block_axis_table(self.block_index, "chan", block)

    def get_probe_axis_table(self, block: int = 0) -> QTable:
        """Get probe axis data as an astropy QTable.

        Parameters
        ----------
        block : int, optional
            Zero-based block index; default is 0.

        Returns
        -------
        QTable
            Table with probe step metadata including:
            - step: Block-local probe step index
            - f_sweep: Sweep frequency offset from LO center
            - f_lo: LO frequency at this sweep step
            - n_samples: Actual samples acquired at this step
            - sample_start, sample_end: Acquisition sample bounds (end exclusive)

        Raises
        ------
        ValueError
            If the block is out of range

        Examples
        --------
        With the reference observation under ``data_lmt`` in the working directory:

        >>> from tolteca_datamodels.toltec import open_toltec_kids
        >>> filepath = (
        ...     "data_lmt/toltec/tcs/toltec0/"
        ...     "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc"
        ... )
        >>> with open_toltec_kids(filepath) as ds:
        ...     sweep_data = ds.toltec_kids.get_probe_axis_table()
        >>> frequencies = sweep_data["f_sweep"]
        """
        return _block_axis_table(self.block_index, "probe", block)

    def select_channels(self, query: str | slice | list | np.ndarray) -> xr.Dataset:
        """Select channels by various criteria.

        Parameters
        ----------
        query : str, slice, list, or ndarray
            Channel selection criteria:
            - str: Name of a boolean channel-table column (e.g., "tone_mask")
            - slice: Index slice (e.g., slice(0, 10))
            - list/array: List of channel indices or boolean mask

        Returns
        -------
        xr.Dataset
            Dataset with selected channels and original channel IDs retained
            as dimension coordinates. Repeated selections are positional within
            the current subset; detector and tone/model header rows stay aligned.

        Examples
        --------
        With the reference observation under ``data_lmt`` in the working directory,
        select enabled channels, a positional slice, or individual positions:

        >>> from tolteca_datamodels.toltec import open_toltec_kids
        >>> filepath = (
        ...     "data_lmt/toltec/tcs/toltec0/"
        ...     "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc"
        ... )
        >>> with open_toltec_kids(filepath) as ds:
        ...     ds_enabled = ds.toltec_kids.select_channels("tone_mask")
        ...     ds_first = ds.toltec_kids.select_channels(slice(0, 10))
        ...     ds_subset = ds.toltec_kids.select_channels([0, 5, 10, 15])
        ...     selected_tones = ds_subset.toltec_kids.f_tone.load()
        ...     selected_count = ds_subset.toltec_kids.meta.n_chans
        >>> selected_count
        4
        """
        ds = self.data_source

        chan_dim = self.mapper.get_name(self.mapper.schema.n_chans)
        if chan_dim is None:
            msg = "Missing readout channel dimension"
            raise ValueError(msg)
        # Seed IDs before slicing, so repeated/reordered selections retain them.
        ds = ds.assign_coords({chan_dim: ds[chan_dim]})

        if isinstance(query, str):
            chan_table = self.get_chan_axis_table()
            if query in chan_table.colnames:
                mask = chan_table[query]
                if mask.dtype.kind != "b":
                    msg = f"Channel selection column {query!r} must be boolean"
                    raise ValueError(msg)
                indices = np.where(mask)[0]
            else:
                msg = (
                    f"Query '{query}' not recognized. "
                    f"Available columns: {chan_table.colnames}"
                )
                raise ValueError(msg)
        elif isinstance(query, slice):
            indices = query
        else:
            # List or array
            indices = query

        indexers: dict[Any, Any] = {chan_dim: indices}
        for field in (
            self.mapper.schema.f_tones,
            self.mapper.schema.mask_tones,
            self.mapper.schema.amp_tones,
            self.mapper.schema.phase_tones,
            self.mapper.schema.model_params,
        ):
            if field in self.mapper:
                array = self.mapper.get_arr(ds, field)
                dim = array.dims[-1]
                if ds.sizes[dim] != ds.sizes[chan_dim]:
                    msg = f"Header {array.name!r} does not match readout channels"
                    raise ValueError(msg)
                indexers[dim] = indices
                ds = ds.assign_coords({dim: (dim, ds[chan_dim].values)})
        return ds.isel(indexers)
