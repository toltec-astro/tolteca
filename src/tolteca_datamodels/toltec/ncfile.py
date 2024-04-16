import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, cast

import astropy.units as u
import netCDF4
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.table import Column, QTable
from loguru import logger
from tollan.utils.fileloc import FileLoc
from tollan.utils.fmt import pformat_fancy_index, pformat_yaml
from tollan.utils.general import add_to_dict
from tollan.utils.nc import NcNodeMapper, ncstr
from tollan.utils.np import make_complex

from tolteca_kidsproc.kidsdata.sweep import MultiSweep, Sweep
from tolteca_kidsproc.kidsdata.timestream import MultiTimeStream

from .core import ToltecFileIO, base_doc, format_doc
from .kidsdata import (
    KidsDataAxis,
    KidsDataAxisInfoMixin,
    KidsDataAxisSlicer,
    KidsDataAxisSlicerMeta,
    ToltecDataKind,
)
from .types import DB_RawObsMaster, DB_RawObsType


class _NcFileIOKidsDataAxisSlicerMixin(
    KidsDataAxisInfoMixin,
    metaclass=KidsDataAxisSlicerMeta,
):
    """A helper class to enable slicer interface to `NcFileIO`."""

    _slicer_cls = KidsDataAxisSlicer


# maps of the data record for various types
# the first level maps to the purpose of the mapper,
# the second level maps to the data object class,
# the inner most level maps to the netcdf dataset.
_nc_node_mapper_defs = {
    # this is a dummy mapper to hold the actual opened dataset.
    # this is not used for reading the dataset.
    "ncopen": {},
    # this is mapper to use when data_kind is called
    "identify": {
        ToltecDataKind.ReducedKidsData: {
            "kind_str": "Header.Kids.kind",
            "obs_type": "Header.Toltec.ObsType",
        },
        ToltecDataKind.RawKidsData: {
            "kind_str": "Header.Kids.kind",
            "obs_type": "Header.Toltec.ObsType",
        },
    },
    # this is mapper to use when meta is called
    "meta": {
        ToltecDataKind.KidsData: {
            "fsmp": "Header.Toltec.SampleFreq",
            "f_lo_center": "Header.Toltec.LoCenterFreq",
            "atten_drive": "Header.Toltec.DriveAtten",
            "atten_sense": "Header.Toltec.SenseAtten",
            "roach": "Header.Toltec.RoachIndex",
            "obsnum": "Header.Toltec.ObsNum",
            "subobsnum": "Header.Toltec.SubObsNum",
            "scannum": "Header.Toltec.ScanNum",
            # assoc
            "cal_roach": "Header.Toltec.RoachIndex",
            "cal_obsnum": "Header.Toltec.TargSweepObsNum",
            "cal_subobsnum": "Header.Toltec.TargSweepSubObsNum",
            "cal_scannum": "Header.Toltec.TargSweepScanNum",
        },
        ToltecDataKind.RawKidsData: {
            "n_kids_design": "loclen",
            "n_chans_max": "Header.Toltec.MaxNumTones",
            "filename_orig": "Header.Toltec.Filename",
            "mastervar": "Header.Toltec.Master",
            "repeatvar": "Header.Toltec.RepeatLevel",
            # data shape
            "n_times": "time",
            "n_chans": "iqlen",
        },
        ToltecDataKind.RawSweep: {
            "n_sweepreps": "Header.Toltec.NumSamplesPerSweepStep",
            "n_sweepsteps": "Header.Toltec.NumSweepSteps",
            "n_sweeps_max": "numSweeps",
        },
        ToltecDataKind.ReducedKidsData: {
            "n_chans": "n_chans",
        },
        ToltecDataKind.ReducedSweep: {
            "n_sweepsteps": "nsweeps",
        },
        ToltecDataKind.SolvedTimeStream: {
            "n_times": "ntimes",
        },
    },
    "axis_data": {
        "f_tones": "Header.Toltec.ToneFreq",
        "mask_tones": "Header.Toltec.ToneMask",
        "amp_tones": "Header.Toltec.ToneAmp",
        "amp_tones_deprecated": "Header.Toltec.ToneAmps",
        "phase_tones": "Header.Toltec.TonePhase",
        "f_los": "Data.Toltec.LoFreq",
    },
    "data": {
        ToltecDataKind.RawKidsData: {
            "I": "Data.Toltec.Is",
            "Q": "Data.Toltec.Qs",
        },
        ToltecDataKind.ReducedSweep: {
            "I": "Data.Kids.Is",
            "Q": "Data.Kids.Qs",
        },
        ToltecDataKind.SolvedTimeStream: {
            "r": "Data.Kids.rs",
            "x": "Data.Kids.xs",
        },
    },
    # these are ancillary data items
    "data_extra": {
        "d21": {
            "d21_f": "Data.Kids.d21_fs",
            "d21": "Data.Kids.d21_adiqs",
            "d21_cov": "Data.Kids.d21_adiqscov",
            "d21_mean": "Data.Kids.d21_adiqsmean",
            "d21_std": "Data.Kids.d21_adiqsstd",
        },
        "candidates": {
            "candidates": "Header.Kids.candidates",
        },
        "psd": {
            "f_psd": "Header.Kids.PsdFreq",
            "I_psd": "Data.Kids.ispsd",
            "Q_psd": "Data.Kids.qspsd",
            "phi_psd": "Data.Kids.phspsd",
            "r_psd": "Data.Kids.rspsd",
            "x_psd": "Data.Kids.xspsd",
        },
    },
}


def _create_nc_node_mappers(nc_node_mapper_defs) -> dict:
    """Create node mappers for the inner most level of dict of node_maps."""

    def _get_sub_node(n) -> Any:
        if all(not isinstance(v, dict) for v in n.values()):
            return NcNodeMapper(nc_node_map=n)
        return {k: _get_sub_node(v) for k, v in n.items()}

    return _get_sub_node(nc_node_mapper_defs)


@format_doc(base_doc)
class NcFileIO(ToltecFileIO, _NcFileIOKidsDataAxisSlicerMixin):
    """A class to read data from TolTEC netCDF files."""

    @classmethod
    def _get_source_file_info(cls, source, source_loc):
        if isinstance(source, netCDF4.Dataset):
            file_loc_orig = None
            file_loc = source_loc or source.filepath()
            # TODO: here we always re-open the file when open is requested
            # this allows easier management of the pickle but
            # may not be desired. Need to revisit this later.
            file_obj = None
        elif isinstance(source, str | Path | FileLoc):
            file_loc_orig = source_loc
            file_loc = source
            file_obj = None
        else:
            raise TypeError(f"invalid source {source}")
        file_loc_orig = cls._validate_file_loc(file_loc_orig)
        file_loc = cls._validate_file_loc(file_loc)
        # TODO: add logic to handle remote file
        if file_loc.is_remote():
            raise ValueError(f"remote file is not supported: {file_loc}")
        return {
            "file_loc_orig": file_loc_orig,
            "file_loc": file_loc,
            "file_obj": file_obj,
        }

    def _post_init(self):
        self._node_mappers = _create_nc_node_mappers(_nc_node_mapper_defs)
        super()._post_init()

    # the node mappers holds the actual opened netCDF nodes.
    _node_mappers: dict

    @property
    def node_mappers(self):
        """The tree of low level netCDF dataset mappers."""
        # the opened file is at the root node mapper
        return self._node_mappers

    @property
    def nc_node(self):
        """The underlying netCDF dataset."""
        return self.node_mappers["ncopen"].nc_node

    def open(self):
        """Return the context for netCDF file IO."""
        # we just use one of the node_mapper to open the dataset, and
        # set the rest via set_nc_node
        # the node_mapper.open will handle the different types of source
        if self.io_state.is_open():
            # the file is already open
            return self
        # TODO: here we alwasy open the netcdf regardless if
        # an opened file is passed to source and set as file_obj.
        # need to revisit this later
        nc_node = None

        def _open_sub_node(n):
            nonlocal nc_node
            for v in n.values():
                if isinstance(v, NcNodeMapper):
                    if nc_node is None:
                        v.open(self.filepath)
                        nc_node = v.nc_node
                    else:
                        v.set_nc_node(nc_node)
                    # push to the exit stack
                    self.enter_context(v)
                else:
                    _open_sub_node(v)

        _open_sub_node(self.node_mappers)
        self._set_open_state(nc_node)
        return self

    # a registry to store all data kind identifiers
    # this returns a tuple of (valid, meta)
    _data_kind_identifiers: ClassVar = {}

    @staticmethod
    @add_to_dict(_data_kind_identifiers, ToltecDataKind.RawKidsData)
    def _identify_raw_kids_data(node_mapper):
        nm = node_mapper
        if not nm.has_var("obs_type") or nm.has_var("kind_str"):
            return False, {}

        obs_type = nm.get_scalar("obs_type")
        logger.debug(f"found obs_type -> {nm['obs_type']}: {obs_type}")

        data_kind = DB_RawObsType.get_data_kind(obs_type)
        # here we allow identification of unknown kids data
        # but in this case only minimal meta data will be available
        return True, {
            "obs_type": obs_type,
            "data_kind": data_kind,
        }

    @staticmethod
    @add_to_dict(_data_kind_identifiers, ToltecDataKind.ReducedKidsData)
    def _identify_reduced_kids_data(node_mapper):
        nm = node_mapper
        if not nm.has_var("kind_str") and not nm.has_var("kind_str_deprecated"):
            return False, {}

        kind_str = nm.get_str("kind_str") if nm.has_var("kind_str") else "unknown"
        logger.debug(f"found kind_str={kind_str} from {nm['kind_str']}")

        data_kind = {
            "d21": ToltecDataKind.D21,
            "processed_sweep": ToltecDataKind.ReducedSweep,
            "processed_timestream": ToltecDataKind.SolvedTimeStream,
            "SolvedTimeStream": ToltecDataKind.SolvedTimeStream,
        }.get(kind_str, ToltecDataKind.Unknown)
        return True, {"kind_str": kind_str, "data_kind": data_kind}

    def _load_data_kind_meta_from_io_obj(self):
        meta = self._meta
        for k, m in self.node_mappers["identify"].items():
            valid, _meta = self._data_kind_identifiers[k](m)
            logger.debug(f"check data kind as {k}: {valid} {_meta}")
            if valid:
                meta.update(_meta)
        # remove the unkown entry to trigger error later if needed.
        if meta["data_kind"] == ToltecDataKind.Unknown:
            meta.pop("data_kind")

    def _load_meta_from_io_obj(self):
        # load the full meta data
        meta = self._meta
        data_kind = self.data_kind
        if not self.io_state.is_open():
            # stop if file is not open
            return
        # load data kind specific meta
        for k, m in self.node_mappers["meta"].items():
            if k & data_kind:
                # read all entries in mapper
                for kk in m.nc_node_map:
                    meta[kk] = m.get_value(kk)

        # update meta using the per type registered updater.
        for k, _meta_updater in self._meta_updaters.items():
            if data_kind & k:
                _meta_updater(self)
        logger.debug(f"loaded meta data:\n{pformat_yaml(meta)}")

    # a registry to metadata updaters
    _meta_updaters: ClassVar = {}

    @add_to_dict(_meta_updaters, ToltecDataKind.KidsData)
    def _update_derived_info(self):
        meta = self._meta
        meta["instru"] = "toltec"
        meta["interface"] = f'toltec{meta["roach"]}'
        # TODO: someday we may need to change the mapping between this
        master = meta["master"] = meta.get("mastervar", 1)
        meta["master_name"] = DB_RawObsMaster.get_master_name(master)
        meta["repeat"] = meta.get("repeatvar", 1)
        meta["nw"] = meta["roach"]

    @add_to_dict(_meta_updaters, ToltecDataKind.ReducedSweep)
    def _update_reduced_sweep_block_info(self):
        meta = self._meta
        result = {}
        result["n_sweeps"] = 1
        result["sweep_slices"] = [slice(0, meta["n_sweepsteps"])]
        result["n_blocks_max"] = 1
        result["n_timesperblock"] = meta["n_sweepsteps"]
        result["n_blocks"] = result["n_sweeps"]
        result["block_slices"] = result["sweep_slices"]
        meta.update(result)

    @add_to_dict(_meta_updaters, ToltecDataKind.RawSweep)
    def _update_raw_sweep_block_info(self):
        meta = self._meta
        data_kind = self.data_kind
        # This function is to populate raw sweep block info to meta
        # The block info is a set of meta data that indicate
        # logical unit of the dataset.
        # For sweeps, each block is one monotonic frequency sweep;
        # for timestreams, this could be arbitrary, and is default
        # to one block as of 20200722.

        result = {}

        # this is the theoretical number of samples per sweep block.
        result["n_timespersweep"] = meta["n_sweepsteps"] * meta["n_sweepreps"]
        # we need to load f_los to properly handle the sweeps with potentially
        # missing samples
        nm = self.node_mappers["axis_data"]
        f_los_Hz = nm.get_var("f_los")[:]
        result["f_los"] = f_los_Hz << u.Hz

        # for tune file, we expect multiple sweep blocks
        # because there could be samples missing, the only
        # reliable way to identify the blocks is to
        # check the f_los array looking for monotonically
        # increasing blocks
        # sometimes there could be corrupted data and the f_lo is set to 0.
        # this assert detects such case because otherwise the break
        # indices will be incorrect.
        if not np.all(f_los_Hz > 0):
            raise ValueError("invalid f_lo found in data file.")
        # this gives the index of the first item after a decrease in f_lo
        break_indices = np.where(np.diff(f_los_Hz) < 0)[0] + 1

        if break_indices.size == 0:
            # we have one (maybe incompleted) block
            result["n_sweeps"] = 1
            result["sweep_slices"] = [slice(0, meta["n_times"])]
        else:
            # this can only happend to tune files
            if data_kind != ToltecDataKind.Tune:
                raise ValueError(f"too many blocks found in data kind={data_kind}")
            # the blocks are partitioned by the break indices
            result["n_sweeps"] = break_indices.size + 1
            sweep_starts = (
                [
                    0,
                ]
                + break_indices.tolist()
                + [meta["n_times"]]
            )
            sweep_sizes = np.diff(sweep_starts)
            result["sweep_slices"] = [
                slice(sweep_starts[i], sweep_starts[i + 1])
                for i in range(result["n_sweeps"])
            ]
            if np.any(sweep_sizes < result["n_timespersweep"]):
                incompleted_sweeps = np.where(sweep_sizes < result["n_timespersweep"])[
                    0
                ].tolist()
                logger.warning(
                    (
                        f"missing data in sweep blocks {incompleted_sweeps} "
                        f"sizes={sweep_sizes[incompleted_sweeps]}."
                    ),
                )

        # populate block meta
        # this is the maximum number of blocks
        result["n_blocks_max"] = meta["n_sweeps_max"]
        result["n_timesperblock"] = result["n_timespersweep"]
        result["n_blocks"] = result["n_sweeps"]
        result["block_slices"] = result["sweep_slices"]
        meta.update(result)

    @add_to_dict(
        _meta_updaters,
        ToltecDataKind.ReducedKidsData | ToltecDataKind.TimeStream,
    )
    def _update_raw_timestream_block_info(self):
        result = {}
        result["n_blocks_max"] = 1
        result["n_blocks"] = 1
        self._meta.update(result)

    def _resolve_block_index(self, block_index=None):
        """Return the block info.

        This returns a 3-tuple describe the blocks in the data file as
        ``(iblock, n_blocks, n_blocks_max)``.

        The special `block_index` value ``None`` will resolve to
        the last block corresponding to n_blocks,
        while any integer will be resolved with the range 0 to n_blocks_max.
        """
        meta = self.meta
        n_blocks = meta["n_blocks"]
        n_blocks_max = meta["n_blocks_max"]
        if block_index is None:
            iblock = n_blocks - 1
        else:
            iblock = range(n_blocks_max)[block_index]
        logger.debug(
            (
                f"resolve block_index {block_index} -> iblock={iblock} "
                f"n_blocks={n_blocks} (n_blocks_max={meta['n_blocks_max']})"
            ),
        )
        return iblock, n_blocks, n_blocks_max

    @cached_property
    def _chan_axis_data(self):
        """Returns the channel data.

        This is a list of tables.
        """
        nm = self.node_mappers["axis_data"]
        meta = self.meta
        n_blocks = meta["n_blocks"]
        n_chans = meta["n_chans"]
        v = nm.get_var("f_tones")
        # for multi block data, v should be of 2-dimensional
        # and for single block data, v could be of 2-dimensional or one
        if n_blocks == 1 and len(v.shape) == 1:
            data = v[:].reshape((n_blocks, n_chans))
        else:
            data = v[:n_blocks, :]
        data = data << u.Hz
        assert data.shape == (n_blocks, n_chans)
        if nm.has_var("mask_tones"):
            # with masking
            data_mask = nm.get_var("mask_tones")[:].reshape(data.shape).astype(bool)
            data_amp = nm.get_var("amp_tones")[:].reshape(data.shape)
            data_phase = nm.get_var("phase_tones")[:].reshape(data.shape) << u.rad
        else:
            data_mask = np.ones(data.shape, dtype=bool)
            # the legacy amps value
            if nm.has_var("amp_tones_deprecated"):
                data_amp = nm.get_var("amp_tones_deprecated")[:].reshape(data.shape)
            else:
                data_amp = np.ones(data.shape)
            data_phase = np.zeros(data.shape) << u.rad

        result = []
        for i in range(n_blocks):
            chan_axis_data = QTable()
            chan_axis_data["id"] = Column(
                range(n_chans),
                description="The channel index",
            )
            chan_axis_data["f_tone"] = Column(
                data[i],
                description="The ROACH tone frequency as used in the FFT comb",
            )
            chan_axis_data["f_chan"] = Column(
                data[i] + (meta["f_lo_center"] << u.Hz),
                description="The channel reference frequency.",
            )
            chan_axis_data["mask_tone"] = Column(
                data_mask[i],
                description="The tone mask.",
            )
            chan_axis_data["amp_tone"] = Column(
                data_amp[i],
                description="The tone amplitude.",
            )
            chan_axis_data["phase_tone"] = Column(
                data_phase[i],
                description="The tone phase.",
            )

            cast(dict, chan_axis_data.meta).update(meta)
            cast(dict, chan_axis_data.meta)["block_index"] = i
            result.append(chan_axis_data)
        return result

    def get_chan_axis_data(self, block_index=None):
        """Return the chan axes data at `block_index`."""
        iblock, _, _ = self._resolve_block_index(block_index)
        return self._chan_axis_data[iblock]

    @cached_property
    def _model_params_tables(self):
        """Return the model parameter tables.

        This is a list of tables.
        """
        nc_node = self.nc_node
        meta = self.meta
        n_blocks = meta["n_blocks"]

        model_params_header = ncstr(
            nc_node.variables["Header.Toltec.ModelParamsHeader"],
        )
        result = []
        for i in range(n_blocks):
            # note the first dim is to select the block.
            model_params_data = nc_node.variables["Header.Toltec.ModelParams"][i, :, :]
            model_params_table = QTable(
                data=list(model_params_data),
                names=model_params_header,
            )
            result.append(model_params_table)
        return result

    def get_model_params_table(self, block_index=None):
        """Return the model params table at `block_index`."""
        iblock, _, _ = self._resolve_block_index(block_index)
        return self._model_params_tables[iblock]

    @staticmethod
    def _populate_sweep_axis_data(  # noqa: PLR0913
        sweep_axis_data,
        id,
        f_sweep,
        f_los,
        n_samples,
        sample_start,
        sample_end,
        meta,
        block_index,
    ):
        sweep_axis_data["id"] = Column(id, description="The sweep step index")
        sweep_axis_data["f_lo"] = Column(
            f_los,
            description="The sweeping LO frequency",
        )
        sweep_axis_data["f_sweep"] = Column(
            f_sweep,
            description="The sweep step frequency",
        )
        sweep_axis_data["n_samples"] = Column(
            n_samples,
            description="The number of repeats in a sweep step",
        )
        sweep_axis_data["sample_start"] = Column(
            sample_start,
            description="The sample index at the start of sweep step.",
        )
        sweep_axis_data["sample_end"] = Column(
            sample_end,
            description="The sample index after the end of sweep step.",
        )
        cast(dict, sweep_axis_data.meta).update(meta)
        cast(dict, sweep_axis_data.meta)["block_index"] = block_index

    def _get_raw_sweep_sweep_axis_data(self):
        """Return the sweep steps.

        This is a list of tables.
        """
        meta = self.meta
        f_los_Hz = meta["f_los"].to_value(u.Hz)
        result = []
        for iblock, block_slice in enumerate(meta["block_slices"]):
            sweep_axis_data = QTable()
            # the raw sweep axis is constructed from checking the
            # f_lo frequencies
            uf_los, uif_los, urf_los = np.unique(
                f_los_Hz[block_slice],
                return_index=True,
                return_counts=True,
            )
            # pad the index with block slice start
            uif_los = uif_los + block_slice.start
            f_sweep = uf_los - meta["f_lo_center"]
            self._populate_sweep_axis_data(
                sweep_axis_data,
                id=range(len(uf_los)),
                f_sweep=f_sweep << u.Hz,
                f_los=uf_los << u.Hz,
                n_samples=urf_los,
                sample_start=uif_los,
                sample_end=uif_los + urf_los,
                meta=meta,
                block_index=iblock,
            )
            result.append(sweep_axis_data)
        return result

    def _get_reduced_sweep_sweep_axis_data(self):
        """Return the sweep steps for reduced sweeps.

        This contains one table since the reduced sweep data
        only have one block.
        """
        nm = self.node_mappers["axis_data"]
        meta = self.meta

        f_sweep = nm.get_var("sweeps")[:]
        f_los = f_sweep + meta["f_lo_center"]
        sweep_axis_data = QTable()
        sweep_id = np.arange(len(f_sweep))
        self._populate_sweep_axis_data(
            sweep_axis_data,
            id=sweep_id,
            f_sweep=f_sweep << u.Hz,
            f_los=f_los << u.Hz,
            n_samples=1,
            sample_start=sweep_id,
            sample_end=sweep_id + 1,
            meta=meta,
            block_index=0,
        )
        return [sweep_axis_data]

    @cached_property
    def _sweep_axis_data(self):
        """Return the sweep axis data.

        This is a list containing ``n_blocks`` arrays, each of which
        is of shape (n_sweepsteps, 2), specifying the start and
        end sample indices.
        """
        if KidsDataAxis.Sweep not in self.axis_types:
            return None
        if self.data_kind & ToltecDataKind.RawSweep:
            return self._get_raw_sweep_sweep_axis_data()
        return self._get_reduced_sweep_sweep_axis_data()

    def get_sweep_axis_data(self, block_index=None):
        """Return the sweep axis data at `block_index`."""
        data = self._sweep_axis_data
        if data is None:
            raise RuntimeError(f"no sweep axis for data kind of {self.data_kind}")
        iblock, _, _ = self._resolve_block_index(block_index)
        return data[iblock]

    def read(self, **slicer_args):
        """Read the file and return a data object.

        Parameters
        ----------
        slicer_args : dict
            The arguments to specify the data to load.
            The keys shall be one of the axis names
            block, chan, sweep, time, or sample.
        """
        # create the slicer object
        slicer = self.block_loc(None)
        for t, arg in slicer_args.items():
            slicer = getattr(slicer, f"{t}_loc")(arg)
        return self._read_sliced(slicer)

    def read_meta(self, **slicer_args):
        """Read additional metadata."""
        slicer = self.block_loc(None)
        for t, arg in slicer_args.items():
            slicer = getattr(slicer, f"{t}_loc")(arg)
        return self._read_sliced(slicer, meta_only=True)

    def _resolve_slice(self, slicer):  # noqa: C901, PLR0915
        """Read the file for data specified by the `slicer`."""
        result = {}
        # parse the slicer args and get the data.
        ops: dict[KidsDataAxis, Any] = {}
        result["ops"] = ops
        for t in self.axis_types:
            ops[t] = slicer.get_slice_op(t)

        # do some check
        if (
            sum(
                [
                    ops.get(KidsDataAxis.Sample, None) is not None,
                    ops.get(KidsDataAxis.Sweep, None) is not None,
                ],
            )
            == 2  # noqa: PLR2004
        ):
            raise ValueError("can only slice on one of sample or sweep.")

        if (
            sum(
                [
                    ops.get(KidsDataAxis.Sample, None) is not None,
                    ops.get(KidsDataAxis.Time, None) is not None,
                ],
            )
            == 2  # noqa: PLR2004
        ):
            raise ValueError("can only slice on one of sample or time.")

        logger.debug(f"slicer_ops:\n{pformat_yaml(ops)}")

        # apply the slicer ops
        def slice_table(tbl, op):
            if op is None:
                return tbl, slice(None, None)
            if isinstance(op, str):
                # slice the table with pandas query
                with warnings.catch_warnings():
                    # this is to supress the ufunc size warning
                    # and the numexpr
                    warnings.simplefilter("ignore")
                    tbl_df = tbl.to_pandas()
                    op = tbl_df.eval(op).to_numpy(dtype=bool)
            tbl = tbl[op]
            tbl.meta["_slice_op"] = op
            return tbl, op

        block_index = result["block_index"] = ops[KidsDataAxis.Block]
        chan_axis_data, chan_op = slice_table(
            self.get_chan_axis_data(block_index=block_index),
            ops[KidsDataAxis.Chan],
        )
        # logger.debug(f"sliced chan axis data:\n{chan_axis_data}")
        logger.debug(
            f"sliced {len(chan_axis_data)} chans out of {self.meta['n_chans']}",
        )

        result["chan_axis_data"] = chan_axis_data
        # this will be used to load the data
        result["chan_slice"] = chan_op

        if KidsDataAxis.Sweep in self.axis_types:
            # in this case,
            # we re-build the sample slice from the range of the sliced
            # sweep table
            sweep_axis_data, _ = slice_table(
                self.get_sweep_axis_data(block_index=block_index),
                ops[KidsDataAxis.Sweep],
            )
            # logger.debug(f"sliced sweep axis data:\n{sweep_axis_data}")
            logger.debug(
                (
                    f"sliced {len(sweep_axis_data)} "
                    f"sweep steps out of {self.meta['n_sweepsteps']}"
                ),
            )
            # this will be used to load the data
            # this data will be reduced for each sweep step later
            # TODO: this is less optimal for sweep_op being a mask.
            # but for now the sweep is small and we can afford doing so
            sample_slice = slice(
                np.min(sweep_axis_data["sample_start"]),
                np.max(sweep_axis_data["sample_end"]),
            )
            result["sweep_axis_data"] = sweep_axis_data
        elif KidsDataAxis.Time in self.axis_types:
            # in this case,
            # we build the sample slice from the range of time,
            fsmp_Hz = self.meta["fsmp"]

            def _t_to_sample(t):
                if t is None:
                    return None
                if not isinstance(t, u.Quantity):
                    t = t << u.s
                if t.unit is not None and t.unit.is_equivalent(u.s):
                    return int(t.to_value(u.s) * fsmp_Hz)
                raise ValueError("invalid time loc argument.")

            # we have ensured the time_loc arg is always slice
            # in _KidsDataAxisSlicer
            time_slice = ops[KidsDataAxis.Time]
            _sample_slice = ops[KidsDataAxis.Sample]
            if time_slice is not None:
                sample_slice = list(
                    map(
                        _t_to_sample,
                        [
                            time_slice.start,
                            time_slice.stop,
                            time_slice.step,
                        ],
                    ),
                )
            elif _sample_slice is not None:
                sample_slice = [
                    _sample_slice.start,
                    _sample_slice.stop,
                    _sample_slice.step,
                ]
            else:
                sample_slice = [None, None, None]
            # make sure the step size is at least 1
            if sample_slice[2] is not None and sample_slice[2] < 1:
                raise ValueError("invalid time slice step.")
            sample_slice = slice(*sample_slice)
        else:
            # read from the sample loc
            sample_slice = ops[KidsDataAxis.Sample] or slice(None, None, None)
        result["sample_slice"] = sample_slice
        return result

    def _read_sliced(self, slicer, meta_only=False):  # noqa: C901, PLR0912
        """Read the file for data specified by the `slicer`."""
        s = self._resolve_slice(slicer)
        # now that we have the chan slice and sample slice
        # we can read the data
        data_kind = self.data_kind
        # we create a copy of meta data here to store the slicer info
        meta = self.meta.copy()
        meta.update(s)
        if meta_only:
            return meta
        data = {}
        for k, m in self.node_mappers["data"].items():
            if data_kind & k:
                logger.debug(
                    (
                        f"read data {m.nc_node_map.keys()} sample_slice="
                        f"{pformat_fancy_index(s['sample_slice'])}"
                        " chan_slice="
                        f"{pformat_fancy_index(s['chan_slice'])}"
                    ),
                )
                for key in m.nc_node_map:
                    # we arrange the data so that the data
                    # chan axis is first, as required by the
                    # kidsproc.kidsdata containers
                    data[key] = m.get_var(key)[
                        s["sample_slice"],
                        s["chan_slice"],
                    ].T

        # for reduced sweep it may have d21 data
        if data_kind & ToltecDataKind.ReducedSweep:
            m = self.node_mappers["data_extra"]["d21"]
            logger.debug(f"read extra data {m.nc_node_map.keys()}")
            for key in m.nc_node_map:
                if m.has_var(key):
                    data[key] = m.get_var(key)[:]
        # for vna sweep we can load the candidates list
        if data_kind & ToltecDataKind.ReducedVnaSweep:
            m = self.node_mappers["data_extra"]["candidates"]
            logger.debug(f"read extra data {m.nc_node_map.keys()}")
            for key in m.nc_node_map:
                if m.has_var(key):
                    data[key] = m.get_var(key)[:]

        # for the solved timestreams we also load the psd info if they
        # are available
        if data_kind & ToltecDataKind.SolvedTimeStream:
            m = self.node_mappers["data_extra"]["psd"]
            logger.debug(
                (
                    f"read extra data {m.nc_node_map.keys()}"
                    " chan_slice="
                    f"{pformat_fancy_index(s['chan_slice'])}"
                ),
            )
            for key in m.nc_node_map:
                # we arrange the data so that the data
                # chan axis is first, as required by the
                # kidsproc.kidsdata containers
                if m.has_var(key):
                    v = m.get_var(key)
                    # the psd_fs is vector so skip the slice
                    v = v[:] if len(v.shape) == 1 else v[:, s["chan_slice"]].T
                    data[key] = v
        # for the raw sweeps we do the reduction for each step
        if data_kind & ToltecDataKind.RawSweep:
            sweep_axis_data = s["sweep_axis_data"]
            b0 = s["sample_slice"].start  # this is the ref index
            for k in ("I", "Q"):
                a = np.full(
                    (len(s["chan_axis_data"]), len(sweep_axis_data)),
                    np.nan,
                    dtype="d",
                )
                unc_a = np.full(a.shape, np.nan, dtype="d")
                for i, row in enumerate(sweep_axis_data):
                    i0 = row["sample_start"] - b0
                    i1 = row["sample_end"] - b0
                    a[:, i] = np.mean(data[k][:, i0:i1], axis=-1)
                    unc_a[:, i] = np.std(data[k][:, i0:i1], axis=-1)
                data[k] = a
                data[f"unc_{k}"] = unc_a
        for k, m in self._kidsdata_obj_makers.items():
            if data_kind & k:
                return m(self.__class__, data_kind, meta, data)
        # generic maker, this is just to return the things as a dict
        return {"data_kind": data_kind, "meta": meta, "data": data}

    _kidsdata_obj_makers: ClassVar = {}
    """A registry to store the data obj makers."""

    @classmethod
    @add_to_dict(_kidsdata_obj_makers, ToltecDataKind.Sweep)
    def _make_kidsdata_sweep(cls, _data_kind, meta, data):
        f_chans = meta["chan_axis_data"]["f_chan"]
        f_sweep = meta["sweep_axis_data"]["f_sweep"]
        # we need to pack the I and Q
        S21 = make_complex(data["I"], data["Q"])
        if "unc_I" in data:
            unc_S21 = StdDevUncertainty(make_complex(data["unc_I"], data["unc_Q"]))
        else:
            unc_S21 = None
        result = MultiSweep(
            meta=meta,
            f_chans=f_chans,
            f_sweep=f_sweep,
            S21=S21,
            uncertainty=unc_S21,
        )
        if "d21" in data:
            d21_unit = u.adu / u.Hz  # type: ignore
            # make a unified D21 sweep and set that as the unified
            # data of this multisweep
            result.set_unified(
                Sweep(
                    S21=None,
                    D21=data["d21"] << d21_unit,
                    frequency=data["d21_f"] << u.Hz,
                    extra_attrs={
                        "D21_cov": data["d21_cov"],
                        "D21_mean": data["d21_mean"] << d21_unit,
                        "D21_std": data["d21_std"] << d21_unit,
                    },
                    meta={"candidates": data["candidates"] << u.Hz},
                ),
            )
        return result

    @classmethod
    @add_to_dict(_kidsdata_obj_makers, ToltecDataKind.RawTimeStream)
    def _make_kidsdata_rts(cls, _data_kind, meta, data):
        f_chans = meta["chan_axis_data"]["f_chan"]
        return MultiTimeStream(
            meta=meta,
            f_chans=f_chans,
            I=data["I"],
            Q=data["Q"],
        )

    @classmethod
    @add_to_dict(_kidsdata_obj_makers, ToltecDataKind.SolvedTimeStream)
    def _make_kidsdata_sts(cls, _data_kind, meta, data):
        f_chans = meta["chan_axis_data"]["f_chan"]
        # add the psd data to the meta
        meta = meta.copy()
        for k, v in data.items():
            if k.endswith("_psd"):
                meta[k] = v
        return MultiTimeStream(
            meta=meta,
            f_chans=f_chans,
            r=data["r"],
            x=data["x"],
        )

    def __getstate__(self):
        # need to reset the object before pickling
        is_open = self.io_state.is_open()
        self.close()
        return {
            "_auto_close_on_pickle_wrapped": self.__dict__,
            "_auto_close_on_pickle_is_open": is_open,
        }

    def __setstate__(self, state):
        state, is_open = (
            state["_auto_close_on_pickle_wrapped"],
            state["_auto_close_on_pickle_is_open"],
        )
        self.__dict__.update(state)
        if is_open:
            # try open the object
            self.open()

    @classmethod
    def identify(cls, file_loc, file_obj=None):
        """Return if this class can handle ths given file."""
        if file_obj is not None:
            return isinstance(file_obj, netCDF4.Dataset)
        if file_loc is not None:
            return file_loc.path.suffix == ".nc"
        return False
