from loguru import logger
import numpy as np
import warnings
from functools import cached_property
from astropy.table import QTable, Column
import astropy.units as u
from astropy.nddata import StdDevUncertainty
from .kidsdata import (
    KidsDataAxisSlicer,
    KidsDataAxisSlicerMeta,
    KidsDataKind,
    KidsDataAxis,
)
from ..base import DataFileIO, DataFileIOError
from tollan.utils.nc import NcNodeMapper, NcNodeMapperError, ncstr
from tollan.utils.general import add_to_dict
from tollan.utils.fmt import pformat_yaml, pformat_fancy_index
from tollan.utils.np import make_complex
from ...kidsproc.kidsdata.sweep import MultiSweep, Sweep
from ...kidsproc.kidsdata.timestream import TimeStream


class _NcFileIOKidsDataAxisSlicerMixin(object, metaclass=KidsDataAxisSlicerMeta):
    """A helper class to enable slicer interface to `NcFileIO`."""

    _slicer_cls = KidsDataAxisSlicer


class NcFileIO(DataFileIO, _NcFileIOKidsDataAxisSlicerMixin):
    """A class to read data from TolTEC netCDF files.

    Parameters
    ----------
    loc : str, `pathlib.Path`, `FileLoc`, `netCDF4.Dataset`
        The data file location or netCDF dataset.
        This is passed to `~tollan.utils.nc.NcNodeMapper`.

    open : bool
        If True and `source` is set, open the file at constuction time.
    load_meta_on_open : bool
        If True, the meta data will be loaded upon opening of the file.
    auto_close_on_pickle : bool
        If True, the dataset is automatically closed when pickling.
        This is ignored if `source` is None.
    """

    def __init__(
        self, loc=None, open_=True, load_meta_on_open=True, auto_close_on_pickle=True
    ):
        loc = self._validate_file_loc(loc)
        if loc is None:
            source = None
        else:
            if loc.is_remote:
                raise ValueError("remote source is not supported yet.")
            source = loc.path
        super().__init__(file_loc=loc, source=source, io_obj=None, meta=None)
        # this hold the cached metadata.
        self._meta_cached = {}

        self._load_meta_on_open = load_meta_on_open
        self._auto_close_on_pickle = auto_close_on_pickle
        # setup the mapper for read meta data
        self._node_mappers = self._create_node_mappers(self._node_maps)
        # if source is given, we just open it right away
        if self.source is not None and open_:
            self.open()

    @property
    def node_mappers(self):
        """The tree of low level netCDF dataset mappers."""
        # the opened file is at the root node mapper
        return self._node_mappers

    @property
    def nc_node(self):
        """The underlying netCDF dataset."""
        return self.node_mappers["ncopen"].nc_node

    @property
    def io_obj(self):
        # we expose the raw netcdf dataset as the low level file object.
        # this returns None if no dataset is open.
        try:
            return self.nc_node
        except NcNodeMapperError:
            return None

    @property
    def file_loc(self):
        # here we return the _source if it is passed to the constructor
        # we had ensured in open that if self._source is given,
        # the source passed to open can only be None.
        # so that source is always the same as self.nc_node.file_loc
        if self._source is not None:
            return self._source
        # if no dataset is open, we just return None
        if self._io_obj is None:
            return None
        # the opened dataset file loc.
        return self.node_mappers["ncopen"].file_loc

    def open(self, source=None):
        """Return a context to operate on `source`.

        Parameters
        ----------
        source : str, `pathlib.Path`, `FileLoc`, `netCDF4.Dataset`, optional
            The data file location or netCDF dataset. If None, the
            source passed to constructor is used. Noe that source has to
            be None if it has been specified in the constructor.
        """
        source = self._resolve_source_arg(source, remote_ok=False)
        # we just use one of the node_mapper to open the dataset, and
        # set the rest via set_nc_node
        # the node_mapper.open will handle the different types of source
        if self.io_obj is not None:
            # the file is already open
            return self
        nc_node = None

        def _open_sub_node(n):
            nonlocal nc_node
            for _, v in n.items():
                if isinstance(v, NcNodeMapper):
                    if nc_node is None:
                        v.open(source)
                        nc_node = v.nc_node
                    else:
                        v.set_nc_node(nc_node)
                    # push to the exit stack
                    self.enter_context(v)
                else:
                    _open_sub_node(v)

        _open_sub_node(self.node_mappers)
        # trigger loading the meta on open if requested
        if self._load_meta_on_open:
            _ = self.meta  # noqa: F841
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}({self.filepath})"

    # we use this to cache the meta to avoid query the same info multiple times.
    _meta_cached: dict

    def _reset_instance_state(self):
        """Reset the instance state."""
        self._meta_cached.clear()

    # maps of the data record for various types
    # the first level maps to the method of this class,
    # the second level maps to the data object class,
    # the inner most level maps to the netcdf dataset.
    _node_mappers: dict
    _node_maps = {
        # this is a dummy mapper to hold the actual opened dataset.
        # this is not used for reading the dataset.
        "ncopen": {},
        # this is mapper to use when data_kind is called
        "identify": {
            KidsDataKind.ReducedKidsData: {
                "kind_str": "Header.Kids.kind",
                "obs_type": "Header.Toltec.ObsType",
            },
            KidsDataKind.RawKidsData: {
                "kind_str": "Header.Kids.kind",
                "obs_type": "Header.Toltec.ObsType",
            },
        },
        # this is mapper to use when meta is called
        "meta": {
            KidsDataKind.KidsData: {
                "fsmp": "Header.Toltec.SampleFreq",
                "flo_center": "Header.Toltec.LoCenterFreq",
                "atten_drive": "Header.Toltec.DriveAtten",
                "atten_sense": "Header.Toltec.SenseAtten",
                "roachid": "Header.Toltec.RoachIndex",
                "obsnum": "Header.Toltec.ObsNum",
                "subobsnum": "Header.Toltec.SubObsNum",
                "scannum": "Header.Toltec.ScanNum",
                # assoc
                "cal_roachid": "Header.Toltec.RoachIndex",
                "cal_obsnum": "Header.Toltec.TargSweepObsNum",
                "cal_subobsnum": "Header.Toltec.TargSweepSubObsNum",
                "cal_scannum": "Header.Toltec.TargSweepScanNum",
            },
            KidsDataKind.RawKidsData: {
                "n_kids_design": "loclen",
                "n_chans_max": "Header.Toltec.MaxNumTones",
                "filename_orig": "Header.Toltec.Filename",
                "mastervar": "Header.Toltec.Master",
                "repeatvar": "Header.Toltec.RepeatLevel",
                # data shape
                "n_times": "time",
                "n_chans": "iqlen",
            },
            KidsDataKind.RawSweep: {
                "n_sweepreps": "Header.Toltec.NumSamplesPerSweepStep",
                "n_sweepsteps": "Header.Toltec.NumSweepSteps",
                "n_sweeps_max": "numSweeps",
            },
            KidsDataKind.ReducedKidsData: {
                "n_chans": "n_chans",
            },
            KidsDataKind.ReducedSweep: {
                "n_sweepsteps": "nsweeps",
            },
            KidsDataKind.SolvedTimeStream: {
                "n_times": "ntimes",
            },
        },
        "axis_data": {
            "f_tones": "Header.Toltec.ToneFreq",
            "flos": "Data.Toltec.LoFreq",
        },
        "data": {
            KidsDataKind.RawKidsData: {
                "I": "Data.Toltec.Is",
                "Q": "Data.Toltec.Qs",
            },
            KidsDataKind.ReducedSweep: {
                "I": "Data.Kids.Is",
                "Q": "Data.Kids.Qs",
            },
            KidsDataKind.SolvedTimeStream: {
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

    @staticmethod
    def _create_node_mappers(node_maps) -> dict:
        """Create node mappers for the inner most level of dict of node_maps."""

        def _get_sub_node(n):
            if all(not isinstance(v, dict) for v in n.values()):
                return NcNodeMapper(nc_node_map=n)
            return {k: _get_sub_node(v) for k, v in n.items()}

        return _get_sub_node(node_maps)  # type: ignore

    _data_kind_identifiers = {}
    """A registry to store all data kind identifiers."""

    @classmethod
    @add_to_dict(_data_kind_identifiers, KidsDataKind.RawKidsData)
    def _identify_raw_kids_data(cls, node_mapper):
        # this returns a tuple of (valid, meta)
        nm = node_mapper
        if not nm.hasvar("obs_type") or nm.hasvar("kind_str"):
            return False, dict()

        obs_type = nm.getscalar("obs_type")
        logger.debug(f"found obs_type -> {nm['obs_type']}: {obs_type}")

        data_kind = {
            0: KidsDataKind.RawTimeStream,
            1: KidsDataKind.RawTimeStream,
            2: KidsDataKind.VnaSweep,
            3: KidsDataKind.TargetSweep,
            4: KidsDataKind.Tune,
        }.get(obs_type, KidsDataKind.RawUnknown)
        # here we allow identification of unknown kids data
        # but in this case only minimal meta data will be available
        return True, {
            "obs_type": obs_type,
            "data_kind": data_kind,
        }

    @classmethod
    @add_to_dict(_data_kind_identifiers, KidsDataKind.ReducedKidsData)
    def _identify_reduced_kids_data(cls, node_mapper):
        nm = node_mapper
        if not nm.hasvar("kind_str") and not nm.hasvar("kind_str_deprecated"):
            return False, dict()

        if nm.hasvar("kind_str"):
            kind_str = nm.getstr("kind_str")
        else:
            kind_str = "unknown"
        logger.debug(f"found kind_str={kind_str} from {nm['kind_str']}")

        data_kind = {
            "d21": KidsDataKind.D21,
            "processed_sweep": KidsDataKind.ReducedSweep,
            "processed_timestream": KidsDataKind.SolvedTimeStream,
            "SolvedTimeStream": KidsDataKind.SolvedTimeStream,
        }.get(kind_str, KidsDataKind.ReducedUnknown)
        return True, {"kind_str": kind_str, "data_kind": data_kind}

    @cached_property
    def data_kind(self):
        """The data kind."""
        # The below is called once and only once as the first step to read the
        # netCDF dataset.
        # here we reset various cache in order to have a clean start
        self._reset_instance_state()
        for k, m in self.node_mappers["identify"].items():
            valid, meta = self._data_kind_identifiers[k](self.__class__, m)
            logger.debug(f"check data kind as {k}: {valid} {meta}")
            if valid:
                self._meta_cached.update(meta)
                return meta["data_kind"]
        else:
            # none of the identify mappers work
            raise DataFileIOError("invalid file or data format.")

    @cached_property
    def meta(self):
        data_kind = self.data_kind
        # load data kind specific meta
        _meta = self._meta_cached
        for k, m in self.node_mappers["meta"].items():
            if k & data_kind:
                # read all entries in mapper
                for kk in m.nc_node_map.keys():
                    _meta[kk] = m.getany(kk)

        # update meta using the per type registered updater.
        for k, _meta_updater in self._meta_updaters.items():
            if data_kind & k:
                _meta_updater(self)
        return _meta

    # a registry to metadata updaters
    _meta_updaters = {}

    @add_to_dict(_meta_updaters, KidsDataKind.KidsData)
    def _update_derived_info(self):
        meta = self._meta_cached
        meta["file_loc"] = self.file_loc
        meta["instru"] = "toltec"
        meta["interface"] = f'toltec{meta["roachid"]}'
        # TODO someday we may need to change the mapping between this
        meta["master"] = meta.get("mastervar", 1)
        meta["repeat"] = meta.get("repeatvar", 1)
        meta["nw"] = meta["roachid"]

    @add_to_dict(_meta_updaters, KidsDataKind.ReducedSweep)
    def _update_reduced_sweep_block_info(self):
        meta = self._meta_cached
        result = dict()
        result["n_sweeps"] = 1
        result["sweep_slices"] = [slice(0, meta["n_sweepsteps"])]
        result["n_blocks_max"] = 1
        result["n_timesperblock"] = meta["n_sweepsteps"]
        result["n_blocks"] = result["n_sweeps"]
        result["block_slices"] = result["sweep_slices"]
        meta.update(result)

    @add_to_dict(_meta_updaters, KidsDataKind.RawSweep)
    def _update_raw_sweep_block_info(self):
        # This function is to populate raw sweep block info to meta
        # The block info is a set of meta data that indicate
        # logical unit of the dataset.
        # For sweeps, each block is one monotonic frequency sweep;
        # for timestreams, this could be arbitrary, and is default
        # to one block as of 20200722.
        meta = self._meta_cached

        data_kind = meta["data_kind"]

        result = dict()

        # this is the theoretical number of samples per sweep block.
        result["n_timespersweep"] = meta["n_sweepsteps"] * meta["n_sweepreps"]
        # we need to load flos to properly handle the sweeps with potentially
        # missing samples
        nm = self.node_mappers["axis_data"]
        flos_Hz = nm.getvar("flos")[:]
        result["flos"] = flos_Hz << u.Hz

        # for tune file, we expect multiple sweep blocks
        # because there could be samples missing, the only
        # reliable way to identify the blocks is to
        # check the flos array looking for monotonically
        # increasing blocks
        # sometimes there could be corrupted data and the flo is set to 0.
        # this assert detects such case because otherwise the break
        # indices will be incorrect.
        if not np.all(flos_Hz > 0):
            raise ValueError("invalid flo found in data file.")
        # this gives the index of the first item after a decrease in flo
        break_indices = np.where(np.diff(flos_Hz) < 0)[0] + 1

        if break_indices.size == 0:
            # we have one (maybe incompleted) block
            result["n_sweeps"] = 1
            result["sweep_slices"] = [slice(0, meta["n_times"])]
        else:
            # this can only happend to tune files
            assert data_kind == KidsDataKind.Tune
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
                    f"missing data in sweep blocks {incompleted_sweeps} "
                    f"sizes={sweep_sizes[incompleted_sweeps]}."
                )

        # populate block meta
        # this is the maximum number of blocks
        result["n_blocks_max"] = meta["n_sweeps_max"]
        result["n_timesperblock"] = result["n_timespersweep"]
        result["n_blocks"] = result["n_sweeps"]
        result["block_slices"] = result["sweep_slices"]
        meta.update(result)

    @add_to_dict(_meta_updaters, KidsDataKind.ReducedKidsData | KidsDataKind.TimeStream)
    def _update_raw_timestream_block_info(self):
        result = dict()
        result["n_blocks_max"] = 1
        result["n_blocks"] = 1
        self._meta_cached.update(result)

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
            iblock = range(0, n_blocks_max)[block_index]
        logger.debug(
            f"resolve block_index {block_index} -> iblock={iblock} "
            f"n_blocks={n_blocks} (n_blocks_max={meta['n_blocks_max']})"
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
        v = nm.getvar("chans")
        # for multi block data, v should be of 2-dimensional
        # and for single block data, v could be of 2-dimensional or one
        if n_blocks == 1 and len(v.shape) == 1:
            data = v[:].reshape((n_blocks, n_chans))
        else:
            data = v[:n_blocks, :]
        data = data << u.Hz
        assert data.shape == (n_blocks, n_chans)

        result = []
        for i in range(n_blocks):
            chan_axis_data = QTable()
            chan_axis_data["id"] = Column(
                range(n_chans), description="The channel index"
            )
            chan_axis_data["f_tone"] = Column(
                data[i], description="The ROACH tone frequency as used in the FFT comb"
            )
            chan_axis_data["f_chan"] = Column(
                data[i] + (meta["flo_center"] << u.Hz),
                description="The channel reference frequency.",
            )
            chan_axis_data.meta.update(meta)  #  type: ignore
            chan_axis_data.meta["block_index"] = i  # type: ignore
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
            nc_node.variables["Header.Toltec.ModelParamsHeader"]
        )
        result = []
        for i in range(n_blocks):
            # note the first dim is to select the block.
            model_params_data = nc_node.variables["Header.Toltec.ModelParams"][i, :, :]
            model_params_table = QTable(
                data=list(model_params_data), names=model_params_header
            )
            result.append(model_params_table)
        return result

    def get_model_params_table(self, block_index=None):
        """Return the model params table at `block_index`."""
        iblock, _, _ = self._resolve_block_index(block_index)
        return self._model_params_tables[iblock]

    @staticmethod
    def _populate_sweep_axis_data(
        sweep_axis_data,
        id,
        f_sweeps,
        n_samples,
        sample_start,
        sample_end,
        meta,
        block_index,
    ):
        sweep_axis_data["id"] = Column(id, description="The sweep step index")
        sweep_axis_data["f_sweep"] = Column(
            f_sweeps, description="The sweep step frequency"
        )
        sweep_axis_data["n_samples"] = Column(
            n_samples, description="The number of repeats in a sweep step"
        )
        sweep_axis_data["sample_start"] = Column(
            sample_start, description="The sample index at the start of sweep step."
        )
        sweep_axis_data["sample_end"] = Column(
            sample_end, description="The sample index after the end of sweep step."
        )
        sweep_axis_data.meta.update(meta)  # type: ignore
        sweep_axis_data.meta["block_index"] = block_index  # type: ignore

    def _get_raw_sweep_sweep_axis_data(self):
        """Return the sweep steps.

        This is a list of tables.
        """
        meta = self.meta
        flos_Hz = meta["flos"].to_value(u.Hz)
        result = []
        for iblock, block_slice in enumerate(meta["block_slices"]):
            sweep_axis_data = QTable()
            # the raw sweep axis is constructed from checking the
            # flo frequencies
            uflos, uiflos, urflos = np.unique(
                flos_Hz[block_slice], return_index=True, return_counts=True
            )
            # pad the index with block slice start
            uiflos = uiflos + block_slice.start
            self._populate_sweep_axis_data(
                sweep_axis_data,
                id=range(len(uflos)),
                f_sweeps=uflos << u.Hz,
                n_samples=urflos,
                sample_start=uiflos,
                sample_end=uiflos + urflos,
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

        f_sweeps = nm.getvar("sweeps")[:] + meta["flo_center"]

        sweep_axis_data = QTable()
        sweep_id = np.arange(len(f_sweeps))
        self._populate_sweep_axis_data(
            sweep_axis_data,
            id=sweep_id,
            f_sweeps=f_sweeps << u.Hz,
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
        if KidsDataAxis.Sweep not in self.axis_types:  # type: ignore[attr-defined]
            return None
        if self.data_kind & KidsDataKind.RawSweep:
            return self._get_raw_sweep_sweep_axis_data()
        return self._get_reduced_sweep_sweep_axis_data()

    def get_sweep_axis_data(self, block_index=None):
        """Return the tones at `block_index`."""

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
        slicer = self.block_loc(None)  # type: ignore[attr-defined]
        for t, arg in slicer_args.items():
            slicer = getattr(slicer, f"{t}_loc")(arg)
        return self._read_sliced(slicer)

    def _resolve_slice(self, slicer):
        """Read the file for data specified by the `slicer`."""

        result = dict()
        # parse the slicer args and get the data.
        result["ops"] = ops = dict()
        for t in self.axis_types:  # type: ignore[attr-defined]
            ops[t] = slicer.get_slice_op(t)

        # do some check
        if (
            sum(
                [
                    ops.get(KidsDataAxis.Sample, None) is not None,
                    ops.get(KidsDataAxis.Sweep, None) is not None,
                ]
            )
            == 2
        ):
            raise ValueError("can only slice on one of sample or sweep.")

        if (
            sum(
                [ops.get("sample", None) is not None, ops.get("time", None) is not None]
            )
            == 2
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
                    df = tbl.to_pandas()
                    op = df.eval(op).to_numpy(dtype=bool)
            tbl = tbl[op]
            tbl.meta["_slice_op"] = op
            return tbl, op

        block_index = result["block_index"] = ops["block"]
        chan_axis_data, chan_op = slice_table(
            self.get_chan_axis_data(block_index=block_index), ops[KidsDataAxis.Chan]
        )
        # logger.debug(f"sliced chan axis data:\n{chan_axis_data}")
        logger.debug(
            f"sliced {len(chan_axis_data)} " f"chans out of {self.meta['n_chans']}"
        )

        result["chan_axis_data"] = chan_axis_data
        # this will be used to load the data
        result["chan_slice"] = chan_op

        if KidsDataAxis.Sweep in self.axis_types:  # type: ignore[attr-defined]
            # in this case,
            # we re-build the sample slice from the range of the sliced
            # sweep table
            sweep_axis_data, _ = slice_table(
                self.get_sweep_axis_data(block_index=block_index),
                ops[KidsDataAxis.Sweep],
            )
            # logger.debug(f"sliced sweep axis data:\n{sweep_axis_data}")
            logger.debug(
                f"sliced {len(sweep_axis_data)} "
                f"sweep steps out of {self.meta['n_sweepsteps']}"
            )
            # this will be used to load the data
            # this data will be reduced for each sweep step later
            # TODO this is less optimal for sweep_op being a mask.
            # but for now the sweep is small and we can afford doing so
            sample_slice = slice(
                np.min(sweep_axis_data["sample_start"]),
                np.max(sweep_axis_data["sample_end"]),
            )
            result["sweep_axis_data"] = sweep_axis_data
        elif KidsDataAxis.Time in self.axis_types:  # type: ignore[attr-defined]
            # in this case,
            # we build the sample slice from the range of time,
            fsmp_Hz = self.meta["fsmp"]

            def _t_to_sample(t):
                if t is None:
                    return None
                if not isinstance(t, u.Quantity):
                    t = t << u.s
                if t.unit.is_equivalent(u.s):  # type: ignore[attr-defined]
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
                    )
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

    def _read_sliced(self, slicer):
        """Read the file for data specified by the `slicer`."""
        s = self._resolve_slice(slicer)
        # now that we have the chan slice and sample slice
        # we can read the data
        data_kind = self.data_kind
        # we create a copy of meta data here to store the slicer info
        meta = self.meta.copy()
        meta.update(s)
        data = dict()
        for k, m in self.node_mappers["data"].items():
            if data_kind & k:
                logger.debug(
                    f"read data {m.nc_node_map.keys()} sample_slice="
                    f"{pformat_fancy_index(s['sample_slice'])}"
                    f" chan_slice="
                    f"{pformat_fancy_index(s['chan_slice'])}"
                )
                for key in m.nc_node_map.keys():
                    # we arrange the data so that the data
                    # chan axis is first, as required by the
                    # kidsproc.kidsdata containers
                    data[key] = m.getvar(key)[
                        s["sample_slice"],
                        s["chan_slice"],
                    ].T

        # for reduced sweep it may have d21 data
        if data_kind & KidsDataKind.ReducedSweep:
            m = self.node_mappers["data_extra"]["d21"]
            logger.debug(f"read extra data {m.nc_node_map.keys()}")
            for key in m.nc_node_map.keys():
                if m.hasvar(key):
                    data[key] = m.getvar(key)[:]
        # for vna sweep we can load the candidates list
        if data_kind & KidsDataKind.ReducedVnaSweep:
            m = self.node_mappers["data_extra"]["candidates"]
            logger.debug(f"read extra data {m.nc_node_map.keys()}")
            for key in m.nc_node_map.keys():
                if m.hasvar(key):
                    data[key] = m.getvar(key)[:]

        # for the solved timestreams we also load the psd info if they
        # are available
        if data_kind & KidsDataKind.SolvedTimeStream:
            m = self.node_mappers["data_extra"]["psd"]
            logger.debug(
                f"read extra data {m.nc_node_map.keys()}"
                f" chan_slice="
                f"{pformat_fancy_index(s['chan_slice'])}"
            )
            for key in m.nc_node_map.keys():
                # we arrange the data so that the data
                # chan axis is first, as required by the
                # kidsproc.kidsdata containers
                if m.hasvar(key):
                    v = m.getvar(key)
                    if len(v.shape) == 1:
                        # the psd_fs is vector so skip the slice
                        v = v[:]
                    else:
                        v = v[:, s["chan_slice"]].T
                    data[key] = v
        # for the raw sweeps we do the reduction for each step
        if data_kind & KidsDataKind.RawSweep:
            sweep_axis_data = s["sweep_axis_data"]
            b0 = s["sample_slice"].start  # this is the ref index
            for k in ("I", "Q"):
                a = np.full(
                    (len(s["chan_axis_data"]), len(sweep_axis_data)), np.nan, dtype="d"
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
        else:
            # generic maker, this is just to return the things as a dict
            return {"data_kind": data_kind, "meta": meta, "data": data}

    _kidsdata_obj_makers = {}
    """A registry to store the data obj makers."""

    @classmethod
    @add_to_dict(_kidsdata_obj_makers, KidsDataKind.Sweep)
    def _make_kidsdata_sweep(cls, data_kind, meta, data):
        f_chans = meta["chan_axis_data"]["f_chan"]
        f_sweeps = meta["sweep_axis_data"]["f_sweep"]
        # we need to pack the I and Q
        S21 = make_complex(data["I"], data["Q"])
        if "unc_I" in data:
            unc_S21 = StdDevUncertainty(make_complex(data["unc_I"], data["unc_Q"]))
        else:
            unc_S21 = None
        result = MultiSweep(
            meta=meta,
            f_chans=f_chans,
            f_sweeps=f_sweeps,
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
                )
            )
        return result

    @classmethod
    @add_to_dict(_kidsdata_obj_makers, KidsDataKind.RawTimeStream)
    def _make_kidsdata_rts(cls, data_kind, meta, data):
        f_chans = meta["chan_axis_data"]["f_chan"]
        return TimeStream(
            meta=meta,
            f_chans=f_chans,
            I=data["I"],  # noqa: E741
            Q=data["Q"],
        )

    @classmethod
    @add_to_dict(_kidsdata_obj_makers, KidsDataKind.SolvedTimeStream)
    def _make_kidsdata_sts(cls, data_kind, meta, data):
        f_chans = meta["chan_axis_data"]["f_chan"]
        # add the psd data to the meta
        meta = meta.copy()
        for k, v in data.items():
            if k.endswith("_psd"):
                meta[k] = v
        return TimeStream(
            meta=meta,
            f_chans=f_chans,
            r=data["r"],
            x=data["x"],
        )

    def __getstate__(self):
        # need to reset the object before pickling
        if self._auto_close_on_pickle and self._source is not None:
            is_open = self.io_obj is not None
            self.close()
            return {
                "_auto_close_on_pickle_wrapped": self.__dict__,
                "_auto_close_on_pickle_is_open": is_open,
            }
        return self.__dict__

    def __setstate__(self, state):
        if "_auto_close_on_pickle_wrapped" in state:
            state, is_open = (
                state["_auto_close_on_pickle_wrapped"],
                state["_auto_close_on_pickle_is_open"],
            )
            self.__dict__.update(state)
            if is_open:
                # try open the object
                self.open()
            return
        self.__dict__.update(state)