#! /usr/bin/env python

from cached_property import cached_property

import numpy as np
from kidsproc.kidsdata import (
        Sweep,
        RawTimeStream, SolvedTimeStream, VnaSweep, TargetSweep)
from tollan.utils.nc import ncopen, ncinfo, NcNodeMapper
from tollan.utils.log import get_logger
from tollan.utils.slice_chain import BoundedSliceChain
from tollan.utils.numpy_dance import flex_reshape
from .registry import register_io_class
from pathlib import Path
from contextlib import ExitStack
from ..fs.toltec import ToltecDataFileSpec
import re
import astropy.units as u
from astropy.table import Table, Column
from astropy.nddata import StdDevUncertainty
from kidsproc import kidsmodel
import numexpr as ne


__all__ = ['NcFileIO']


UNKNOWN_KIND = "UnknownKind"


def identify_toltec_nc(filepath):
    filepath = Path(filepath)
    pattern = r'^toltec.*\.nc$'
    return re.match(pattern, filepath.name) is not None


@register_io_class("nc.toltec.kidsdata", identifier=identify_toltec_nc)
class NcFileIO(ExitStack):
    """This class provides methods to access data in netCDF files."""

    spec = ToltecDataFileSpec
    logger = get_logger()

    def __init__(self, source):
        super().__init__()
        self._open_nc(source)
        # setup key mappers
        # items with __ is no longer valid for new files
        self.nm = NcNodeMapper(self.nc, {
                # data
                "is": "Data.Toltec.Is",
                "qs": "Data.Toltec.Qs",
                "flos": "Data.Toltec.LoFreq",
                "__sweeps": "Data.Toltec.SweepFreq",
                "tones": "Header.Toltec.ToneFreq",
                "rs": "Data.Kids.rs",
                "xs": "Data.Kids.xs",
                # tone axis
                "__tonemodelparams": "Header.Toltec.ModelParams",
                "__tonemodelparamsheader": "Header.Toltec.ModelParamsHeader",
                # meta
                "kindvar": "Header.Toltec.ObsType",
                "n_tones_design": "loclen",
                "n_tones_max": "Header.Toltec.MaxNumTones",
                "fsmp": "Header.Toltec.SampleFreq",
                "atten_in": "Header.Toltec.InputAtten",
                "atten_out": "Header.Toltec.OutputAtten",
                "source_orig": "Header.Toltec.Filename",
                "mastervar": "Header.Toltec.Master",
                "roachid": "Header.Toltec.RoachIndex",
                "obsid": "Header.Toltec.ObsNum",
                "subobsid": "Header.Toltec.SubObsNum",
                "scanid": "Header.Toltec.ScanNum",
                "__flo": "Header.Toltec.LoFreq",
                "__flo_offset": "Header.Toltec.LoOffset",
                # assoc
                "cal_roachid": "Header.Toltec.RoachIndex",
                "cal_obsid": "Header.Toltec.TargSweepObsNum",
                "cal_subobsid": "Header.Toltec.TargSweepSubObsNum",
                "cal_scanid": "Header.Toltec.TargSweepScanNum",
                # data shape
                "n_times": "time",
                "n_tones": "iqlen",
                "__n_tones": "toneFreqLen",
                "n_sweepreps": "Header.Toltec.NumSamplesPerSweepStep",
                "n_sweepsteps": "Header.Toltec.NumSweepSteps",
                "n_sweeps_max": "numSweeps",
                "n_kidsmodelparams": "modelParamsNum",
                })
        self.reset_selections()

    def __repr__(self):
        r = f"{self.__class__.__name__}({self.filepath})"
        try:
            # check if the nc file is still open
            self.nc.__repr__()
        except RuntimeError:
            return f'{r} (file closed)'
        else:
            return r

    def _open_nc(self, source):
        nc, _close = ncopen(source)
        self.push(_close)
        self.logger.debug("ncinfo: {}".format(ncinfo(nc)))
        self.nc = nc
        self.filepath = Path(nc.filepath())

    def open(self):
        self._open_nc(self.filepath)

    @cached_property
    def kind_cls(self):
        m = self.nm
        kind_cls = None
        # check header info
        if m.hasvar('kindvar'):
            kindvar = m.getscalar('kindvar')
            self.logger.debug(f"found kindvar={kindvar} from {m['kindvar']}")

            kind_cls = {
                    1: RawTimeStream,
                    2: VnaSweep,
                    3: TargetSweep,
                    4: TargetSweep  # tone file
                    }.get(kindvar, None)
            if kind_cls is None:
                self.logger.warn(f"kindvar={kindvar} unrecognized")
        self.logger.debug(f"check kind_cls hint={kind_cls}")
        # check data entries
        if not m.hasvar('is', 'qs') and m.hasvar("rs", "xs"):
            if kind_cls != SolvedTimeStream:
                kind_cls = SolvedTimeStream
                self.logger.debug(f"updated kind_cls={kind_cls}")
        self.logger.debug(f"found kind_cls={kind_cls}")
        return kind_cls

    @cached_property
    def kind(self):
        cls = self.kind_cls
        return UNKNOWN_KIND if cls is None else cls.__name__

    @cached_property
    def is_sweep(self):
        return issubclass(self.kind_cls, Sweep)

    def sync(self):
        self.nc.sync()
        for k in ('n_times', ):
            old = self.meta[k]
            new = self.nm.getdim(k)
            if old != new:
                self.logger.info(f"updated {k} {old}->{new} via sync")
            self.meta[k] = new

    def _get_sweep_meta(self, meta):
        nm = self.nm
        result = dict()
        for k in (
                "n_sweepreps", "n_sweepsteps",
                "n_sweeps_max", "n_kidsmodelparams"):
            try:
                result[k] = nm.get(k)
            except Exception:
                self.logger.error(
                        f"missing item in data {k}", exc_info=True)
                continue
        result['n_timespersweep'] = result["n_sweepsteps"] * \
            result["n_sweepreps"]
        result["n_sweeps"] = meta["n_times"] // result['n_timespersweep']
        # populate block meta
        result['n_blocks_max'] = result['n_sweeps_max']
        result['n_blocks'] = result['n_sweeps']
        result['n_timesperblock'] = result['n_timespersweep']
        result['block_shape'] = (-1, result['n_sweepreps'])
        return result

    @cached_property
    def meta(self):
        nm = self.nm

        def logged_update_dict(l, r):
            for k, v in r.items():
                if k in l and l[k] != v:
                    self.logger.error(
                            f"entry changed"
                            f" {k} {l[k]} -> {v}")
                l[k] = v

        result = dict()
        # all kind
        for k in (
                "kindvar", "n_tones_design", "n_tones_max", "fsmp",
                "atten_in", "atten_out", "mastervar", "source_orig",
                "roachid", "obsid", "subobsid", "scanid",
                "cal_roachid", "cal_obsid", "cal_subobsid", "cal_scanid",
                "n_times", "n_tones", "__n_tones", ):
            try:
                result[k] = nm.get(k)
            except Exception:
                self.logger.error(f"missing item in data {k}", exc_info=True)
                continue
        # sweep only
        if self.is_sweep:
            logged_update_dict(result, self._get_sweep_meta(result))

        # handle lofreqs, which are no longer there in new files
        if nm.hasvar("__flo", "__flo_offset"):
            logged_update_dict(result, {
                "flo": nm.getscalar('__flo'),
                "flo_offset": nm.getscalar('__flo_offset'),
            })
        else:
            logged_update_dict(result, {
                'flo': 0.,
                'flo_offset': 0.,
                })

        logged_update_dict(
                result, self.spec.info_from_filename(
                    self.filepath, resolve=True))
        return result

    @staticmethod
    def _slice_table(tbl, slice_):
        if isinstance(slice_, str):
            df = tbl.to_pandas()
            df.query(slice_, inplace=True)
            return Table.from_pandas(df)
        return tbl[slice_]

    def _get_block_index(self, index):
        meta = self.meta
        n_blocks = meta['n_blocks']
        n_blocks_max = meta['n_blocks_max']
        if index is None:
            iblock = n_blocks
        elif index >= n_blocks_max or index < -n_blocks_max:
            raise ValueError(
                    f"invalid block index {index}"
                    f" (n_blocks_max={n_blocks_max})")
        else:
            iblock = (n_blocks_max + index) % n_blocks_max
        if iblock == n_blocks_max:
            iblock -= 1
        self.logger.debug(
            f"get block {iblock} of "
            f"{n_blocks} blocks (n_blocks_max={n_blocks_max})")
        return iblock, n_blocks, n_blocks_max

    def _get_tone_axis_data(self, index=None):
        """Returns the tone table for given block index.

        If index is None, the used index is the **actual** last block,
        and if index is -1, the used index is the last block of all
        pre-allocated blocks.
        """
        nm = self.nm
        if not nm.hasvar("tones"):
            raise RuntimeError("no tone data found")
        # tone param header
        # tone param data
        # n_tones = meta['ntones']
        self.logger.debug("get tones for block_index={block_index}")
        iblock, n_blocks, n_blocks_max = self._get_block_index(index)
        tfs = nm.getvar("tones")[iblock, :]
        tis = range(len(tfs))

        tone_axis_data = Table([
            Column(tis, name='id'),
            Column(tfs, name='fc', unit=u.Hz)
            ])
        tone_axis_data.meta.update({
                'n_blocks_max': n_blocks_max,
                'n_blocks': n_blocks,
                'block_index': iblock,
                })
        return tone_axis_data

    @cached_property
    def tone_axis(self):
        """Returns the last tone table."""
        return self._get_tone_axis_data(index=None)

    @cached_property
    def sample_axis(self):
        meta = self.meta
        n = meta['n_times']
        return BoundedSliceChain(range(0, n), n)

    def _block_to_sample_slice(self, index):
        """Return the sample slice to the block of given index."""
        self.logger.debug("get sample slice for block_index={block_index}")
        meta = self.meta
        iblock, n_blocks, n_blocks_max = self._get_block_index(index)
        s = meta['n_timesperblock']
        return slice(iblock * s, (iblock + 1) * s), meta['block_shape']

    def _sweep_axis_data(self, index):
        if not self.is_sweep:
            raise ValueError("data of {self.kind} does not have sweep axis")
        logger = get_logger()
        nm = self.nm

        # get sample_slice for block index
        sample_slice, block_shape = self._block_to_sample_slice(index)
        sfs = nm.getvar("flos")[sample_slice]
        sfs = flex_reshape(sfs, block_shape)
        # do reduction
        rsfs = []
        rn = []
        r0 = []
        r1 = []
        # do the reduce
        for i in range(sfs.shape[0]):
            # check that fs are the same
            fs = sfs[i, :]
            fs = fs[fs > 0]  # sometimes the value could be 0
            if len(fs) < block_shape[-1]:
                logger.warning(
                    f"fs of sweep step {i} has {block_shape[-1] - len(fs)}"
                    f" missing data: {fs}")
            if len(fs) == 0:
                rsfs.append(np.nan)
                rn.append(0)
                continue
            if len(np.unique(fs)) > 1:
                logger.warning(f"fs of sweep step {i} is not uniform: {fs}")
            rsfs.append(fs[0])
            rn.append(len(fs))
            r0.append(sample_slice.start + i * block_shape[-1])
            r1.append(sample_slice.start + (i + 1) * block_shape[-1])
        irsfs = range(len(rsfs))
        sweep_axis_data = Table([
            Column(irsfs, name='id'),
            Column(rsfs, name='flo', unit=u.Hz),
            Column(rn, name='n_samples'),
            Column(r0, name='sample_start'),
            Column(r1, name='sample_end'),
            ])
        sweep_axis_data.meta.update({
                'sample_slice': self.sample_axis[sample_slice],
                'block_shape': block_shape,
                })
        return sweep_axis_data

    @cached_property
    def sweep_axis(self):
        """Returns the last sweep table."""
        return self._sweep_axis_data(index=None)

    def time_axis(self):
        return self.sample_axis

    class ILoc(object):
        """
        This provide a interface similar to that of pandas DataFrame.xloc.
        """
        def __init__(self, file_obj, axis_name):
            self._fo = file_obj
            self._axis_name = axis_name
            self._pop_kwargs()

        def _pop_kwargs(self):
            kwargs = getattr(self, '_kwargs', None)
            self._kwargs = dict()
            return kwargs

        def _push_kwrags(self, kwargs):
            self._kwargs.update(kwargs)

        def __getitem__(self, *args):
            return getattr(self._fo, f'select_{self._axis_name}')(
                    *args, **self._pop_kwargs())

        def __call__(self, **kwargs):
            self._push_kwrags(kwargs)
            return self

    def _reset_selection(self, axis_name):
        """Reset the select registry for specified axis."""
        setattr(self, f'_{axis_name}_axis_cb', None)

    def reset_selections(self):
        """Reset the selection registry for all axis."""
        for name in ('tone', 'sample', 'sweep', 'time'):
            self._reset_selection(name)
            setattr(self, f'i{name}', self.ILoc(self, name))

    def select_tone(self, *args):
        """Register tone selection callback."""
        if len(args) > 1:
            return self.select_tone(slice(*args))
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, int):
                arg = slice(arg, arg + 1)

            def cb(slice_=arg, cb0=self._tone_axis_cb):
                if cb0 is not None:
                    tbl = cb0()
                else:
                    tbl = self.tone_axis
                return self._slice_table(tbl, slice_)
            self._tone_axis_cb = cb
            return self
        # no args
        return self

    def select_sample(self, *args, rfunc=None):
        """Register callback for selecting samples."""
        for n in ('sweep', 'time'):
            if getattr(self, f'_{n}_axis_cb') is not None:
                raise ValueError("can only select one of {sample,sweep,time}")

        if len(args) > 1:
            return self.select_sample(slice(*args), rfunc=rfunc)
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, int):
                arg = slice(arg, arg + 1)

            def cb(slice_=arg, rfunc=rfunc, cb0=self._sample_axis_cb):
                if cb0 is not None:
                    slice0, rfunc0 = cb0()
                else:
                    slice0 = self.sample_axis

                    def rfunc0(d):
                        return d

                if rfunc is not None:
                    def rfunc1(d, rfunc=rfunc, rfunc0=rfunc0):
                        return rfunc(rfunc0(d))
                else:
                    rfunc1 = rfunc0
                return slice0[slice_], rfunc1
            self._sample_axis_cb = cb
            return self
        # no args
        return self

    def select_sweep(self, *args, index=None):
        """Register callback for selecting sweeps."""

        for n in ('sample', 'time'):
            if getattr(self, f'_{n}_axis_cb') is not None:
                raise ValueError("can only select one of {sample,sweep,time}")

        if len(args) > 1:
            return self.select_sweep(slice(*args), index=index)

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, int):
                arg = slice(arg, arg + 1)

            def cb(slice_=arg, index=index, cb0=self._sweep_axis_cb):
                if cb0 is not None:
                    tbl0, slice0, rfunc0 = cb0()
                    tbl1 = self._slice_table(tbl0, slice_)
                    rslice = tbl1['id'].tolist()

                    # this function slice on existing reduced data object.
                    def rfunc1(d, rslice=rslice, rfunc0=rfunc0):
                        return {
                                k: v[:, rslice]
                                for k, v in rfunc0(d).items()
                                }
                else:
                    tbl0 = self._sweep_axis_data(index)
                    tbl1 = self._slice_table(tbl0, slice_)
                    rslice = tbl1['id'].tolist()

                    # this function slice on the raw data object after slice
                    def rfunc1(
                            d,
                            rshape=tbl1.meta['block_shape'],
                            rslice=rslice,
                            ):
                        a = d['data']
                        mf = np.mean
                        sf = np.std
                        rs = (a.shape[0], ) + rshape
                        ma = mf(a.reshape(rs), axis=-1)[:, rslice]
                        sa = sf(np.abs(a).reshape(rs), axis=-1)[:, rslice]
                        sa = StdDevUncertainty(sa)
                        return dict(d, **{'data': ma, 'uncertainty': sa})
                return tbl1, rfunc1
            self._sweep_axis_cb = cb
            return self
        # no args
        return self.select_sweep(slice(None), index=None)

    def read(self):
        """Return KIDs data instance from reading the file.

        The selection callbacks are evaluated here.
        """
        logger = get_logger()
        logger.debug(f'read {self.kind_cls.__name__} data')

        meta = self.meta
        kwargs = {'meta': meta}

        tone_axis_data = self._tone_axis_cb()
        kwargs['meta']['tones'] = tone_axis_data

        logger.debug(f"selected tone axis:\n{tone_axis_data}")

        tone_slice = tone_axis_data['id'].tolist()

        if self._sample_axis_cb is not None:
            sample_slice, rfunc = self._sample_axis_cb()
        elif self._sweep_axis_cb is not None:
            sweep_axis_data, rfunc = self._sweep_axis_cb()
            sample_slice = sweep_axis_data.meta['sample_slice']
            kwargs['meta']['sweeps'] = sweep_axis_data
            logger.debug(f"selected sweep axis:\n{sweep_axis_data}")
        else:
            raise NotImplementedError
        sample_slice = sample_slice.to_slice()
        logger.debug(
                f"read data with sample_slice={sample_slice}"
                f" tone_slice={tone_slice}")
        # actually read the data
        nm = self.nm
        is_ = nm.getvar('is')[sample_slice, tone_slice].T
        qs = nm.getvar('qs')[sample_slice, tone_slice].T
        iqs = ne.evaluate('I + 1.j * Q', local_dict={'I': is_, 'Q': qs})
        # call the reduce function
        kwargs.update(rfunc({'data': iqs}))
        return self.kind_cls(**kwargs)

    def read0(
            self,
            tone_slice=None,
            sample_slice=None,
            time_slice=None,
            sweep_slice=None,
            ):
        """Return kids data instance from reading the file.

        Parameters
        ----------
        tone_slice: slice, str, or None
            The tones to use. Effective for all kinds.

        sample_slice: slice, str, or None
            The sample index to use. Effective for all kinds.

        time_slice: slice, str, or None
            The times to use. Effective for all kinds.

        sweep_slice: slice, str, or None
            The sweep steps to use. Effective for sweeps.
        """
        logger = get_logger()
        logger.debug(f'read {self.kind_cls.__name__} data')

        if tone_slice is None:
            row_slice = slice(None)
            logger.debug(
                f"read all {len(self.tone_axis)} tones")
        else:
            tones_axis = self.slice_tone_axis(tone_slice)
            row_slice = tones_axis['id'].tolist()
            logger.debug(
                f"slice {len(row_slice)} out of {len(self.tone_axis)} tones")

        col_slices = [sample_slice, time_slice, sweep_slice]
        n_col_slices = sum(0 if s is None else 1 for s in col_slices)
        if n_col_slices > 1:
            raise ValueError(
                    "can only have one of {sample,time,sweep}_slice specified")
        elif n_col_slices == 0:
            col_slice = self.sample_axis.to_slice()
            if self.is_sweep:
                logger.debug(
                    f"read all {len(self.sweep_axis)} sweep steps")
            else:
                logger.debug(
                    f"read all {len(self.sample_axis)} samples")
        elif sweep_slice is not None:
            col_slice = self.slice_sweep_axis(sweep_slice)
            logger.debug(
                    f"slice {len(col_slice)} out of "
                    f"{len(self.sweep_axis)} sweep steps")
            logger.debug(f"sweep axis:\n{col_slice}")
            col_slice, n_reps = self._sweep_to_sample_slice

            def reduce_func(arr):
                rfunc = np.mean
                sfunc = np.std
                rshape = (arr.shape[0], -1, n_reps)
                rarr = rfunc(arr.reshape(rshape), axis=-1)
                sarr = sfunc(np.abs(arr).reshape(rshape), axis=-1)
                sarr = StdDevUncertainty(sarr)
                return {'data': rarr, 'uncertainty': sarr}
        elif sample_slice is not None:
            col_slice = self.slice_sample_axis(sample_slice)
            logger.debug(
                    f"slice {len(col_slice)} out of "
                    f"{len(self.sample_axis)} samples")
            logger.debug(f"slice object {col_slice}")
            col_slice = col_slice.to_slice()
            reduce_func = None
        else:
            raise NotImplementedError

        # actually read the data
        nm = self.nm
        meta = self.meta
        is_ = nm.getvar('is')[col_slice, row_slice].T
        qs = nm.getvar('qs')[col_slice, row_slice].T
        iqs = ne.evaluate('I + 1.j * Q', local_dict={'I': is_, 'Q': qs})
        kwargs = {'data': iqs, 'meta': meta}
        if reduce_func is not None:
            kwargs.update(reduce_func(kwargs['data']))
        return self.kind_cls(**kwargs)


def identify_toltec_model_params(filepath):
    filepath = Path(filepath)
    pattern = r'^toltec.*\.txt$'
    return re.match(pattern, filepath.name) is not None


@register_io_class(
        "txt.toltec.model_params", identifier=identify_toltec_model_params)
class KidsModelParams(object):
    """This class provides methods to access model parameter files."""

    spec = ToltecDataFileSpec

    logger = get_logger()

    def __init__(self, source):
        self.filepath = Path(source)
        self._table = Table.read(
                self.filepath, format='ascii.commented_header')
        self.model_cls = self._get_model_cls(self._table)
        self.model = self._get_model(self.model_cls, self._table)
        self.meta = {
                'source': source,
                'model_cls': self.model_cls.__name__,
                'n_models': len(self.model),
                }

    @staticmethod
    def _get_model_cls(tbl):
        dispatch = {
                kidsmodel.KidsSweepGainWithLinTrend: {
                    'columns': [
                        'fp', 'Qr',
                        'Qc', 'fr', 'A',
                        'normI', 'normQ', 'slopeI', 'slopeQ',
                        'interceptI', 'interceptQ'
                        ],
                    }
                }
        for cls, v in dispatch.items():
            # check column names
            cols_required = set(v['columns'])
            if cols_required.issubset(set(tbl.colnames)):
                return cls
        raise NotImplementedError

    @staticmethod
    def _get_model(model_cls, tbl):
        if model_cls is kidsmodel.KidsSweepGainWithLinTrend:
            # print(model_cls.model_params)
            # print(model_cls)
            dispatch = [
                    ('fr', 'fr', u.Hz),
                    ('Qr', 'Qr', None),
                    ('g0', 'normI', None),
                    ('g1', 'normQ', None),
                    ('g', 'normI', None),
                    ('phi_g', 'normQ', None),
                    ('f0', 'fp', u.Hz),
                    ('k0', 'slopeI', None),
                    ('k1', 'slopeQ', None),
                    ('m0', 'interceptI', None),
                    ('m1', 'interceptQ', None),
                    ]
            args = []
            for k, kk, unit in dispatch:
                if kk is None:
                    args.append(None)
                elif unit is None:
                    args.append(np.asanyarray(tbl[kk]))
                else:
                    args.append(np.asanyarray(tbl[kk]) * unit)
            kwargs = dict(
                    n_models=len(tbl)
                    )
            return model_cls(*args, **kwargs)
        raise NotImplementedError

    def __repr__(self):
        return f"{self.model_cls.__name__}({self.filepath})"
