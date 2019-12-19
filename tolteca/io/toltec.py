#! /usr/bin/env python

from cached_property import cached_property

from ..kidsdata import (
        Sweep,
        RawTimeStream, SolvedTimeStream, VnaSweep, TargetSweep)
from ..utils.nc import ncopen, ncinfo, NcNodeMapper
from ..utils.log import get_logger
from .registry import register_io_class
from pathlib import Path
from contextlib import ExitStack
from ..fs.toltec import ToltecDataFileSpec
import re


__all__ = ['NcFileIO']


UNKNOWN_KIND = "UnknownKind"


def identify_toltec_nc(filepath):
    filepath = Path(filepath)
    pattern = r'^toltec.*\.nc$'
    return re.match(pattern, filepath.name) is not None


@register_io_class("nc.toltec", identifier=identify_toltec_nc)
class NcFileIO(ExitStack):
    """This class provides methods to access data in netCDF files."""

    spec = ToltecDataFileSpec

    logger = get_logger()

    def __init__(self, source):
        super().__init__()
        self._open_nc(source)
        # setup mappers
        self.nm = NcNodeMapper(self.nc, {
                # data
                "is": "Data.Toltec.Is",
                "qs": "Data.Toltec.Qs",
                "flos": "Data.Toltec.LoFreq",
                "sweeps": "Data.Toltec.SweepFreq",
                "tones": "Header.Toltec.ToneFreq",
                "rs": "Data.Generic.Rs",
                "xs": "Data.Generic.Xs",
                # tone axis
                "tonemodelparams": "Header.Toltec.ModelParams",
                "tonemodelparamsheader": "Header.Toltec.ModelParamsHeader",
                # meta
                "kindvar": "Header.Toltec.ObsType",
                "ntones_design": "loclen",
                "ntones_max": "Header.Toltec.MaxNumTones",
                "fsmp": "Header.Toltec.SampleFreq",
                "atten_in": "Header.Toltec.InputAtten",
                "atten_out": "Header.Toltec.OutputAtten",
                "source_orig": "Header.Toltec.Filename",
                "mastervar": "Header.Toltec.Master",
                "roachid": "Header.Toltec.RoachIndex",
                "obsid": "Header.Toltec.ObsNum",
                "subobsid": "Header.Toltec.SubObsNum",
                "scanid": "Header.Toltec.ScanNum",
                # meta -- deprecated
                "flo": "Header.Toltec.LoFreq",
                "flo_offset": "Header.Toltec.LoOffset",
                # assoc
                "cal_roachid": "Header.Toltec.RoachIndex",
                "cal_obsid": "Header.Toltec.TargSweepObsNum",
                "cal_subobsid": "Header.Toltec.TargSweepSubObsNum",
                "cal_scanid": "Header.Toltec.TargSweepScanNum",
                # data shape
                "ntimes_all": "time",
                "ntones": "iqlen",
                "ntones_": "toneFreqLen",
                "nreps": "Header.Toltec.NumSamplesPerSweepStep",
                "nsweepsteps": "Header.Toltec.NumSweepSteps",
                "nsweeps_all": "numSweeps",
                "ntonemodelparams": "modelParamsNum",
                })

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

    def sync(self):
        self.nc.sync()
        for k in ('ntimes_all', ):
            old = self.meta[k]
            new = self.nm.getdim(k)
            if old != new:
                self.logger.info(f"updated {k} {old}->{new} via sync")
            self.meta[k] = new

    @cached_property
    def meta(self):
        nm = self.nm

        def logged_update_dict(l, r):
            for k, v in r.items():
                if k in l and l[k] != v:
                    self.logger.error(
                            f"inconsistent entry during update"
                            f" {k} {l[k]} -> {v}")
                l[k] = v

        result = {}
        # all
        for k in (
                "kindvar", "ntones_design", "ntones_max", "fsmp",
                "atten_in", "atten_out", "mastervar", "source_orig",
                "roachid", "obsid", "subobsid", "scanid",
                "cal_roachid", "cal_obsid", "cal_subobsid", "cal_scanid",
                "ntimes_all", "ntones", "ntones_", ):
            try:
                result[k] = nm.get(k)
            except Exception:
                self.logger.error(f"missing item in data {k}", exc_info=True)
                continue
        # sweep only
        if self.is_sweep:
            for k in (
                    "nreps", "nsweepsteps",
                    "nsweeps_all", "ntonemodelparams"):
                try:
                    result[k] = nm.get(k)
                except Exception:
                    self.logger.error(
                            f"missing item in data {k}", exc_info=True)
                    continue
            ntimespersweep = result["nsweepsteps"] * result["nreps"]
            result["nsweeps"] = result["ntimes_all"] / ntimespersweep

        # handle lofreqs, which are no longer there in new files
        if nm.hasvar("flo", "flo_offset"):
            logged_update_dict(result, {
                "flo": nm.getscalar('flo'),
                "flo_offset": nm.getscalar('flo_offset'),
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

    @cached_property
    def tone_axis(self):
        nm = self.nm
        meta = self.meta
        if not nm.hasvar("tones"):
            raise RuntimeError("no tone data found")
        # tone param header
        # tone param data
        n_tones = meta['ntones']
        if self.is_sweep:
            n_sweeps = meta['nsweeps']
            last_sweep = n_sweeps - 1
            self.logger.debug(f"load tones from {last_sweep} of {n_sweeps} sweep blocks")
        else:
            last_sweep = 0 
        tfs = nm.getvar("tones")[last_sweep, :]

        return dict(tfs=tfs)

    @cached_property
    def is_sweep(self):
        return issubclass(self.kind_cls, Sweep)

    @cached_property
    def sweep_axis(self):
        return []

    @cached_property
    def time_axis(self):
        return []

    @cached_property
    def data(self):
        return []
