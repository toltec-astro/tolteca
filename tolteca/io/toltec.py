#! /usr/bin/env python

from cached_property import cached_property

from ..kidsdata import (
        RawTimeStream, SolvedTimeStream, VnaSweep, TargetSweep)
from ..utils.nc import ncopen, ncinfo, NcNodeMapper
from ..utils.log import get_logger
from .registry import register_io_class
from pathlib import Path
from contextlib import ExitStack
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

    logger = get_logger()

    def __init__(self, source):
        super().__init__()
        self._open_nc(source)
        # setup mappers
        self.nm = NcNodeMapper(self.nc, {
                "kind": "Header.Toltec.ObsType",
                "is": "Data.Toltec.Is",
                "qs": "Data.Toltec.Qs",
                "sweeps": "Data.Toltec.SweepFreq",
                "tones": "Header.Toltec.ToneFreq",
                "rs": "Data.Generic.Rs",
                "xs": "Data.Generic.Xs",
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
        if m.hasvar('kind'):
            kindvar = m.getvar('kind')
            self.logger.debug(f"found kindvar={kindvar} from {m['kind']}")

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
        if not m.hasvar("sweeps") and kind_cls in (VnaSweep, TargetSweep):
            kind_cls = RawTimeStream
            self.logger.debug(f"updated kind_cls={kind_cls}")
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
    def meta(self):
        return {}

    @cached_property
    def tone_axis(self):
        return []

    @cached_property
    def sweep_axis(self):
        return []

    @cached_property
    def time_axis(self):
        return []

    @cached_property
    def data(self):
        return []
