#! /usr/bin/env python

from tolteca.utils.nc import ncopen, ncinfo
from tolteca.utils.log import get_logger
from contextlib import ExitStack
from pathlib import Path


class NcScope(ExitStack):
    """A class that provides live view to netCDF file."""

    logger = get_logger()

    def __init__(self, source):
        super().__init__()
        self._open_nc(source)

    def _open_nc(self, source):
        nc, _close = ncopen(source)
        self.push(_close)
        self.logger.debug("ncinfo: {}".format(ncinfo(nc)))
        self.nc = nc
        self.filepath = Path(nc.filepath())

    def sync(self):
        self.nc.sync()
