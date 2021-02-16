#! /usr/bin/env python

from tollan.utils.log import get_logger
from tollan.utils.nc import ncopen, NcNodeMapperMixin, NcNodeMapperError
from .base import DataFileIO


__all__ = ['SimpleNcFileIO', ]


class SimpleNcFileIO(DataFileIO, NcNodeMapperMixin):
    """A class to read netCDF data files.

    Subclass of this class should define :attr:`_nc_node_map` to make
    available the low-level operation with the `netCDF4.Dataset`.

    Parameters
    ----------
    source : str, `pathlib.Path`, `FileLoc`, `netCDF4.Dataset`
        The data file location or netCDF dataset.
        This is passed to `~tollan.utils.nc.NcNodeMapper`.
    """

    logger = get_logger()

    def __init__(self, source=None):

        source = self._normalize_file_loc(source)
        self._source = source
        # init the exit stack
        super(DataFileIO, self).__init__()

        # if source is given, we just open it right away
        if self._source is not None:
            self.open()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.filepath})'

    def open(self, source=None):
        """Return a context to operate on `source`.

        Parameters
        ----------
        source : str, `pathlib.Path`, `FileLoc`, `netCDF4.Dataset`, optional
            The data file location or netCDF dataset. If None, the
            source passed to constructor is used. Noe that source has to
            be None if it has been specified in the constructor.
        """
        source = self._normalize_source(source)
        if self.file_obj is not None:
            return self
        self._nc_node = self.enter_context(ncopen(source))
        return self

    @property
    def _file_obj(self):
        # we expose the raw netcdf dataset as the low level file object.
        # this returns None if no dataset is open.
        try:
            return self.nc_node
        except NcNodeMapperError:
            return None

    @property
    def _file_loc(self):
        # here we return the _source if it is passed to the constructor
        # we had ensured in open that if self._source is given,
        # the source passed to open can only be None.
        # so that source is always the same as self.nc_node.file_loc
        if self._source is not None:
            return self._source
        # if no dataset is open, we just return None
        if self._file_obj is None:
            return None
        # the opened dataset file loc.
        return self.nc_node.file_loc
