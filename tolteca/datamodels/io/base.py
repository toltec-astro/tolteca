#! /usr/bin/env python

from contextlib import ExitStack
from astropy.utils.metadata import MetaData
from pathlib import Path
from tollan.utils import fileloc, FileLoc
# from tollan.utils.nc import ncopen, NcNodeMapperMixin
from .registry import io_registry as io_registry


__all__ = ['DataFileIOError', 'DataFileIO']


class DataFileIOError(RuntimeError):
    pass


class DataFileIO(ExitStack):
    """A base class to help access data file contents.

    This class provide a common interface to work with file location
    :attr:`file_loc` (subclasses implement :attr:`_file_loc`) and low level
    file I/O object :attr:`file_obj` (subclasses implement :attr:`_file_obj`),
    which shall be implemented/setup by the subclasses.

    This class is also a subclass of `~contextlib.ExitStack`, which can be
    used to manage context objects such as opened files.

    Subclasses of this class can provide attribute :attr:`io_registry_info`,
    which is a dict of keyword arguments that is passed
    to :meth:`IORegistry.register` to register itself to `io_registry`, make it
    one of the supported format of :meth:`IORegistry.open`.

    """

    def _setup(self, file_loc, file_obj, meta):
        """Setup the instance."""
        self._file_loc = self._normalize_file_loc(file_loc)
        self._file_obj = file_obj
        if meta is not None:
            self.meta.update(meta)

    def _normalize_file_loc(self, file_loc):
        """Return a `~tollan.utils.FileLoc` object."""
        if file_loc is None or isinstance(file_loc, FileLoc):
            return file_loc
        return fileloc(file_loc)

    def _normalize_source(self, source):
        """Return a `pathlib.Path` object."""
        # ensure that we don't have source set twice.
        if source is not None and self._source is not None:
            raise ValueError(
                    'source needs to be None for '
                    'object with source set at construction time.')
        # use the constructor source
        if source is None:
            source = self._source
        if source is None:
            raise ValueError('source is not specified')
        if isinstance(source, FileLoc):
            if not source.is_local:
                raise ValueError('source should point to a local file.')
            source = source.path
        elif isinstance(source, (Path, str)):
            pass
        else:
            pass
        return source

    meta = MetaData(copy=False)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        io_registry_info = getattr(cls, 'io_registry_info', None)
        if io_registry_info is None:
            return
        io_registry.register(cls, **io_registry_info)

    @property
    def file_loc(self):
        """The file location."""
        return self._file_loc

    @property
    def file_uri(self):
        """The file URI."""
        return self.file_loc.uri

    @property
    def filepath(self):
        """The file path."""
        return self.file_loc.path

    @property
    def file_obj(self):
        """The low level file object.

        """
        return self._file_obj

    def __repr__(self):
        r = f"{self.__class__.__name__}({self.file_loc})"
        return r

    def open(self):
        """Return the context for access the `file_obj`."""
        if self.file_obj is None:
            raise DataFileIOError(f"file object not available for {self}")
        return self.file_obj.open()
