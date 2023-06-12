from contextlib import ExitStack
from pathlib import Path
from typing import Any

from tollan.utils.fileloc import FileLoc

__all__ = ["DataFileIOError", "DataFileIO"]


class DataFileIOError(RuntimeError):
    """An exception class related to data file IO."""


class DataFileIO(ExitStack):
    """A base class to help access data file contents.

    This class provide a common interface to work with data read and write.
    It manages the properties  ``file_loc_orig``, ``file_loc``, and ``io_obj``:

    * ``file_loc_orig``. This is ``FileLoc`` instance of the origin of the file.

    * ``file_loc``. This is ``FileLoc`` instance of local file location to be opened.

    * ``io_obj``. This is the data IO object after openning ``file_loc``.

    * ``meta``. Arbitury data.

    Subclass should implement ``_file_loc_orig``, ``_file_loc`` and ``_io_obj``.

    This class is also a subclass of `~contextlib.ExitStack`, which can be
    used to manage context objects such as opened files.

    """

    _file_loc_orig: None | FileLoc
    _file_loc: None | FileLoc
    _io_obj: Any
    _meta: dict

    def __init__(self, file_loc_orig=None, file_loc=None, io_obj=None, meta=None):
        self._file_loc_orig = self._validate_file_loc(file_loc_orig)
        self._file_loc = self._validate_file_loc(file_loc)
        self._io_obj = io_obj
        self._meta = {}
        if meta is not None:
            self.meta.update(meta)
        # init the exit stack
        super().__init__()

    def _validate_file_loc(self, file_loc):
        if file_loc is None or isinstance(file_loc, FileLoc):
            return file_loc
        return FileLoc(file_loc)

    @property
    def file_loc_orig(self):
        """The original file location.

        Fallback to ``file_loc`` is not specified
        """
        if self._file_loc_orig is None:
            return self.file_loc
        return self._file_loc_orig

    @property
    def file_loc(self):
        """The file location."""
        return self._file_loc

    @property
    def file_uri(self):
        """The file URI."""
        if self.file_loc is None:
            return None
        return self.file_loc.uri

    @property
    def filepath(self):
        """The file path."""
        if self.file_loc is None:
            return None
        return self.file_loc.path

    @property
    def io_obj(self):
        """The low level data io object."""
        return self._io_obj

    @property
    def meta(self):
        """The metadata."""
        return self._meta

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file_loc})"

    def open(self):
        """Open the data IO."""
        return NotImplemented
