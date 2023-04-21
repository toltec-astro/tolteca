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
    It manages the properties  ``file_loc``, ``source``, and ``io_obj``:

    * ``file_loc``. This is ``FileLoc`` instance that describe the origing of the file.

    * ``source``. This is the actual local data source (file) to be opened.

    * ``io_obj``. This is the object returned by opening the source.

    * ``meta``. Arbitury data.

    Subclass should implement ``_file_loc``, ``_source`` and ``_io_obj``.

    This class is also a subclass of `~contextlib.ExitStack`, which can be
    used to manage context objects such as opened files.

    """

    _file_loc: None | FileLoc
    _source: None | Path
    _io_obj: Any
    _meta: dict

    def __init__(self, file_loc=None, source=None, io_obj=None, meta=None):
        self._file_loc = self._validate_file_loc(file_loc)
        self._source = source
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

    def _resolve_source_arg(self, source, remote_ok=False):
        """Resolve source argument passed to methods.

        This helper is need to allow supporting specifying source at
        construction time or when ``open`` is called.
        """
        # ensure that we don't have source set twice.
        if source is not None and self._source is not None:
            raise ValueError(
                (
                    "source needs to be None for "
                    "object with source set at construction time."
                ),
            )
        # use the constructor source
        if source is None:
            source = self._source
        if source is None:
            raise ValueError("source is not specified")
        if isinstance(source, FileLoc):
            if source.is_remote and not remote_ok:
                raise ValueError("source should point to a local file.")
            source = source.path
        elif isinstance(source, (str, Path)):
            source = Path(source)
        else:
            raise TypeError(f"invalid source type {type(source)}.")
        return source

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
    def source(self):
        """The data source path."""
        return self._source

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
        """Return the context for access the `file_obj`."""
        if self.io_obj is None:
            raise DataFileIOError(f"file object not available for {self}")
        return self.io_obj.open()
