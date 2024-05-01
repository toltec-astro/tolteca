from pathlib import Path
from typing import Any

from astropy.utils.decorators import format_doc
from tollan.utils.fileloc import FileLoc
from tollan.utils.log import logger
from wrapt import ObjectProxy

from ..base import FileIOBase, FileIOError
from .file0 import guess_meta_from_source
from .types import ToltecDataKind

base_doc = """{__doc__}

    Parameters
    ----------
    source : str, `pathlib.Path`, `FileLoc`, `netCDF4.Dataset`
        The data file location or data file object.
    source_loc : str, `pathlib.Path`, `FileLoc`
        The data file location of source, in case source is data object.
    open : bool
        If True, open file at constuction time.
    load_meta_on_open : bool
        If True, the meta data will be loaded upon opening of the file.
    raise_on_unknown_data_kind : bool
        If True, error raises when no data kind can be inferred.
    data_kind : str, ToltecDataKind
        If provided, try to open the data as this data kind.
"""

_io_cls_registry = set()


def _identify_io_cls(source, source_loc=None):
    """Return the IO class for `source`."""
    if isinstance(source, str | Path | FileLoc):
        # map extension to io cls
        file_loc = FileLoc(source)
        file_obj = None
    else:
        file_loc = FileLoc(source_loc) if source_loc is not None else None
        file_obj = source
    if file_loc is None and file_obj is None:
        raise ValueError("at least one of source, source_loc is required.")
    for io_cls in _io_cls_registry:
        if io_cls.identify(file_loc=file_loc, file_obj=file_obj):
            return io_cls
    return None


@format_doc(base_doc)
class ToltecFileIO(FileIOBase):
    """A low level base class to open and read TolTEC files."""

    __abstract__ = True  # skip the registration

    def __init__(  # noqa: PLR0913
        self,
        source,
        source_loc=None,
        open=True,
        load_meta_on_open=True,
        raise_on_unknown_data_kind=True,
        data_kind=None,
    ):
        self._source = source
        self._source_loc = source_loc
        self._load_meta_on_open = load_meta_on_open
        self._raise_on_unknown_data_kind = raise_on_unknown_data_kind
        self._data_kind_to_validate = self._resolve_data_kind(data_kind)

        file_info = self._get_source_file_info(
            source=source,
            source_loc=source_loc,
        )
        super().__init__(**file_info)
        self._post_init()
        if open:
            self.open()

    def _post_init(self):
        """Do additional init of the instance."""
        self._load_data_kind()

    _source: Any
    _source_loc: FileLoc | None
    _load_meta_on_open: bool
    _raise_on_unknown_data_kind: bool

    @staticmethod
    def _resolve_data_kind(arg):
        if arg is None:
            return arg
        if isinstance(arg, str):
            return ToltecDataKind[arg]
        if isinstance(arg, ToltecDataKind):
            return arg
        raise TypeError("invalid type for data_kind.")

    @classmethod
    def _get_source_file_info(cls, _source, _source_loc):
        # subclass implement this: maps the source spec to file locs.
        return NotImplemented

    def _guess_meta_from_file_loc(self):
        self._meta.update(guess_meta_from_source(self.file_loc))

    def _load_data_kind_meta_from_io_obj(self):
        # subclass implement this: return the data kind meta from io obj.
        return NotImplemented

    def _load_meta_from_io_obj(self):
        # subclass implement this: return the meta from io obj.
        return NotImplemented

    def _load_data_kind(self):
        def _check_data_kind(on_identified, on_missing, error_msg):
            meta = self._meta
            if "data_kind" in meta:
                data_kind = meta["data_kind"]
                logger.debug(f"identified data_kind={meta['data_kind']}")
                if on_identified is not None:
                    on_identified(meta)
            elif on_missing is not None:
                on_missing(meta)
            data_kind = meta.get("data_kind", ToltecDataKind.Unknown)
            if (
                error_msg is not None
                and self._raise_on_unknown_data_kind
                and data_kind == ToltecDataKind.Unknown
            ):
                raise FileIOError(error_msg)
            meta["data_kind"] = data_kind

        # this get called in case data kind is found in metadata.
        data_kind_to_validate = self._data_kind_to_validate
        if data_kind_to_validate is None:
            on_identified = None
        else:

            def on_identified(meta):
                data_kind = meta["data_kind"]
                if data_kind not in data_kind_to_validate:
                    raise FileIOError(
                        (
                            f"identified {data_kind=} is not a valid"
                            f" {data_kind_to_validate}"
                        ),
                    )

        # load meta from file
        self._guess_meta_from_file_loc()
        if not self.io_state.is_open():
            error_msg = f"unable to guess data kind from file loc: {self.file_loc}"
            _check_data_kind(
                on_identified=on_identified,
                on_missing=None,
                error_msg=error_msg,
            )
        else:
            self._load_data_kind_meta_from_io_obj()
            error_msg = f"unable to identify data kind from file obj: {self.file_obj}"
            _check_data_kind(
                on_identified=on_identified,
                on_missing=None,
                error_msg=error_msg,
            )

    @property
    def data_kind(self):
        """The TolTEC data kind."""
        return self.meta["data_kind"]

    def _set_open_state(self, io_obj):
        io_obj_type = type(io_obj)
        self.io_state.set_open_state(io_obj)
        if self._load_meta_on_open:
            logger.debug(f"load meta from {io_obj_type=}")
            self._load_meta_from_io_obj()


@format_doc(base_doc)
class ToltecData(ObjectProxy):
    """The TolTEC data file reader class.

    This is a wrapper to the underlying ``ToltecDataFileIO`` instance.
    """

    def __init__(self, source, source_loc=None, **kwargs):
        io_cls = _identify_io_cls(source, source_loc=source_loc)
        if io_cls is None:
            raise FileIOError(f"unable to identify io class for {source=}.")
        io_inst = io_cls(source, source_loc=source_loc, **kwargs)
        super().__init__(io_inst)
