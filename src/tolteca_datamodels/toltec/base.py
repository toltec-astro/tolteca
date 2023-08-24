from typing import Any

from tollan.utils.fileloc import FileLoc
from tollan.utils.log import logger

from ..base import DataFileIO, DataFileIOError
from .file import guess_meta_from_source
from .types import ToltecDataKind


class ToltecDataFileIO(DataFileIO):
    """A base class to load local TolTEC data.

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
    """

    def __init__(  # noqa: PLR0913
        self,
        source,
        source_loc=None,
        open=True,
        load_meta_on_open=True,
        raise_on_unknown_data_kind=True,
    ):
        self._source = source
        self._source_loc = source_loc
        self._load_meta_on_open = load_meta_on_open
        self._raise_on_unknown_data_kind = raise_on_unknown_data_kind

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

    # def _reset_meta_state(self):
    #     """Reset the instance state for meta data."""
    #     self._meta_cached.clear()
    #     for k in ["meta", "data_kind"]:
    #         if k in self.__dict__:
    #             del self.__dict__[k]

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
        def _check_data_kind_or_error(error_msg):
            meta = self._meta
            if "data_kind" not in meta:
                if self._raise_on_unknown_data_kind:
                    raise DataFileIOError(error_msg)
                meta["data_kind"] = ToltecDataKind.Unknown
            data_kind = meta["data_kind"]
            logger.debug(f"identified {data_kind=}")

        self._guess_meta_from_file_loc()
        if not self.io_state.is_open():
            _check_data_kind_or_error(
                f"unable to guess data kind from file loc: {self.file_loc}",
            )
        else:
            self._load_data_kind_meta_from_io_obj()
            _check_data_kind_or_error(
                f"unable to identify data kind from {self.io_obj}",
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
