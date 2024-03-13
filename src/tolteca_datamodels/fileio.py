import inspect
from collections import UserDict
from contextlib import ExitStack
from typing import Any, ClassVar, Protocol

import netCDF4
from pydantic import BaseModel, Field
from tollan.utils.fileloc import FileLoc
from tollan.utils.log import logger
from tollan.utils.nc import NcNodeMapper
from typing_extensions import Self

__all__ = ["FileIOError", "FileIO", "NcFileIOMixin"]


class FileIOError(RuntimeError):
    """An exception class related to file IO."""


io_cls_registry: set["FileIO"] = set()
"""A registry to hold all file IO subclasses."""


def identify_io_cls(*args, **kwargs):
    """Return the IO class from the registry."""
    for io_cls in io_cls_registry:
        if io_cls.identify(*args, **kwargs):
            return io_cls
    raise ValueError("unable to identify IO class.")


class MetadataBaseModel(BaseModel):
    """A base class for metadata models."""


class FileIODataBaseModel(BaseModel):
    """A container class for file io data."""

    meta: MetadataBaseModel
    file_loc: None | FileLoc = Field(description="The local data file location.")
    source_loc: None | FileLoc = Field(description="The data source location.")
    file_obj: None | Any = Field(description="externally passed file_obj")


class FileIO(ExitStack):
    """A base class to help access data file contents.

    This class provide a common interface to work with file IO.
    It manages the file IO data in the ``_io_data`` attribute,
    and the open/close state of the underlying object through the
    ``_io_obj`` attribute.

    Subclass should implement ``_resolve_io_data()``, ``identify()``,
    and ``open()``.

    This class is also a subclass of `~contextlib.ExitStack`, which can be
    used to manage context objects such as opened files.
    """

    _io_obj: None | Any
    _io_data: FileIODataBaseModel
    _io_data_cls: ClassVar[type[FileIODataBaseModel]]
    _io_data_meta_cls: ClassVar[type[MetadataBaseModel]]

    @classmethod
    def identify(cls, *_args, **_kwargs) -> bool:
        """Identify if the given file can be opened by this class."""
        return NotImplemented

    def open(self, *_args, **_kwargs) -> Self:
        """Open this file for IO."""
        return NotImplemented

    def __init_subclass__(cls, *args, **kwargs):
        io_cls_registry.add(cls)
        cls._io_data_cls = inspect.get_annotations(cls)["_io_data"]
        cls._io_data_meta_cls = inspect.get_annotations(cls._io_data_type)["meta"]
        return super().__init_subclass__(*args, **kwargs)

    def __init__(self, source, **kwargs):
        self._io_data = self._resolve_io_data(source, **kwargs)
        # init the exit stack
        super().__init__()
        self.callback(self._set_close_state)

    def _resolve_io_data(self, *_args, **_kwargs):
        """Return io data instance."""
        return NotImplemented

    def _set_open_state(self, io_obj):
        """Set the open state."""
        logger.debug(f"set io obj {io_obj}")
        self._io_obj = io_obj

    def _set_close_state(self):
        """Set the close state."""
        logger.debug(f"unset io obj {self._io_obj}")
        self._io_obj = None

    def is_open(self):
        """Return True if the IO is open."""
        return self.io_obj is not None

    @property
    def io_obj(self):
        """The active data io obj."""
        return self._io_obj

    @property
    def file_obj(self):
        """The file object."""
        return self._io_data.file_obj

    @property
    def source_loc(self):
        """The data source location."""
        return self._io_data.source_loc

    @property
    def source_url(self):
        """The dawta source URL."""
        if self.file_loc is None:
            return None
        return self.source_loc.url

    @property
    def file_loc(self):
        """The local data file location."""
        return self._io_data.file_loc

    @property
    def file_url(self):
        """The file URL."""
        if self.file_loc is None:
            return None
        return self.file_loc.url

    @property
    def filepath(self):
        """The file path."""
        if self.file_loc is None:
            return None
        return self.file_loc.path

    @property
    def meta(self):
        """The metadata."""
        return self._io_data.meta

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file_loc})"


class FileIOProtocol(Protocol):
    _io_data_cls: ClassVar[type[FileIODataBaseModel]]
    _io_data_meta_cls: ClassVar[type[MetadataBaseModel]]

    def is_open(): ...

    def _set_open_state(): ...

    @property
    def filepath(): ...

    def enter_context(): ...

    def close(): ...


class NodeMapperTree(UserDict[str, dict | NcNodeMapper]):
    """A help container to hold a collection of nc node mappers."""

    def __init__(self, node_mapper_defs: dict[str, dict]):
        node_mapper_defs = node_mapper_defs.copy()
        # ensure root node
        node_mapper_defs.setdefault("__root__", {})
        self.__init__(self._create_nc_node_mappers(node_mapper_defs))

    def nc_node(self):
        return self["__root__"].nc_node

    @staticmethod
    def _create_nc_node_mappers(nc_node_mapper_defs) -> dict:
        """Create node mappers for the inner most level of dict of node_maps."""

        def _get_sub_node(n) -> Any:
            if all(not isinstance(v, dict) for v in n.values()):
                return NcNodeMapper(nc_node_map=n)
            return {k: _get_sub_node(v) for k, v in n.items()}

        return _get_sub_node(nc_node_mapper_defs)

    def open(self, filepath, exitstack: None | ExitStack = None):
        # TODO: here we alwasy open the netcdf regardless if
        # an opened file is passed to source and set as file_obj.
        # need to revisit this later
        nc_node = None

        def _open_sub_node(n: dict):
            nonlocal nc_node
            for v in n.values():
                if isinstance(v, NcNodeMapper):
                    if nc_node is None:
                        v.open(filepath)
                        nc_node = v.nc_node
                    else:
                        v.set_nc_node(nc_node)
                    # push to the exit stack
                    if exitstack is not None:
                        exitstack.enter_context(v)
                else:
                    _open_sub_node(v)

        _open_sub_node(self)
        return nc_node


class NcFileIOMixin(FileIOProtocol):
    """A mixin class that provide handling of netCDF file IO."""

    _node_mappers: NodeMapperTree

    @classmethod
    def _resolve_io_data(cls, source, source_loc=None, **kwargs):
        if isinstance(source, netCDF4.Dataset):
            # TODO: here we always re-open the file when open is requested
            # this allows easier management of the pickle but
            # may not be desired. Need to revisit this later.
            file_loc = source.filepath()
            file_obj = None
        else:
            file_loc = source
            file_obj = None
        return cls._io_data_cls.model_validate(
            file_loc=file_loc,
            file_obj=file_obj,
            source_loc=source_loc,
            **kwargs,
        )

    def _init_node_mapper(self, node_mapper_defs):
        self._node_mappers = self.NodeMapperTree(node_mapper_defs)

    @property
    def node_mappers(self):
        """The tree of low level netCDF dataset mappers."""
        return self._node_mappers

    @property
    def nc_node(self):
        """The low level netCDF data IO object."""
        return self.node_mappers.nc_node

    def open(self):
        """Return the context for netCDF file IO."""
        # we just use one of the node_mapper to open the dataset, and
        # set the rest via set_nc_node
        # the node_mapper.open will handle the different types of source
        if self.is_open():
            # the file is already open
            return self
        nc_node = self.node_mappers.open(self.filepath, self)
        self._set_open_state(nc_node)
        return self

    def __getstate__(self):
        # need to reset the object before pickling
        is_open = self.is_open()
        self.close()
        return {
            "_auto_close_on_pickle_wrapped": self.__dict__,
            "_auto_close_on_pickle_is_open": is_open,
        }

    def __setstate__(self, state):
        state, is_open = (
            state["_auto_close_on_pickle_wrapped"],
            state["_auto_close_on_pickle_is_open"],
        )
        self.__dict__.update(state)
        if is_open:
            # try open the object
            self.open()

    @classmethod
    def identify(cls, source):
        """Return if this class can handle ths given source."""
        if isinstance(source, netCDF4.Dataset):
            return True
        try:
            file_loc = FileLoc(source)
        except ValueError:
            return False
        else:
            return file_loc.path.suffix == ".nc"
