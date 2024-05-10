from contextlib import ExitStack
from typing import ClassVar

import netCDF4
from tollan.utils.fileloc import FileLoc
from tollan.utils.nc import NcNodeMapper, NcNodeMapperTree

from .core import FileIODataModelBase, FileIOProtocol

__all__ = ["NcFileIOMixin"]


class NcFileIOData(FileIODataModelBase):
    """Nc file IO data."""

    @classmethod
    def validate_source(cls, source, **kwargs):  # noqa: ARG003
        if isinstance(source, netCDF4.Dataset):
            source = source.filepath()
        source = FileLoc(source)
        if source.path.suffix not in [".nc"]:
            raise ValueError("invalid file format.")
        return source

    def _set_open_state(
        self,
        node_mapper=None,
        exitstack=None,
        **kwargs,  # noqa: ARG002
    ):
        if node_mapper is None:
            io_obj = netCDF4.Dataset(self.source)
        elif isinstance(node_mapper, NcNodeMapper):
            io_obj = node_mapper.open(self.source).__enter__()
        elif isinstance(node_mapper, NcNodeMapperTree):
            if exitstack is None:
                raise ValueError("exitstack required to open node mapper tree.")
            io_obj = node_mapper.open(self.source, exitstack=exitstack)
        self.io_obj = io_obj

    def _set_close_state(
        self,
        node_mapper=None,
        exitstack: None | ExitStack = None,
        **kwargs,  # noqa: ARG002
    ):
        if node_mapper is None:
            self.io_obj.close()
        elif isinstance(node_mapper, NcNodeMapper):
            self.io_obj.__exit__()
        elif isinstance(node_mapper, NcNodeMapperTree):
            # this will be handled by the exitstack
            if exitstack is None:
                raise ValueError("exitstack required to close node mapper tree.")
            exitstack.close()
        self.io_obj = None


class NcFileIOMixin(FileIOProtocol):
    """A mixin class that provide handling of netCDF file IO."""

    _node_mapper_defs: ClassVar[dict[str, str | dict]]
    _node_mappers: NcNodeMapperTree

    def _init_node_mappers(self):
        self._node_mappers = NcNodeMapperTree(self._node_mapper_defs)

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
        return super().open(node_mapper=self.node_mappers, exitstack=self)
