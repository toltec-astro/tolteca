from pathlib import Path
from typing import ClassVar, Literal, get_args

import pandas as pd

from ..filestore import FileStoreBase
from ..toltec.file import guess_info_from_sources
from ..toltec.filestore import ToltecFileStore

LmtInstruType = Literal["toltec"]


class LmtInstruInterface:
    """LMT instrument insterface."""

    instruments: ClassVar[list[LmtInstruType]] = get_args(LmtInstruType)
    instru_filestore_cls: ClassVar[dict[LmtInstruType, type]] = {
        "toltec": ToltecFileStore,
    }

    instru_interface: ClassVar = {instru: instru for instru in instruments}
    interface_instru: ClassVar = {v: k for k, v in instru_interface.items()}
    interfaces: ClassVar = list(instru_interface.values())


LmtInterfaceType = LmtInstruType | Literal["tel"]


class LmtInterface:
    """LMT insterface."""

    interfaces: ClassVar[list[LmtInterfaceType]] = LmtInstruInterface.interfaces + [
        "tel",
    ]


class LmtFileStore(FileStoreBase):
    """The data_lmt data tree."""

    def __init__(self, path="data_lmt"):
        super().__init__(path)
        self._interface_subpath = {
            interface: self._path.joinpath(self._interface_subpath_name[interface])
            for interface in LmtInterface.interfaces
        }
        self._instru_filestore = {
            self._get_instru_filestore(instru)
            for instru in LmtInstruInterface.instruments
        }
        # current file links
        self._interface_file_current_symlink = {
            interface: self.interface_path[interface].joinpath(symlink_name)
            for (
                interface,
                symlink_name,
            ) in self._interface_file_current_symlink_name.items()
        }

    _interface_subpath: dict[LmtInterfaceType, Path]
    _instru_filestore: dict[LmtInstruType, type]
    _interface_file_current_symlink: dict[LmtInterfaceType, Path]

    _interface_subpath_name: ClassVar = {
        interface: interface for interface in LmtInterface.interfaces
    }

    _interface_file_current_symlink_name: ClassVar = {"tel": "tel.nc"}

    def _get_instru_filestore(self, instru):
        """Return filestore object for instrument."""
        instru_path = self.get_instru_path(instru)
        if not instru_path.exists():
            return None
        filestore_cls = LmtInstruInterface.instru_filestore_cls[instru]
        return filestore_cls(instru_path)

    @property
    def interface_path(self):
        """The interface subpath directories."""
        return self._interface_subpath

    @property
    def tel_path(self):
        """Return the telescope subpath."""
        return self.interface_path["tel"]

    @property
    def toltec_path(self):
        """Return the toltec instrument subpath."""
        return self.interface_path["toltec"]

    @property
    def instruments(self):
        """The list of instruments."""
        return LmtInstruInterface.instruments

    def get_instru_path(self, instru):
        """Return the instrument subpath."""
        interface = LmtInstruInterface.instru_interface[instru]
        return self.interface_path[interface]

    def get_path_info_table(self, exist_only=True, instru_only=False):
        """Return the path info table."""
        result = []
        for interface in LmtInterface.interfaces:
            instru = LmtInstruInterface.interface_instru.get(interface, None)
            if instru_only and instru is None:
                continue
            p = self._interface_subpath[interface]
            if exist_only and not p.exists():
                continue
            result.append(
                {
                    "interface": interface,
                    "instru": instru,
                    "path": p,
                },
            )
        return pd.DataFrame.from_records(result)

    def get_instru_path_info_table(self, exist_only=True):
        """Return the instrument interface subpaths."""
        return self.get_path_info_table(exist_only=exist_only, instru_only=True)

    def get_instru_filestore(self, instru):
        """Return the filestore instance for instrument."""
        return self._instru_filestore[instru]

    def get_current_file_info_table(self):
        """Return the file info table for current files."""
        linkpaths = self._interface_file_current_symlink.values()
        return guess_info_from_sources(linkpaths)

    def get_current_file_info(self):
        """Return the info for the most current file."""
        return self.get_current_file_info_table().toltec_file.get_latest()
