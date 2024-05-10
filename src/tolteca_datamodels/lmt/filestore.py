from pathlib import Path
from typing import ClassVar, Literal, get_args

import pandas as pd

from ..filestore import FileStoreBase
from ..toltec.filestore import ToltecFileStore

LmtInstruType = Literal["toltec"]


class LmtInstruInterface:
    """LMT instrument insterface."""

    instruments: ClassVar[list[LmtInstruType]] = list(get_args(LmtInstruType))
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
            for interface in self.interfaces
        }
        self._instru_filestore = {
            instru: self._get_instru_filestore(instru) for instru in self.instruments
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
    interfaces: ClassVar = LmtInterface.interfaces
    instruments: ClassVar = LmtInstruInterface.instruments
    _instru_filestore_cls: ClassVar = LmtInstruInterface.instru_filestore_cls
    _instru_interface: ClassVar = LmtInstruInterface.instru_interface
    _interface_instru: ClassVar = LmtInstruInterface.interface_instru

    def _get_instru_filestore(self, instru):
        """Return filestore object for instrument."""
        instru_path = self.get_instru_path(instru)
        if not instru_path.exists():
            return None
        filestore_cls = self._instru_filestore_cls[instru]
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

    def get_instru_path(self, instru):
        """Return the instrument subpath."""
        interface = self._instru_interface[instru]
        return self.interface_path[interface]

    def get_instru_filestore(self, instru):
        """Return the filestore instance for instrument."""
        return self._instru_filestore[instru]

    @property
    def toltec(self) -> None | ToltecFileStore:
        """The toltec data filestore."""
        return self.get_instru_filestore("toltec")

    @classmethod
    def _make_interface_path_info_table(
        cls,
        path_maker,
        instru_path_maker,
        exist_only=True,
    ):
        """Return the path info table."""
        result = []
        instru_results = []
        for interface in cls.interfaces:
            instru = cls._interface_instru.get(interface, None)
            if instru is not None:
                instru_result = instru_path_maker(instru)
                if instru_result is not None:
                    instru_results.append(instru_result)
            p = path_maker(interface)
            if p is None:
                continue
            if exist_only and not p.exists():
                continue
            result.append(
                {
                    "interface": interface,
                    "instru": instru,
                    "path": p,
                },
            )
        result = [] if not result else [pd.DataFrame.from_records(result)]
        result.extend(r for r in instru_results if r is not None)
        if not result:
            return None
        return pd.concat(result, ignore_index=True, axis=0).fillna(
            pd.NA,
        )

    def get_path_info_table(self, exist_only=True):
        """Return the path info table."""

        def _path_maker(interface):
            return self._interface_subpath[interface]

        def _instru_path_maker(instru):
            instru_fs = self.get_instru_filestore(instru)
            if instru_fs is not None:
                tbl = instru_fs.get_path_info_table(
                    exist_only=exist_only,
                )
                tbl["instru"] = instru
                return tbl
            return None

        return self._make_interface_path_info_table(
            _path_maker,
            _instru_path_maker,
            exist_only=exist_only,
        )

    def get_symlink_info_table(self, exist_only=True):
        """Return the file table for current files."""
        linkpath_map = self._interface_file_current_symlink

        def _path_maker(interface):
            return linkpath_map.get(interface, None)

        def _instru_path_maker(instru):
            instru_fs = self.get_instru_filestore(instru)
            if instru_fs is not None:
                tbl = instru_fs.get_symlink_info_table(
                    exist_only=exist_only,
                )
                tbl["instru"] = instru
                return tbl
            return None

        return self._make_interface_path_info_table(
            _path_maker,
            _instru_path_maker,
            exist_only=exist_only,
        )
