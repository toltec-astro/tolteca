import itertools
from pathlib import Path
from typing import ClassVar

import pandas as pd

from ..filestore import FileStoreBase
from .types import ToltecInterface, ToltecMaster, ToltecMasterType, ToltecRoachInterface


class ToltecFileStore(FileStoreBase):
    """The toltec data tree."""

    def __init__(self, path="data_lmt/toltec"):
        super().__init__(path)
        self._master_subpath = {
            master: self._path.joinpath(self._master_subpath_name[master])
            for master in self.masters
        }
        self._master_interface_subpath = {
            master: {
                interface: self._master_subpath[master].joinpath(
                    self._interface_subpath_name[interface],
                )
                for interface in self.interfaces
            }
            for master in self.masters
        }
        # current file links
        self._master_interface_file_current_symlink = {
            master: {
                interface: (
                    self.master_interface_path[master][interface]
                    if master is not None
                    else self.path
                ).joinpath(
                    symlink_name,
                )
                for (
                    interface,
                    symlink_name,
                ) in self._interface_file_current_symlink_name.items()
            }
            for master in self.masters + [None]
        }

    _master_subpath: dict[str, Path]
    _master_interface_subpath: dict[ToltecMasterType, dict[str, Path]]
    _master_interface_file_current_symlink: dict[
        None | ToltecMasterType,
        dict[str, Path],
    ]

    _master_subpath_name: ClassVar[dict[ToltecMasterType, str]] = {
        "tcs": "tcs",
        "ics": "ics",
    }
    _interface_subpath_name: ClassVar = {
        interface: interface for interface in ToltecInterface.interfaces
    }
    _interface_file_current_symlink_name: ClassVar = {
        interface: f"{interface}.nc" for interface in ToltecInterface.interfaces
    }
    masters: ClassVar = ToltecMaster.masters
    interfaces: ClassVar = ToltecInterface.interfaces
    roaches: ClassVar = ToltecRoachInterface.roaches
    _roach_interface: ClassVar = ToltecRoachInterface.roach_interface
    _interface_roach: ClassVar = ToltecRoachInterface.interface_roach

    @property
    def master_interface_path(self):
        """The path."""
        return self._master_interface_subpath

    def get_roach_path(self, master, roach):
        """Return the roach interface subpath."""
        interface = self._roach_interface[roach]
        return self.master_interface_path[master][interface]

    @classmethod
    def _make_master_interface_path_info(
        cls,
        path_maker,
        iter_master_interface,
        exist_only=True,
    ):
        result = []
        for master, interface in iter_master_interface:
            roach = cls._interface_roach.get(interface, None)
            p = path_maker(master, interface)
            if p is None:
                continue
            if exist_only and not p.exists():
                continue
            result.append(
                {
                    "master": master,
                    "interface": interface,
                    "roach": roach,
                    "path": p,
                },
            )
        if not result:
            return None
        return pd.DataFrame.from_records(result).astype(
            {
                "roach": "Int64",
            },
        )

    def get_path_info_table(self, exist_only=True):
        """Return the path info table."""

        def _path_maker(master, interface):
            return self.master_interface_path[master][interface]

        return self._make_master_interface_path_info(
            _path_maker,
            itertools.product(self.masters, self.interfaces),
            exist_only=exist_only,
        )

    def get_symlink_info_table(self, exist_only=True):
        """Return the file info table for current files."""
        linkpath_map = self._master_interface_file_current_symlink

        def _path_maker(master, interface):
            return linkpath_map[master][interface]

        def _iter_master_interface():
            for master, d in linkpath_map.items():
                for interface in d:
                    yield master, interface

        return self._make_master_interface_path_info(
            _path_maker,
            _iter_master_interface(),
            exist_only=exist_only,
        )
