from pathlib import Path
from typing import ClassVar, Literal, get_args

import pandas as pd

from ..filestore import FileStoreBase
from .file import guess_info_from_sources

ToltecMasterType = Literal["tcs", "ics"]


class ToltecMaster:
    """Toltec master."""

    masters: ClassVar[list[ToltecMasterType]] = get_args(ToltecMasterType)


class ToltecRoachInterface:
    """TolTEC roach interface."""

    roaches: ClassVar = list(range(13))
    roach_interface: ClassVar = {roach: f"toltec{roach}" for roach in roaches}
    interface_roach: ClassVar = {v: k for k, v in roach_interface.items()}
    interfaces: ClassVar = list(roach_interface.values())


class ToltecInterface:
    """TolTEC interface."""

    interfaces: ClassVar = ToltecRoachInterface.interfaces + ["hwpr"]


class ToltecFileStore(FileStoreBase):
    """The toltec data tree."""

    def __init__(self, path="data_lmt/toltec"):
        super().__init__(path)
        self._master_subpath = {
            master: self._path.joinpath(self._master_subpath_name[master])
            for master in ToltecMaster.masters
        }
        self._master_interface_subpath = {
            master: {
                interface: self._master_subpath[master].joinpath(
                    self._interface_subpath_name[master],
                )
                for interface in ToltecInterface
            }
            for master in ToltecMaster.masters
        }
        # current file links
        self._interface_file_current_symlink = {
            interface: self.path.joinpath(symlink_name)
            for (
                interface,
                symlink_name,
            ) in self._interface_file_current_symlink_name.items()
        }
        self._master_interface_file_current_symlink = {
            master: {
                interface: self.master_interface_path[master][interface].joinpath(
                    symlink_name,
                )
                for (
                    interface,
                    symlink_name,
                ) in self._interface_file_current_symlink_name.items()
            }
            for master in ToltecMaster.masters
        }

    _master_subpath: dict[str, Path]
    _master_interface_subpath: dict[ToltecMasterType, dict[str, Path]]
    _interface_file_current_symlink: dict[str, Path]
    _master_interface_file_current_symlink: dict[ToltecMasterType, dict[str, Path]]

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

    @property
    def master_interface_path(self):
        """The path."""
        return self._master_interface_subpath

    @property
    def masters(self):
        """The list of masters."""
        return ToltecMaster.masters

    @property
    def roaches(self):
        """The list of roaches."""
        return ToltecRoachInterface.roaches

    def get_roach_path(self, master, roach):
        """Return the roach interface subpath."""
        interface = ToltecRoachInterface.roach_interface[roach]
        return self.master_interface_path[master][interface]

    def get_path_info_table(self, masters=None, exist_only=True, roach_only=False):
        """Return the path info table."""
        if masters is None:
            masters = ToltecMaster.masters
        result = []
        for master in masters:
            for interface in ToltecInterface.interfaces:
                roach = ToltecRoachInterface.interface_roach.get(interface, None)
                if roach_only and roach is None:
                    continue
                p = self.master_interface_path[master][interface]
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
        return pd.DataFrame.from_records(result)

    def get_roach_path_info_table(self, masters=None, exist_only=True):
        """Return the roach path info table."""
        return self.get_path_info_table(
            masters=masters,
            exist_only=exist_only,
            roach_only=True,
        )

    def get_current_file_info_table(self, master=None):
        """Return the file info table for current files."""
        if master is None:
            linkpaths = self._interface_file_current_symlink.values()
        else:
            linkpaths = self._master_interface_file_current_symlink[master]
        return guess_info_from_sources(linkpaths)

    def get_current_file_info(self, master=None):
        """Return the file info of the most current file."""
        return self.get_current_file_info_table(master=master).toltec_file.get_latest()
