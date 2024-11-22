import itertools
import re
from pathlib import Path
from typing import ClassVar

import pandas as pd
from tollan.utils.fmt import pformat_yaml
from tollan.utils.general import dict_from_regex_match, resolve_symlink
from tollan.utils.log import logger

from ..filestore import FileStoreBase
from .file import guess_info_from_sources
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


class ObsSpec:
    """Utility to query LMT/TolTEC file store."""

    # TODO: make this a pydantic model

    @classmethod
    def get_raw_obs_info_table(
        cls,
        obs_spec,
        toltec_fs=None,
        lmt_fs=None,
        raise_on_multiple=False,
        raise_on_empty=False,
    ):
        """Return the raw obs info table."""
        if not isinstance(obs_spec, list):
            obs_spec = [obs_spec]
        result = []
        for s in obs_spec:
            r = cls._get_raw_obs_info_table(lmt_fs, toltec_fs, s)
            if r is None:
                continue
            if isinstance(r, list):
                r = guess_info_from_sources([resolve_symlink(f) for f in r])
            result.append(r)
        result = pd.concat(result, ignore_index=True) if result else None
        n_files = 0 if result is None else len(result)
        if n_files > 0:
            logger.debug(
                f"resovled {n_files} files from "
                f"{obs_spec=}\n{result.toltec_file.pformat()}",
            )
        if raise_on_multiple and n_files > 1:
            raise ValueError(
                f"ambiguous files found for "
                f"{obs_spec=}:\n{result.toltec_file.pformat()}",
            )
        if raise_on_empty and n_files == 0:
            raise ValueError(f"no files found for {obs_spec=}")
        return result

    @staticmethod
    def _replace_parent_path(path, parent_path, ensure_exist=False):
        path = Path(path)
        parent_path = Path(parent_path)
        parent_name = parent_path.name
        for pp in path.parents:
            if pp.name == parent_name:
                subpath = path.relative_to(pp)
                break
        else:
            return None
        path = parent_path.joinpath(subpath)
        if ensure_exist and not path.exists():
            return None
        return path

    @classmethod
    def _get_raw_obs_info_table(cls, lmt_fs, toltec_fs, obs_spec):  # noqa: PLR0911
        logger.debug(f"resovle {obs_spec=}")
        if obs_spec is None:
            if toltec_fs is None:
                logger.error("no toltec data path specified.")
                return None
            try:
                tbl = toltec_fs.get_symlink_info_table()
            except ValueError as e:
                logger.opt(exception=True).error(
                    f"error to infer current obs info: {e}",
                )
                return None
            if tbl is None:
                logger.error(
                    "unable to interfer current obs info: no symlink in data root.",
                )
                return None
            logger.debug(f"toltec symlink info in {toltec_fs.path}:\n{tbl}")
            tbl = guess_info_from_sources([resolve_symlink(p) for p in tbl["path"]])
            # get latest raw obs group
            tbl_latest = tbl.toltec_file.get_raw_obs_latest()
            logger.debug(
                f"resolved latest raw obs info from {obs_spec=}:\n"
                f"{tbl_latest.toltec_file.pformat()}",
            )
            return tbl_latest
        if re.match(r"^(.*/)?toltec.+\.nc", obs_spec):
            file = Path(obs_spec)
            if file.exists():
                logger.debug(f"resovled {file=} from {obs_spec=}")
                return [file]
            # try to locate the file under the provided root
            data_lmt_path = lmt_fs.path
            logger.debug(
                f"search file with matched subpath for {file} in {data_lmt_path}",
            )
            file = cls._replace_parent_path(file, data_lmt_path, ensure_exist=True)
            if file is None:
                logger.error(f"no matched file found for {file} in {data_lmt_path}")
                return None
            logger.debug(f"resovled {file=} from {obs_spec=}")
            return [file]
        info = dict_from_regex_match(
            r"^(?P<obsnum>\d+)"
            r"(?:-(?P<subobsnum>\d+)"
            r"(?:-(?P<scannum>\d+)(?:-(?P<roach>\d+))?)?)?",
            obs_spec,
            type_dispatcher={
                "obsnum": int,
                "subobsnum": int,
                "scannum": int,
                "roach": int,
            },
        )
        if info is None:
            logger.error(f"unable to resolve {obs_spec=} by regex match")
            return None
        obsnum = info["obsnum"]
        subobsnum = info["subobsnum"]
        scannum = info["scannum"]
        roach = info["roach"]
        logger.debug(
            f"resolved {obsnum=} {subobsnum=} {scannum=} {roach=} "
            f"from {obs_spec=} by regex match",
        )
        data_lmt_path = lmt_fs.path
        p_interface = "toltec*" if roach is None else f"toltec{roach}"
        p_obsnum = f"_{obsnum:06d}"
        p_subobsnum = "_*" if subobsnum is None else f"_{subobsnum:03d}"
        p_scannum = "_*" if scannum is None else f"_{scannum:04d}"
        p = f"{p_interface}/{p_interface}{p_obsnum}{p_subobsnum}{p_scannum}_*.nc"
        glob_patterns = [
            f"toltec/ics/{p}",
            f"toltec/tcs/{p}",
        ]
        logger.debug(
            f"search file patterns in {data_lmt_path=}:\n"
            f"{pformat_yaml(glob_patterns)}",
        )
        files = list(itertools.chain(*(data_lmt_path.glob(p) for p in glob_patterns)))
        if not files:
            logger.error(f"no files found for {obs_spec=} in {data_lmt_path=}")
            return None
        return files
