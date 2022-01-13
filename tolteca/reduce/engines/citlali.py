#!/usr/bin/env python

import functools
import subprocess
import re
import shutil
import os
import git
import pathlib

from copy import deepcopy
from cached_property import cached_property
from packaging.version import Version, InvalidVersion
from packaging.specifiers import SpecifierSet
from astropy.table import Table
from scalpl import Cut
import numpy as np
import astropy.units as u
from pathlib import Path
from schema import Or
from dataclasses import dataclass, field, replace
from typing import Union
from tollan.utils.dataclass_schema import add_schema
from tollan.utils import rupdate
from tollan.utils import call_subprocess_with_live_output
from tollan.utils.log import get_logger, timeit
from tollan.utils import ensure_abspath
from tollan.utils.fmt import pformat_yaml

from .base import PipelineEngine, PipelineEngineError
from ...utils import get_user_data_dir
from ...utils.misc import get_nested_keys
from ...utils.common_schema import RelPathSchema, PhysicalTypeSchema
from ...utils.runtime_context import yaml_load


REMOTE_CITLALI_REPO_URL = 'https://github.com/toltec-astro/citlali.git'
LOCAL_CITLALI_REPO_PATH = get_user_data_dir().joinpath("engines/citlali")
GIT_REMOTE_TIMEOUT = 10.


@functools.lru_cache(maxsize=1)
def _get_local_citlali_repo():
    """Return the Citlali git repo made available in the user data directory.

    This repo is maintained by this package for check commit histories and
    comparing versions.
    """
    logger = get_logger()

    if LOCAL_CITLALI_REPO_PATH.exists():
        repo = git.Repo(LOCAL_CITLALI_REPO_PATH)
        try:
            repo.remote(name='origin').fetch(
                kill_after_timeout=GIT_REMOTE_TIMEOUT)
            return repo
        except Exception:
            logger.debug(
                f"error fetching local citlali repo in "
                f"{LOCAL_CITLALI_REPO_PATH}")
        return repo

    logger.debug(f"setup local citlali repo in ${LOCAL_CITLALI_REPO_PATH}")

    # class _Progress(git.remote.RemoteProgress):
    #     def update(self, op_code, cur_count, max_count=None, message=''):
    #         logger.debug(self._cur_line)

    repo = git.Repo.clone_from(
        REMOTE_CITLALI_REPO_URL,
        LOCAL_CITLALI_REPO_PATH,
        # progress=_Progress()
        )
    return repo


UNKNOWN_VERSION = 'unknown'
"""Value used when the version is unknown."""


class CitlaliExec(object):
    """A low level wrapper class to run Citlali executable."""

    logger = get_logger()

    def __init__(self, path):
        path = self._path = ensure_abspath(path)
        version = self._version = self.get_version(path)
        if version is UNKNOWN_VERSION:
            raise ValueError(f"invalid Citlali executable path {path}")

    def __repr__(self):
        return f'{self.__class__.__name__}(version={self.version})'

    @property
    def path(self):
        """The path of the Citlali executable."""
        return self._path

    @property
    def version(self):
        """The version of the Citlali executable."""
        return self._version

    @cached_property
    def semver(self):
        """The semantic version of the Citlali executable."""
        return self._ver_to_semver(self._version)

    @classmethod
    def get_version(cls, path):
        """Get the version of the Citlali."""
        output = subprocess.check_output(
                (path, '--version'),
                stderr=subprocess.STDOUT,
                ).decode()
        cls.logger.debug(f'check version of {path}:\n{output}')
        r = re.compile(
            r'^(?P<name>citlali\s)?(?P<version>.+)\s\((?P<timestamp>.+)\)$',
            re.MULTILINE)
        m0 = re.search(r, output)
        # import pdb
        # pdb.set_trace()
        if m0 is None:
            cls.logger.warning(
                f"unable to parse citlali version: \n{output}")
            version = UNKNOWN_VERSION
        else:
            m = m0.groupdict()
            version = m['version']
        return cls._norm_ver(version)

    def get_default_config(self):
        """Get the default config of the Citlali."""
        path = self.path
        output = subprocess.check_output(
                (path, '--dump_config'),
                stderr=subprocess.STDOUT,
                ).decode()
        self.logger.debug(f'dump config of {path}:\n{output}')
        return yaml_load(output)

    @staticmethod
    def _norm_ver(ver, with_rev=True):
        # removes any non-standard suffix
        if with_rev:
            return re.sub(r'(-dirty)$', '', ver)
        return re.sub(r'(-dirty|~\d+|-\d+-.+)$', '', ver)

    @classmethod
    def _ver_to_semver(cls, ver):
        """Convert version string (tag, rev hash, etc.) to SemVer.

        This is done by querying the local Citlali repo history.
        """
        try:
            return Version(ver)
        except InvalidVersion:
            pass
        # try find the latest version tag in the history
        repo = _get_local_citlali_repo()
        try:
            _ver = cls._norm_ver(repo.git.describe(ver, contains=True), with_rev=False)
        except Exception:
            _ver = cls._norm_ver(repo.git.describe(ver, contains=False), with_rev=False)
        cls.logger.debug(f"version {ver} -> semver {_ver}")
        return Version(_ver)

    def check_version(self, version):
        verspec = SpecifierSet(version)
        self.logger.debug(f"check {self.semver} against {verspec}")
        return self.semver in verspec

    def check_for_update(self):
        """Check the current Citlali version agains the Github remote head.

        """
        logger = get_logger()
        repo = _get_local_citlali_repo()

        def get_git_changelog(v1, v2):

            changelog = repo.git.log(f'{v1}..{v2}', oneline=True)
            if changelog == '':
                # same revision
                changelog = None
            return changelog

        # get remote head version
        remote_rev = repo.git.rev_parse('origin/HEAD', short=True)
        changelog = get_git_changelog(self.version, remote_rev)

        if changelog is not None:
            changelog_section = (
                f"\n"
                f"###################### Attention #######################\n"
                f"You are using an outdated version of Citlali. The latest\n"
                f"version has the following changes:\n\n"
                f"{changelog}\n"
                f"########################################################"
                f"\n"
                )
        else:
            changelog_section = 'Citlali is update-to-date!'
        logger.info(
            f"\n\n"
            f"* Executable path: {self.path}"
            f"* Remote Citlali rev: {remote_rev}\n"
            f"* Local Citlali version: {self.version}\n"
            f'{changelog_section}'
            )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _get_line_buf_cmd():
        stdbuf = shutil.which('stdbuf')
        if stdbuf is not None:
            return [stdbuf, '-oL']
        return list()

    def run(self, config_file, log_level="INFO", **kwargs):
        exec_path = self.path
        citlali_cmd = [
                exec_path.as_posix(),
                '-l', log_level.lower(),
                config_file.as_posix(),
                ]
        cmd = self._get_line_buf_cmd() + citlali_cmd
        self.logger.info(
            "run {} cmd: {}".format(self, ' '.join(citlali_cmd)))
        return call_subprocess_with_live_output(cmd, **kwargs)


@functools.lru_cache(maxsize=None)
def _get_citlali_exec(path):
    # This caches all the found exec.
    return CitlaliExec(path=path)


class Citlali(PipelineEngine):
    """A wrapper class of Citlali, the TolTEC data reduction pipeline
    engine.

    It searches for instances of ``citlali`` executables and check their
    versions against the required version. The latest one is adopted if
    multiple are found.

    Parameters
    ----------
    path : str, `pathlib.Path`, list
        The path to search the executable for. A list of paths is accepted
        which searches the executable in the list in order.

    version : str
        A version specifier/predicate that specifies the required version of
        pipeline.

    use_env_path : bool
        If True, the system PATH env var is consulted.
    """

    logger = get_logger()

    def __init__(
            self, path=None, version=None, use_env_path=True):
        citlali_executables = self.find_citlali_executables(
            path=path, version=version, use_env_path=use_env_path)
        if len(citlali_executables) == 0:
            raise PipelineEngineError(
                f"Cannot find Citlali executables for "
                f"version {version}")
        elif len(citlali_executables) > 1:
            self.logger.warning(
                f"Found multiple Citlali executables for "
                f"version={version}\n"
                # f"{citlali_executables['path', 'version', 'timestamp']}"
                )
        else:
            pass
        citlali_exec = self._citlali_exec = citlali_executables[0]
        self.logger.debug(f"use citlali executable: {citlali_exec}")

    def __repr__(self):
        return f'{self.__class__.__name__}(version={self.version})'

    @property
    def exec_path(self):
        return self._citlali_exec.path

    @property
    def version(self):
        return self._citlali_exec.version

    def get_default_config(self):
        return self._citlali_exec.get_default_config()

    def check_for_update(self):
        return self._citlali_exec.check_for_update()

    def run(self, *args, **kwargs):
        return self._citlali_exec.run(*args, **kwargs)

    @classmethod
    def find_citlali_executables(
            cls, path=None, version=None, use_env_path=True):
        """Return a list of `CitlaliExec` objects that satisfy the version
        constraints."""
        exec_name = 'citlali'

        if path is None:
            path = []
        elif isinstance(path, (str, pathlib.Path)):
            path = [path]
        else:
            pass
        if use_env_path:
            # check if the executable is in env path
            exec_path = shutil.which(exec_name, path=None)
            if exec_path is not None:
                path.append(exec_path)

        # now go over each item in path and get executable for checking
        paths = dict()

        def _check_and_add_path(exec_path):
            try:
                exec = CitlaliExec(path=exec_path)
                paths[exec_path] = exec
            except Exception:
                cls.logger.debug(
                    f"skip invalid cialali exec {exec_path}")

        cls.logger.debug(
            f"find citali executables in paths:\n{pformat_yaml(path)}")
        for p in path:
            p = ensure_abspath(p)
            if p.is_dir():
                _check_and_add_path(shutil.which('citlali', path=p))
            elif p.is_file() and os.access(p, os.X_OK):
                _check_and_add_path(p)
            else:
                cls.logger.debug(f'skip invalid path {p}')
        cls.logger.debug(
            f"found {len(paths)} citlali executables:\n{pformat_yaml(paths)}")
        # now check each executable with version
        if version is None:
            cls.logger.debug("skip checking executable versions")
            return list(paths.values())
        valid_execs = list()
        cls.logger.debug("check executable versions")
        for e in paths.values():
            if e.check_version(version):
                valid_execs.append(e)
                cls.logger.debug(f"{e} version satisfies {version}")
            else:
                cls.logger.debug(f"{e} version does not satisfies {version}")
        return valid_execs

    @timeit
    def proc_context(self, config):
        """Return a `CitlaliProc` that run reduction for given input dataset.
        """
        return CitlaliProc(citlali=self, config=config)


class CitlaliProc(object):
    """A context class for running Citlali."""

    logger = get_logger()

    def __init__(self, citlali, config):
        self._citlali = citlali
        self._config = replace(
            config,
            low_level=self._resolve_low_level_config(config.low_level)
            )

    @property
    def citlali(self):
        return self._citlali

    @property
    def config(self):
        return self._config

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(
            self, dataset, output_dir,
            log_level='INFO', logger_func=None):
        # resolve the dataset to input items
        tbl = dataset.index_table
        grouped = tbl.group_by(
            ['obsnum', 'subobsnum', 'scannum', 'master', 'repeat'])
        self.logger.debug(f"collected {len(grouped)} raw obs")
        for key, group in zip(grouped.groups.keys, grouped.groups):
            self.logger.debug(
                    '****** obs name={obsnum}_{subobsnum}_{scannum} '
                    '*******'.format(**key))
            self.logger.debug(f'{group}\n')
        input_items = [
            self._resolve_input_item(d) for d in grouped.groups]
        # create low level config object and dump to file
        # the low level has been resolved to a dict when __init__ is called
        cfg = deepcopy(self.config.low_level)
        rupdate(cfg, {
            'inputs': input_items,
            'runtime': {
                # TODO get rid of the trailing slash
                'output_dir': output_dir.as_posix() + '/'
                }
            })
        cfg_hl = self._resolve_high_level_config(self.config)
        rupdate(cfg, cfg_hl)
        self.logger.debug(
                f'resolved low level config:\n{pformat_yaml(cfg)}')
        # show the high level config entries that over writes the low level
        # values
        chl = Cut(cfg_hl)
        cll = Cut(self.config.low_level)
        updated_entries = []
        for key in get_nested_keys(cfg_hl):
            if key not in cll:
                continue
            new = chl[key]
            old = cll[key]
            if new == old:
                continue
            updated_entries.append((key, old, new))
        if updated_entries:
            updated_entries = Table(
                rows=updated_entries,
                names=['low_level_config_key', 'default', 'updated'])
            self.logger.info(
                f"low level config entries overwitten by high level config:\n\n"
                f"{updated_entries}\n")
        name = input_items[0]['meta']['name']
        output_name = f'citlali_o{name}_c{len(input_items)}.yaml'
        cfg_filepath = output_dir.joinpath(output_name)
        with open(cfg_filepath, 'w') as fo:
            fo.write(pformat_yaml(cfg))
            # yaml_dump(cfg, fo)
        success = self._citlali.run(
            cfg_filepath, log_level=log_level, logger_func=logger_func)
        # TODO implement the logic to locate the generated output files
        # which will be used to create data prod object.
        if success:
            return output_dir
        raise RuntimeError(
            f"failed to run {self.citlali} with config file {cfg_filepath}")

    def _resolve_low_level_config(self, low_level):
        """Return a low-level config dict from low_level config entry."""
        if low_level is None:
            return self.citlali.get_default_config()
        if isinstance(low_level, Path):
            with open(low_level, 'r') as fo:
                return yaml_load(fo)
        # a dict already
        return low_level

    def _resolve_high_level_config(self, high_level):
        """Return a low level config dict from high level dict."""
        # image_frame_params
        cfg = Cut(dict())
        pixel_size = high_level.image_frame_params.pixel_size
        if pixel_size is not None:
            cfg.setdefault(
                'mapmaking.pixel_size_arcsec', pixel_size.to_value(u.arcsec))
        return cfg.data

    @classmethod
    def _resolve_input_item(cls, index_table):
        """Return an citlali input list entry from index table."""
        tbl = index_table
        d0 = tbl[0]
        meta = {
            'name': f'{d0["obsnum"]}_{d0["subobsnum"]}_{d0["scannum"]}'
            }
        data_items = list()
        cal_items = list()
        for entry in tbl:
            instru = entry['instru']
            interface = entry['interface']
            source = entry['source']
            extra = dict()
            if instru == 'toltec':
                c = data_items
            elif interface == 'lmt':
                c = data_items
            elif interface == 'apt':
                c = cal_items
                # TODO implement in citlali the proper
                # ecsv handling
                source = _fix_apt(source)
                extra = {'type': 'array_prop_table'}
            else:
                continue
            c.append(dict({
                    'filepath': source,
                    'meta': {
                        'interface': interface
                        }
                    }, **extra)
                    )
        cls.logger.debug(
                f"collected input item name={meta['name']} "
                f"n_data_items={len(data_items)} "
                f"n_cal_items={len(cal_items)}")
        # this is a hack. TODO fix the proper ordering of data items
        data_items = sorted(
                data_items,
                key=lambda d: (
                    int(d['meta']['interface'][6:])
                    if d['meta']['interface'].startswith('toltec')
                    else -1)
                )
        return {
                'meta': meta,
                'data_items': data_items,
                'cal_items': cal_items,
                }


def _fix_apt(source):
    # this is a temporary fix to make citlali work with the
    # apt
    tbl = Table.read(source, format='ascii.ecsv')
    tbl_new = Table()
    tbl_new['nw'] = np.array(tbl['nw'], dtype='d')
    tbl_new['array'] = np.array(tbl['array'], dtype='d')
    tbl_new['flxscale'] = np.array(tbl['flxscale'], dtype='d')
    tbl_new['x_t'] = tbl['x_t'].quantity.to_value(u.deg)
    tbl_new['y_t'] = tbl['y_t'].quantity.to_value(u.deg)
    tbl_new['a_fwhm'] = tbl['a_fwhm'].quantity.to_value(u.deg)
    tbl_new['b_fwhm'] = tbl['b_fwhm'].quantity.to_value(u.deg)
    source_new = source.replace('.ecsv', '_trimmed.ecsv')
    tbl_new.write(source_new, format='ascii.ecsv', overwrite=True)
    return source_new


# High level config classes
# TODO some of these config can be made more generic and does not needs to
# be citlali specific.


@add_schema
@dataclass
class ImageFrameParams(object):
    """Params related to 2D image data shape and WCS."""
    pixel_size: u.Quantity = field(
        default=None,
        metadata={
            'description': 'The pixel size of image.',
            'schema': PhysicalTypeSchema('angle')
            }
        )


@add_schema
@dataclass
class CitlaliConfig(object):
    """The high-level config for Citlali."""

    low_level: Union[None, Path, dict] = field(
        default=None,
        metadata={
            'description': 'The low level config used as the base.',
            'schema': Or(RelPathSchema(), dict)
            }
        )
    image_frame_params: ImageFrameParams = field(
        default_factory=ImageFrameParams,
        metadata={
            'description':
            'The params related to the output image data shape and WCS.'
            }
        )

    class Meta:
        schema = {
            'ignore_extra_keys': False,
            'description': 'The high level config for Citlali.'
            }
