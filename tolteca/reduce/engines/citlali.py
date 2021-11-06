#!/usr/bin/env python

import functools
import subprocess
import re
import shutil
import os
import git
import pathlib
from tollan.utils.log import get_logger
from tollan.utils import ensure_abspath
from tollan.utils.fmt import pformat_yaml
from .base import PipelineEngine, PipelineEngineError
from ...utils import get_user_data_dir


REMOTE_CITLALI_REPO_URL = 'https://github.com/toltec-astro/citlali.git'
LOCAL_CITLALI_REPO_PATH = get_user_data_dir().joinpath("engines/citlali")


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
            repo.remote(name='origin').fetch()
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

    @staticmethod
    def get_version(path):
        """Get the version of the Citlali."""
        logger = get_logger()
        output = subprocess.check_output(
                (path, '--version'),
                stderr=subprocess.STDOUT,
                ).decode()
        logger.debug(f'check version of {path}:\n{output}')
        r = re.compile(
            r'^(?P<name>citlali\s)?(?P<version>.+)\s\((?P<timestamp>.+)\)$',
            re.MULTILINE)
        m0 = re.search(r, output)
        # import pdb
        # pdb.set_trace()
        if m0 is None:
            logger.warning(
                f"unable to parse citlali version: \n{output}")
            version = UNKNOWN_VERSION
        else:
            m = m0.groupdict()
            version = m['version']
        return version

    def check_version(self, version):
        repo = _get_local_citlali_repo()
        short_hash = repo.git.rev_parse(short=True)
        print(repo)
        print(short_hash)
        return

    def check_for_update(self):
        """Check the current Citlali version agains the Github remote head.

        """
        logger = get_logger()
        repo = _get_local_citlali_repo()

        def get_git_changelog(v1, v2):

            def norm_rev(v):
                return v.replace('-dirty', '')

            v1, v2 = map(norm_rev, [v1, v2])
            changelog = repo.git.log(f'{v1}..{v2}', oneline=True)
            if changelog == '':
                # same revision
                changelog = None
            return changelog

        # get remote head version
        remote_version = repo.git.rev_parse('origin/HEAD', short=True)
        changelog = get_git_changelog(self.version, remote_version)

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
            changelog_section = ''
        logger.info(
            f"\n"
            f"* Remote Citlali version: {remote_version}\n"
            f"* Local Citlali version: {self.version}\n"
            f'{changelog_section}'
            )


@functools.lru_cache(maxsize=None)
def _get_citlali_exec(path):
    # This caches all the found exec.
    return CitlaliExec(path=path)


class Citlali(PipelineEngine):
    """A class to run Citlali, the TolTEC data reduction pipeline engine.

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
                f"{citlali_executables['path', 'version', 'timestamp']}"
                )
        else:
            pass
        self._citlali_exec = citlali_executables[0]

    @property
    def exec_path(self):
        return self._citlali_exec.path

    @property
    def version(self):
        return self._citlali_exec.version

    def check_for_update(self):
        return self._citlali_exec.check_for_update()

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
            if p.is_dir():
                _check_and_add_path(shutil.which('citlali', path=p))
            elif p.is_file() and os.access(p, os.X_OK):
                _check_and_add_path(p)
            else:
                cls.logger.debug(f'skip invalid path {p}')
        cls.logger.debug(
            f"found {len(paths)} citlali executables:\n{pformat_yaml(paths)}")
        return list(paths.values())
