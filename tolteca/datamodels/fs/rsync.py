#! /usr/bin/env python

from .base import FileStoreAccessor
from tollan.utils.log import get_logger
from pathlib import Path
import subprocess
import tempfile
import re
from packaging.version import parse as parse_version
from tollan.utils import FileLoc, fileloc, call_subprocess_with_live_output
from collections import defaultdict


__all__ = ['RsyncAccessor', ]


class RsyncAccessor(FileStoreAccessor):
    """This class provides access to remote files via rsync."""

    logger = get_logger()

    @staticmethod
    def _get_rsync_cmd():
        """Get the rsync executable."""
        rsync_cmd = 'rsync'
        try:
            output = subprocess.check_output(
                    (rsync_cmd, '--version'),
                    stderr=subprocess.STDOUT
                    ).decode().split('\n')[0].strip()
            version = re.match(
                    r'rsync\s+version\s+(?P<version>\d+\.\d+(?:\.[^\s]+)?)',
                    output).groupdict()['version']
        except Exception as e:
            raise RuntimeError(f"error in checking rsync version: {e}")
        oldest_supported_version = '3.1.0'
        if parse_version(version) < parse_version(oldest_supported_version):
            raise RuntimeError(
                    f"rsync version has to be >= {oldest_supported_version}"
                    f", found {version}")
        return rsync_cmd

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    @staticmethod
    def _filter_rsync_output(line):
        return 'skipping non-regular file' not in line

    @staticmethod
    def _decode_remote_path(path):
        d = re.match(r'(?P<host>[^:]+):(?P<path>.*)', path).groupdict()
        return d['host'], d['path']

    @staticmethod
    def _encode_remote_path(host, path):
        if host != '':
            return f'{host}:{path}'
        return f'{path}'

    @classmethod
    def rsync(cls, filepaths, dest, path_filter=None):
        """Rsync the list of files to dest.

        Parameters
        ----------
        filepath : list of str
            The paths to remote files, in scp specifier format.
        dest : str
            The destination.
        path_filter : callable
            The filter to apply to the filepaths.
        """
        logger = get_logger()
        # group the paths by host, and create filelist
        paths_per_host = defaultdict(list)
        for p in filepaths:
            if isinstance(p, str):
                p = fileloc(p)
            if isinstance(p, FileLoc):
                h = p.netloc
                p = p.path.as_posix()
            elif isinstance(p, Path):
                h = ''
                p = p.as_posix()
            else:
                raise ValueError(f"invalid file path {p}")
            if path_filter is not None:
                p = path_filter(p)
            paths_per_host[h].append(p)
        result = set()
        dest = Path(dest)
        for host, paths in paths_per_host.items():
            with tempfile.NamedTemporaryFile('w') as fo:
                for p in paths:
                    fo.write(f"{p}\n")
                fo.flush()
                # cmd_stats = [
                #         'rsync', '--stats', '--dry-run',
                #         '--files-from', fo.name,
                #         src, dest.as_posix()
                #         ]
                cmd = [
                        cls._get_rsync_cmd(), '-avhPR', '--append-verify',
                        '--files-from',
                        fo.name, f'{host}:/' if host is not '' else '/', dest.as_posix()
                        ]
                # get dry run stats
                # subprocess.check_output(cmd_stats)
                logger.debug("rsync with cmd: {}".format(' '.join(cmd)))
                call_subprocess_with_live_output(cmd)
            for p in paths:
                p = dest.joinpath(p.lstrip('/'))
                print(p)
                if p.exists():
                    result.add(p)
        return result

    @classmethod
    def glob(cls, *args):
        """Return a set of remote paths.

        Parameters
        ----------
        args : list of str or `~tollan.utils.FileLoc`
            The remote paths to query.
        """
        logger = get_logger()
        result = set()
        with tempfile.TemporaryDirectory() as tmp:
            paths = list()
            for path in args:
                path = fileloc(path)
                hostname = path.netloc
                path = cls._encode_remote_path(
                        hostname, path.path.as_posix())
                paths.append(path)

            def map_rsync_output(line):
                if hostname == '':
                    return f'{line}'
                return cls._encode_remote_path(hostname, f'/{line}')

            # get file list
            rsync_cmd = [
                    cls._get_rsync_cmd(),
                    '-rn', '--info=name', '--relative', ] + paths + [tmp, ]
            logger.debug("rsync with cmd: {}".format(' '.join(rsync_cmd)))
            # this is used to get rid of parent dirs
            # _path = map_rsync_output(_path.path.as_posix())
            # if not _path.endswith('/'):
            #     _path += '/'
            for p in map(
                    map_rsync_output,
                    filter(
                        cls._filter_rsync_output,
                        subprocess.check_output(
                            rsync_cmd).decode().strip().split('\n'))):
                # if _path.startswith(p):
                # get rid of directories
                if p.endswith('/'):
                    continue
                result.add(p)
        return result
