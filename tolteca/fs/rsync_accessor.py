#! /usr/bin/env python

"""Access files via rsync."""

from . import Accessor
from tollan.utils.log import get_logger
from pathlib import Path
import subprocess
import tempfile
import re
import sys
import functools
from io import TextIOWrapper
from packaging.version import parse as parse_version


__all__ = ['RsyncAccessor', ]


class RsyncAccessor(Accessor):
    """This class provide access to remote files via rsync."""

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
        return f'{host}:{path}'

    @classmethod
    def rsync(cls, filepaths, dest):
        """Rsync the list of files to dest.

        Parameters
        ----------
        filepath: list of str
            The paths to remote files, in scp specifier format.
        dest: str
            The destination of the operation.
        """
        logger = get_logger()
        # group the paths by host, and create filelist
        paths_per_src = {}
        for p in filepaths:
            if isinstance(p, Path):
                p = p.as_posix()
            # here we need to split path so that the hostnames
            # contains the leading slash as host:/
            d = re.match(r'(?P<src>[^:]+:\/)(?P<path>.*)', p).groupdict()
            src = d['src']
            p = d['path']
            if src in paths_per_src:
                paths_per_src[src].append(p)
            else:
                paths_per_src[src] = [p, ]
        result = set()
        dest = Path(dest)
        for src, paths in paths_per_src.items():
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
                        cls._get_rsync_cmd(), '-avhP',
                        '--files-from',
                        fo.name, src, dest.as_posix()
                        ]
                # get dry run stats
                # subprocess.check_output(cmd_stats)
                logger.debug("rsync with cmd: {}".format(' '.join(cmd)))
                with subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        # stderr=subprocess.STDOUT,
                        bufsize=1,
                        ) as proc:
                    reader = TextIOWrapper(proc.stdout, newline='')
                    for char in iter(
                            functools.partial(reader.read, 1), b''):
                        # logger.debug(ln.decode().strip())
                        sys.stderr.write(char)
                        if proc.poll() is not None:
                            sys.stderr.write('\n')
                            break
            for p in paths:
                p = dest.joinpath(p)
                if p.exists():
                    result.add(p)
        return result

    @classmethod
    def glob(cls, *args):
        """Return a set of remote paths.

        Parameters
        ----------
        *args: list of str
            The remote paths to query. Each path shall be an valid scp remote
            path specifier.
        """
        result = set()
        with tempfile.TemporaryDirectory() as tmp:
            for path in args:
                hostname, _ = cls._decode_remote_path(path)

                def _map_rsync_output(line):
                    return cls._encode_remote_path(hostname, f'/{line}')

                # get file list
                rsync_cmd = [
                        cls._get_rsync_cmd(),
                        '-rn', '--info=name', '--relative',
                        path, tmp]
                for p in map(
                    _map_rsync_output,
                    filter(
                        cls._filter_rsync_output,
                        subprocess.check_output(
                            rsync_cmd).decode().strip().split('\n'))):
                    result.add(p)
        return result
