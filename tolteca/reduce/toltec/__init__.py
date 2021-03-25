#! /usr/bin/env python

from packaging.specifiers import SpecifierSet
from packaging.version import parse as parse_version
from pathlib import Path
import subprocess
import re
from tollan.utils.log import get_logger


class Citlali(object):
    """A class to run citlali, the TolTEC data reduction engine.

    Parameters
    -----------
    binpath : str or `~pathlib.Path`, optional
        The path to the citlali executable.
    version_specifiers : str, optional
        If set, the citlali version is checked against these specifiers.
    calobj : `~tolteca.cal.ToltecCalib`, optional
        The calibration object to use.
    """

    def __init__(
            self, binpath=None, version_specifiers=None,
            calobj=None):
        self._version_specifiers = version_specifiers
        self._binpath = None if binpath is None else Path(binpath)
        self._citlali_cmd = self._get_citlali_cmd(
                binpath=self._binpath,
                version_specifiers=self._version_specifiers)
        self._calobj = calobj

    @staticmethod
    def _get_citlali_cmd(binpath=None, version_specifiers=None):
        """Get the citlali executable."""
        logger = get_logger()
        if binpath is not None:
            if binpath.is_dir():
                binpath = binpath.joinpath('citlali')
            citlali_cmd = Path(binpath).as_posix()
        else:
            citlali_cmd = 'citlali'
        try:
            output = subprocess.check_output(
                    (citlali_cmd, '--version'),
                    stderr=subprocess.STDOUT
                    ).decode().split('\n')[0].strip()
            version = re.match(
                    r'(?P<version>\d+\.\d+(?:\.[^\s]+)?)',
                    output).groupdict()['version']
        except Exception as e:
            # raise RuntimeError(f"unable to get citlali version: {e}")
            logger.debug(f"unable to get citlali version: {e}")
            # pass
        if version_specifiers is not None and \
                parse_version(version) not in SpecifierSet(version_specifiers):
            raise RuntimeError(
                    f"citlali version does not satisfy {version_specifiers}"
                    f", found {version}")
        return citlali_cmd

    def __repr__(self):
        return f'{self.__class__.__name__}()'
