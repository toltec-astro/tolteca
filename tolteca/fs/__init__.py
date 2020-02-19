#!/usr/bin/env python
from tollan.utils.log import get_logger
from pathlib import Path


class DataFileStore(object):
    """A base class that represent a path that stores data files."""

    def __init__(self, rootpath=None):
        self.rootpath = rootpath

    @property
    def rootpath(self):
        return self._rootpath

    @rootpath.setter
    def rootpath(self, path):
        self._rootpath = self._normalize_path(path)

    @staticmethod
    def _normalize_path(p):
        logger = get_logger()
        try:
            return Path(p).expanduser().absolute()
        except Exception:
            logger.error(f"unable to expand user for path {p}")
            return Path(p).absolute()
