#! /usr/bin/env python

from urllib.parse import urlparse
from tollan.utils import fileloc
from pathlib import Path
from astropy.io.misc.yaml import load as yaml_load
from tollan.utils.schema import create_relpath_validator


class CalibBase(object):
    """Base class for calibration data.

    The calibration data is typically static data that are generated from
    external procedures. This class defines a scheme to access such data via
    the notion of the "index file", which is created for each calibration
    dataset and describes all the details of the data items.

    Parameters
    ----------
    index_filepath : str, `pathlib.Path`

        The path of the index file.
    """

    def __init__(self, index_filepath):
        index_filepath = Path(index_filepath)
        with open(index_filepath, 'r') as fo:
            index = yaml_load(fo)
        self._index = index
        self._rootpath = index_filepath.parent
        self._path_validator = create_relpath_validator(self._rootpath)

    @property
    def index(self):
        """The index of the calibration object."""
        return self._index

    @classmethod
    def from_uri(cls, uri, **kwargs):
        """Create instance from URI."""

        u = urlparse(uri)

        m = getattr(cls, f"from_{u.scheme}", None)
        if m is None:
            raise ValueError("scheme {u.scheme} is not supported.")
        dispatch_uri = {
                'file': lambda v: fileloc(v).path
                }.get(u.scheme, lambda v: v)
        return m(dispatch_uri(uri), **kwargs)

    @classmethod
    def from_indexfile(cls, index_filepath):
        """Create instance from index filepath."""
        return cls(index_filepath)

    def resolve_path(self, path):
        """Return the absolute path by prefixing with :attr:`rootpath`."""
        return self._path_validator(path)
