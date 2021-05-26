#! /usr/bin/env python

from urllib.parse import urlparse
from tollan.utils import fileloc
from pathlib import Path
from astropy.io.misc.yaml import load as yaml_load
from tollan.utils.schema import create_relpath_validator
from schema import Schema
from cached_property import cached_property


class CalibBase(object):
    """Base class for calibration object.

    Calibration object (calobj) are typically static data that are generated
    from external procedures. This class defines a scheme to access such data
    via the notion of the "index file", which is created for each dataset and
    describes all the details of the data items.

    Parameters
    ----------
    index_filepath : str, `pathlib.Path`

        The path of the index file.
    """

    def __init__(self, index_filepath):
        index_filepath = Path(index_filepath)
        with open(index_filepath, 'r') as fo:
            index = yaml_load(fo)
        self._index = self.validate_index(index)
        self._rootpath = index_filepath.parent
        self._path_validator = create_relpath_validator(self._rootpath)

    @property
    def index(self):
        """The index of the calibration object."""
        return self._index

    @classmethod
    def validate_index(cls, index):
        """The schema that validates the index.

        The default is to not validate.
        """
        return index

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


class CalibStack(CalibBase):
    """A class to manage multiple calibration objects.

    A calibration stack is a composite calibration object that consists of
    one or more calibration objects as its component.

    A calibration stack can be either created on the fly by composing
    multiple calibration objects, or from a index file that specifies
    the component calibration index files as a dictionary.

    Parameters
    ----------
    index_filepath : str or `pathlib.Path`, optional
        The index file path that defines the calibration stack.

    index : dict, optional
        The index dict that defines the calibration stack.
    """

    def __init__(self, index_filepath=None, index=None):
        if sum([index_filepath is None, index is None]) != 1:
            raise ValueError("one of index_filepath or index has to be set.")
        if index_filepath is not None:
            super().__init__(index_filepath)
        elif index is not None:
            self._index = self._validate_index(index)
            self._rootpath = None
        else:
            raise  # should not happen

    def resolve_path(self, path):
        if self._rootpath is None:
            raise NotImplementedError(
                    'unable to resolve path: no rootpath')
        return super().resolve_path(path)

    @staticmethod
    def _validate_index(index):
        return Schema({
            str: CalibBase,
            }).validate(index)
