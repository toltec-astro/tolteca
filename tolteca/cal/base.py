#! /usr/bin/env python

from urllib.parse import urlparse
from tollan.utils import fileloc
from pathlib import Path
from astropy.io.misc.yaml import load as yaml_load
from tollan.utils.schema import create_relpath_validator
from schema import Schema
# from cached_property import cached_property


class CalibBase(object):
    """Base class for calibration object.

    Calibration object (calobj) is typically static dataset that are generated
    from external procedures. This class provides a common interface to access
    such data via the notion of the "index file", which is a YAML file created
    along with the data items in a calibration object, describing the details
    of the contents.

    Parameters
    ----------
    indexfile : str, `pathlib.Path`, optional
        The path of the index file.
    index : dict, optional
        The loaded index.
    rootpath : str, `pathlib.Path`, optional
        The rootpath to use when resolving relative paths in the index.
        This has to be provided when creating from `index`.
    """

    def __init__(self, indexfile=None, index=None, rootpath=None):
        if sum([indexfile is None, index is None]) != 1:
            raise ValueError("need one of indexfile or index.")
        if indexfile is not None:
            indexfile = Path(indexfile)
            with open(indexfile, 'r') as fo:
                index = yaml_load(fo)
            if rootpath is None:
                rootpath = indexfile.parent
        elif index is not None:
            if rootpath is None:
                raise ValueError("need rootpath when index is given.")
            pass
        else:
            raise  # should not happen
        self._index = self._validate_index(index)
        self._rootpath = rootpath
        self._path_validator = create_relpath_validator(self._rootpath)

    @property
    def index(self):
        """The index of the calibration object."""
        return self._index

    @property
    def rootpath(self):
        """The rootpath to resolve paths in the index."""
        return self._rootpath

    @property
    def _index_schema(self):
        # default schema is arbitrary dict
        return Schema({object: object})

    def _validate_index(self, index):
        return self._index_schema.validate(index)

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
    def from_indexfile(cls, indexfile):
        """Create instance from index file."""
        return cls(indexfile=indexfile)

    @classmethod
    def from_index(cls, index, rootpath):
        """Create instance from index and rootpath."""
        return cls(index=index, rootpath=rootpath)

    def resolve_path(self, path, validate=True):
        """Return the absolute path by prefixing with :attr:`rootpath`.

        Parameters
        ----------
        path : str, pathlib.Path
            The path to resolve.
        validate : bool
            If true, check existence use the rootpath validator.
        """
        if validate:
            return self._path_validator(path)
        path = Path(path)
        if path.is_absolute():
            return path
        return self._rootpath / path


class CalibStack(CalibBase):
    """A class to manage multiple calibration objects.

    A calibration stack is a composite calibration object that consists of
    one or more calibration objects as its component.

    A calibration stack can be either created on the fly by composing
    multiple calibration objects, or from a index file that specifies
    the component calibration index files as a dictionary.

    """
    pass
