#!/usr/bin/env python
from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml
from pathlib import Path
from .ssh_accessor import SSHAccessor
import stat
from tollan.utils.sys import touch_file


class Accessor(object):
    """A common interface for accessing resources."""

    def query(self, *args, **kwargs):
        """Return a table info object."""
        return NotImplemented

    def glob(self, *args, **kwargs):
        """Return a list of file paths."""
        return NotImplemented


class DataFileStore(object):
    """A base class that represent a path that stores data files."""

    _rootpath = None
    _accessor = None

    def __init__(self, rootpath):
        self.rootpath = rootpath

    @property
    def accessor(self):
        return self._accessor

    @property
    def rootpath(self):
        return self._rootpath

    @rootpath.setter
    def rootpath(self, path):
        self._rootpath = self._normalize_path(path)

    @staticmethod
    def _normalize_path(p):
        logger = get_logger()
        if ':' in str(p):
            # remote path
            return Path(p)
        try:
            return Path(p).expanduser().absolute()
        except Exception:
            logger.error(f"unable to expand user for path {p}")
            return Path(p).absolute()

    def glob(self, *patterns):
        result = set()
        for p in patterns:
            for f in self.accessor.glob(self.rootpath, p):
                result.add(f)
        return list(result)


class RemoteDataFileStore(DataFileStore):
    """A class that works with remote data files."""

    logger = get_logger()

    def __init__(self, uri, **kwargs):
        self._uri = uri
        self._protocol, rest = uri.split("://", 1)
        self._hostname, rootpath = rest.split(":", 1)
        super().__init__(rootpath=rootpath)
        self.logger.debug(f"config: {pformat_yaml(kwargs)}")
        dispatch_protocol = {
                'ssh': (
                    SSHAccessor,
                    dict(**kwargs)
                    )
                }
        if self._protocol in dispatch_protocol:
            accessor_cls, kwargs = dispatch_protocol[self._protocol]
            self._accessor = accessor_cls(**kwargs)
            self.logger.debug(f"accessor: {self._accessor}")
        else:
            raise RuntimeError(
                    "uknown remote access protocol {self._protocol}")

    def __repr__(self):
        return f'{self.__class__.__name__}(uri={self._uri}, ' \
            f'accessor={self.accessor})'

    @property
    def uri(self):
        return self._uri

    @property
    def hostname(self):
        return self._hostname

    @DataFileStore.rootpath.setter
    def rootpath(self, path):
        self._rootpath = Path(path).absolute()

    def check_sync(self, dest, **kwargs):
        """Populate file placeholds in `dest` that mirros the
        remote root path.
        """
        for path, attr in self.accessor.walk(
                self.rootpath, return_attr=True, **kwargs):
            rpath = path.relative_to(self.rootpath)
            dpath = dest.joinpath(rpath)
            if stat.S_ISDIR(attr.st_mode):
                # mkdir in dest
                self.logger.debug(f"+ {dpath}")
                dpath.mkdir(exist_ok=True, parents=True)
            else:
                touch_file(dpath)
