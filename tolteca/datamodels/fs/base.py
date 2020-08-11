#!/usr/bin/env python


__all__ = ['FileStoreAccessor', ]


class FileStoreAccessor(object):
    """A common interface for accessing file storage."""

    def query(self, *args, **kwargs):
        """Return a table info object."""
        return NotImplemented

    def glob(self, *args, **kwargs):
        """Return a list of file paths."""
        return NotImplemented
