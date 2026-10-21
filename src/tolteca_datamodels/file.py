"""File tables with built-in LMT and TolTEC support."""

from . import data_model as _data_model  # noqa: F401 -- enable built-in support.
from .core.file import FileAccessor

__all__ = ["FileAccessor"]
