"""Filename information with built-in LMT and TolTEC parsers.

Importing this module enables those parsers. `FilepathInfo` is the same class
exported by ``tolteca_datamodels.core.filepath``; both use all available parser
registrations. Optional extensions must load their own parsers.
"""

from .core.filepath import FilepathInfo
from .lmt import filepath as _lmt_filepath  # noqa: F401 -- enable LMT parsers.
from .toltec import filepath as _toltec_filepath  # noqa: F401 -- enable TolTEC parsers.

__all__ = ["FilepathInfo"]
