"""Single-source preparation and opening with built-in filename support.

Importing this module enables LMT and TolTEC filename parsers. The exported
class is the same as in ``core.file`` and uses all available registrations.
Use ``item.open()`` for the default format opener, or pass a SourceOpener policy
with ``item.open(opener=...)``.
"""

from . import filepath as _filepath  # noqa: F401 -- enable built-in filename parsers.
from .core.file import SourceSourceLoc

__all__ = ["SourceSourceLoc"]
