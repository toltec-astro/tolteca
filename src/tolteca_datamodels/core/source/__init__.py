"""Format opener and data-model reader declarations for extensions."""

from .opener import SourceOpener, source_opener
from .reader import SourceReader, SourceReaderRules

__all__ = [
    "SourceOpener",
    "SourceReader",
    "SourceReaderRules",
    "source_opener",
]
