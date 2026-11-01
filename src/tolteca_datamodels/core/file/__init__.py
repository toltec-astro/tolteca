"""File tables, source lifetimes and metadata collection.

Importing this package does not load instrument parsers. Filename parsing uses
all available registrations, including those loaded after this package is imported.
"""

from .accessor import FileAccessorBase
from .composite import CompositeFileAccessor, CompositeFileWorkflow, FileAccessorView
from .source import SourceSourceLoc
from .trace import SourceErrorPolicyT, SourceTrace, SourceTraceEntry, SourceTraceKind
from .workflow import FileAccessorWorkflow, FileTableModeT

__all__ = [
    "CompositeFileAccessor",
    "CompositeFileWorkflow",
    "FileAccessorBase",
    "FileAccessorView",
    "FileAccessorWorkflow",
    "FileTableModeT",
    "SourceErrorPolicyT",
    "SourceSourceLoc",
    "SourceTrace",
    "SourceTraceEntry",
    "SourceTraceKind",
]
