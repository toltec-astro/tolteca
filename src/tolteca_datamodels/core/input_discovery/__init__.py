"""Discovery-entry bases and configuration APIs for input extensions."""

from .config import InputDiscovery, InputDiscoveryConfig
from .entries import (
    FileStoreInputDiscovery,
    GlobSource,
    InputDiscoveryEntry,
    SourcesInputDiscovery,
)
from .query import query_file_store

__all__ = [
    "FileStoreInputDiscovery",
    "GlobSource",
    "InputDiscovery",
    "InputDiscoveryConfig",
    "InputDiscoveryEntry",
    "SourcesInputDiscovery",
    "query_file_store",
]
