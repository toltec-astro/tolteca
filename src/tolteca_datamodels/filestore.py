"""Shared configuration for file discovery and product locations."""

from typing import Literal

from pydantic import ConfigDict
from tollan.config import ConfigHandler, FrozenBaseModel, SubConfigKeyTransformer
from tollan.config.types import AbsAnyPath, AbsDirectoryPath

__all__ = ["FileAccess", "FileAccessConfig"]


class FileAccessConfig(FrozenBaseModel):
    """Paths and catalog selection shared by file-based applications.

    Explicit input files do not require ``data_lmt_path``. A product root need
    not exist yet; validation does not create it or enable product output.
    ``select`` is a pandas expression evaluated on the input metadata catalog.
    """

    model_config = ConfigDict(extra="forbid")
    data_lmt_path: AbsDirectoryPath | None = None
    dataprod_toltec_path: AbsAnyPath | None = None
    select: str | None = None


class FileAccess(
    SubConfigKeyTransformer[Literal["files"]], ConfigHandler[FileAccessConfig]
):
    """Read shared file-access settings from a runtime's ``files`` section.

    Examples
    --------
    >>> from tollan.config import RuntimeContext
    >>> files = FileAccess(RuntimeContext({"files": {"select": "roach < 7"}}))
    >>> files.config.select
    'roach < 7'

    Configure these settings unconditionally: discovery and selection happen
    before per-input configuration is evaluated.
    """
