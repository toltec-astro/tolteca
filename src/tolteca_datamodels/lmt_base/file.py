"""Shared records for identifiers encoded in acquisition paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from pathlib import Path

    from tollan.utils.fileloc import FileLoc

__all__ = ["FilepathInfo"]


class FilepathInfo(BaseModel):
    """Observation identifiers parsed from an acquisition filepath.

    Telescope filename example (the bracketed suffix is optional)::

        {prefix}_{ut}_{obsnum}_{subobsnum}_{scannum}[_{suffix}].{ext}
        tel_toltec_2025-10-31_145609_00_0002.nc

    ``ut`` records the writer's file-creation time, not the observation time or
    filesystem modification time. Date-only values are represented as midnight UTC.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prefix: str
    interface: str
    instru: str | None
    instru_component: str | None
    master_str: str | None
    obsnum: int = Field(ge=0)
    subobsnum: int = Field(ge=0)
    scannum: int = Field(ge=0)
    ut: AwareDatetime | None
    suffix: str | None
    proc_suffix: str | None = None
    ext: str

    @classmethod
    def from_path(
        cls, source: str | Path | FileLoc, *, raise_on_error: bool = True
    ) -> FilepathInfo | None:
        """Parse a local or remote path without opening the source.

        Return the registered record type with all filename-derived fields
        populated. Calling on a subclass limits parsing to that record type.
        Unrecognized names raise ValueError, or return None when
        ``raise_on_error=False``. Invalid dates always raise ValueError.

        Examples
        --------
        >>> info = FilepathInfo.from_path("tel_toltec_2025-10-31_145609_00_0002.nc")
        >>> info.prefix, info.interface
        ('tel_toltec', 'tel')
        """
        # Assemble domain registrations only when parsing is requested. The
        # shared record and metadata classes can be imported independently.
        from ..file import FileIdentifier  # noqa: PLC0415

        return FileIdentifier.parse_filepath(
            source, info_cls=cls, raise_on_error=raise_on_error
        )
