"""File discovery in the TolTEC acquisition directory tree."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, ClassVar

from ..lmt_base.file_store import LmtFileStoreBase
from .obs_spec import obs_spec_context
from .types import ToltecInfo

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tollan.config.types import SubPathOrRoot

    from ..lmt_base.obs_spec import ObsSpec

__all__ = ["ToltecFileStore"]


class ToltecFileStore(LmtFileStoreBase):
    """Discover raw files in a TolTEC master/interface tree.

    Parameters
    ----------
    path : str or pathlib.Path
        TolTEC subtree, usually ``data_lmt/toltec``.
    """

    _obs_spec_context: ClassVar = obs_spec_context
    _subpaths: ClassVar = (PurePosixPath("."),)
    _interfaces: ClassVar = tuple(
        interface
        for interface, info in ToltecInfo.interface_interface_info.items()
        if info.instru == ToltecInfo.instru
    )

    def _iter_candidate_patterns(
        self, spec: ObsSpec, subpath: SubPathOrRoot
    ) -> Iterator[str]:
        master = str(spec.master) if isinstance(spec.master, str) else "*"
        obsnum = f"{spec.obsnum:06d}" if isinstance(spec.obsnum, int) else "*"
        for interface in spec.resolve_values(
            "interface",
            self._interfaces,
            context=self._obs_spec_context,
        ):
            filename = (
                f"{interface}_{obsnum}_*.nc"
                if interface in ToltecInfo.interface_roach
                else f"{interface}_*.nc"
            )
            yield f"{master}/{interface}/{filename}"
