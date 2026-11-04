"""Candidate discovery in a synthetic instrument tree."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, ClassVar

from tolteca_datamodels.lmt_base.file_store import LmtFileStoreBase

from ._obs_spec import _obs_spec_context
from .types import DummyInstruInfo

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tollan.config.types import SubPathOrRoot

    from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

__all__ = ["DummyInstruFileStore"]


class DummyInstruFileStore(LmtFileStoreBase):
    """Discover synthetic files under channel directories.

    Parameters
    ----------
    path : str or pathlib.Path
        Dummy instrument root, containing ``chan00`` through ``chan03``.
    """

    _obs_spec_context: ClassVar = _obs_spec_context
    _subpaths: ClassVar = (PurePosixPath("."),)

    def _iter_local_glob_patterns(
        self, spec: ObsSpec, subpath: SubPathOrRoot
    ) -> Iterator[str]:
        master = spec.master if isinstance(spec.master, str) else "*"
        obsnum = str(spec.obsnum) if isinstance(spec.obsnum, int) else "*"
        for interface in spec.resolve_values(
            "interface", DummyInstruInfo.interfaces, context=self._obs_spec_context
        ):
            yield f"{interface}/{interface}-{master}-{obsnum}-*.nc"
