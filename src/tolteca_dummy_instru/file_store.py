"""Candidate discovery in a synthetic instrument tree."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tolteca_datamodels.file_store import FileStoreBase
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

from .obs_spec import obs_spec_context
from .types import DummyInstruInfo

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

__all__ = ["DummyInstruFileStore"]


class DummyInstruFileStore(FileStoreBase[ObsSpec]):
    """Discover synthetic files under channel directories.

    Parameters
    ----------
    path : str or pathlib.Path
        Dummy instrument root, containing ``chan00`` through ``chan03``.
    """

    _obs_spec_context: ClassVar = obs_spec_context

    def _iter_candidate_patterns(self, spec: ObsSpec) -> Iterator[tuple[Path, str]]:
        master = spec.master if isinstance(spec.master, str) else "*"
        obsnum = str(spec.obsnum) if isinstance(spec.obsnum, int) else "*"
        for interface in spec.resolve_values(
            "interface", DummyInstruInfo.interfaces, context=self._obs_spec_context
        ):
            yield self.path, f"{interface}/{interface}-{master}-{obsnum}-*.nc"
