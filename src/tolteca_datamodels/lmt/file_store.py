"""File discovery in the LMT acquisition directory tree."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, ClassVar

from ..lmt_base.file_store import LmtFileStoreBase
from ..toltec.file_store import ToltecFileStore
from ._obs_spec import _obs_spec_context
from .types import LmtInfo

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from tollan.config.types import SubPath, SubPathOrRoot

    from ..file_store import SubstoreRegistryT
    from ..lmt_base.obs_spec import ObsSpec

__all__ = ["LmtFileStore"]


class LmtFileStore(LmtFileStoreBase):
    """Discover raw files in an LMT data tree.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory containing ``tel`` and instrument subdirectories.

    Notes
    -----
    An omitted or wildcard interface searches both telescope and instrument
    trees. An explicit ``tel`` searches only the telescope directory. An
    interface constraint excluding ``tel`` skips that directory.

    Examples
    --------
    >>> from pathlib import Path
    >>> store = LmtFileStore("/data_lmt")
    >>> store.tel_path == Path("/data_lmt/tel")
    True
    >>> store.toltec.path == Path("/data_lmt/toltec")
    True
    >>> from tolteca_datamodels.toltec.file_store import ToltecFileStore
    >>> store.get_substore(ToltecFileStore) is store.toltec
    True
    """

    _obs_spec_context: ClassVar = _obs_spec_context
    _tel_subpath: ClassVar[SubPath] = PurePosixPath("tel")
    _subpaths: ClassVar = (_tel_subpath,)
    _substore_paths: ClassVar[SubstoreRegistryT] = {
        ToltecFileStore: PurePosixPath("toltec")
    }

    @property
    def toltec(self) -> ToltecFileStore:
        """TolTEC instrument file store."""
        return self.get_substore(ToltecFileStore)

    @property
    def tel_path(self) -> Path:
        """Telescope acquisition directory."""
        return self.path / self._tel_subpath

    def _iter_candidate_patterns(
        self, spec: ObsSpec, subpath: SubPathOrRoot
    ) -> Iterator[str]:
        if subpath != self._tel_subpath:
            return
        if not spec.resolve_values(
            "interface", LmtInfo.interfaces, context=self._obs_spec_context
        ):
            return
        obsnum = f"{spec.obsnum:06d}" if isinstance(spec.obsnum, int) else "*"
        # The writer date precedes the observation number in telescope filenames.
        yield f"tel_*_{obsnum}_*.nc"
