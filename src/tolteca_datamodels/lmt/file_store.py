"""File discovery in the LMT acquisition directory tree."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, cast

from tollan.utils.log import logger

from ..file_store import FileStoreBase
from ..lmt_base.obs_spec import ObsSpec
from ..toltec.file_store import ToltecFileStore
from .obs_spec import obs_spec_context
from .types import LmtInfo

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ..obs_spec import ObsSpecContext

__all__ = ["LmtFileStore"]


class LmtFileStore(FileStoreBase[ObsSpec]):
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
    """

    _obs_spec_context: ClassVar = obs_spec_context
    _substore_classes: ClassVar = {"toltec": ToltecFileStore}

    @property
    def toltec(self) -> ToltecFileStore:
        """TolTEC instrument file store."""
        return cast("ToltecFileStore", self.get_substore("toltec"))

    @property
    def tel_path(self) -> Path:
        """Telescope acquisition directory."""
        return self.path / "tel"

    def get_obs_spec_context(
        self, *, instru: str | re.Pattern[str] = ".*"
    ) -> ObsSpecContext:
        """Get selector context for the telescope and selected instrument stores.

        Parameters
        ----------
        instru : str or compiled regex, optional
            Instrument-name filter, as in `find_candidates`. Default is ".*".
            Telescope labels are always included.

        Returns
        -------
        ObsSpecContext
            Combined label interpretations for selecting the discovered table.

        Raises
        ------
        ValueError
            If selected stores declare conflicting numeric values for a label.
        """
        return self._get_obs_spec_context(self._get_instru_file_stores(instru))

    def find_candidates(
        self, spec: ObsSpec | str, *, instru: str | re.Pattern[str] = ".*"
    ) -> list[Path]:
        """Find candidate paths in the telescope and instrument directories.

        Parameters
        ----------
        spec : ObsSpec or str
            Observation selector. An omitted or wildcard interface includes both
            telescope and instrument trees. An explicit ``tel`` includes only the
            telescope directory; constraints excluding ``tel`` skip that directory.
        instru : str or compiled regex, optional
            Regex selecting instrument stores to search, matched from the start
            of each registered name. Defaults to ``".*"``. Telescope discovery
            is unaffected.

        Returns
        -------
        list of pathlib.Path
            Unique candidate paths without reading headers. Telescope candidates
            precede instrument candidates, retaining discovery order and the
            first occurrence of each path. Candidates may not satisfy every
            constraint; apply the complete selector after table population.
            No matches or a nonexistent directory produce an empty list.
        """
        return self._find_candidates(spec, self._get_instru_file_stores(instru))

    def _iter_candidate_patterns(self, spec: ObsSpec) -> Iterator[tuple[Path, str]]:
        if not spec.resolve_values(
            "interface", LmtInfo.interfaces, context=self._obs_spec_context
        ):
            return
        obsnum = f"{spec.obsnum:06d}" if isinstance(spec.obsnum, int) else "*"
        # The writer date precedes the observation number in telescope filenames.
        yield self.tel_path, f"tel_*_{obsnum}_*.nc"

    def _get_instru_file_stores(
        self, instru: str | re.Pattern[str]
    ) -> list[FileStoreBase[ObsSpec]]:
        pattern = re.compile(instru)
        selected = []
        for name, store in self.substores.items():
            included = pattern.match(name) is not None
            logger.debug(
                "{} instrument {} under {}: filter {!r}",
                "Including" if included else "Skipping",
                name,
                self.path,
                pattern.pattern,
            )
            if included:
                selected.append(store)
        return selected
