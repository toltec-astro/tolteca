"""Configured discovery in an LMT acquisition tree."""

from __future__ import annotations

import re
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import Field, field_validator
from tollan.config.types import AbsAnyPath

from ..file_store import FileStoreInputDiscovery
from ..lmt_base.obs_spec import ObsSpec, ObsSpecField
from ..toltec.file import ToltecFileAccessor
from .file_store import LmtFileStore

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from ..file import SourceInfoErrorPolicyT
    from ..obs_spec import ObsSpecContext

__all__ = ["LmtInputDiscovery"]


class LmtInputDiscovery(FileStoreInputDiscovery):
    """Discover candidates in an LMT acquisition tree.

    Parameters
    ----------
    data_lmt_path : pathlib.Path
        Directory containing ``tel`` and instrument subtrees. A missing
        directory yields no candidates.
    instru : str, optional
        Regex selecting instrument stores to search, matched from the start of
        each registered name. Defaults to ``".*"``. The root store's own
        search is unaffected.
    obs_spec : str, int or ObsSpec, optional
        Observation selector, default ``"*"``. Used both to narrow discovery
        globs and to select rows after table population.
    mode : {"guess", "guess_or_read", "read"}, optional
        How to populate candidate information before ObsSpec selection.
        Default ``"guess"`` uses filenames only.
    """

    data_lmt_path: AbsAnyPath
    instru: str = ".*"
    obs_spec: ObsSpecField = Field(default_factory=ObsSpec)

    @field_validator("instru")
    @classmethod
    def _validate_instru(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.PatternError as exc:
            msg = f"Invalid instrument regex: {exc}"
            raise ValueError(msg) from exc
        return value

    @cached_property
    def _store(self) -> LmtFileStore:
        """File store for this discovery entry."""
        return LmtFileStore(self.data_lmt_path)

    def _find_candidates(self) -> list[Path]:
        return self._store.find_candidates(self.obs_spec, instru=self.instru)

    def _populate_candidates(
        self, paths: list[Path], *, on_error: SourceInfoErrorPolicyT
    ) -> pd.DataFrame:
        # TODO: Use an LMT accessor for telescope candidates and each selected
        # instrument's accessor for its candidates, preserving ownership/order.
        # Concatenate their tables before the shared discover() applies ObsSpec;
        # positional selection must operate on the combined discovery entry.
        return ToltecFileAccessor.from_sources(paths, mode=self.mode, on_error=on_error)

    @property
    def _obs_spec_context(self) -> ObsSpecContext:
        return self._store.get_obs_spec_context(instru=self.instru)
