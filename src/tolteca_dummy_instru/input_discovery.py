"""Configured discovery in a dummy instrument subtree."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from tollan.config.types import AbsAnyPath

from tolteca_datamodels.file_store import FileStoreInputDiscovery
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec, ObsSpecField

from .file import DummyInstruFileAccessor
from .file_store import DummyInstruFileStore

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from tolteca_datamodels.file import SourceInfoErrorPolicyT
    from tolteca_datamodels.obs_spec import ObsSpecContext

__all__ = ["DummyInstruInputDiscovery"]


class DummyInstruInputDiscovery(FileStoreInputDiscovery):
    """Discover files in a dummy instrument subtree.

    Parameters
    ----------
    data_dummy_instru_path : pathlib.Path
        Directory containing the dummy channel directories. A missing directory
        yields no candidates. Relative paths use the runtime's rootpath.
    obs_spec : str, int or ObsSpec, optional
        Observation selector, default ``"*"``. Narrows candidate discovery and
        selects rows after population. Numeric interfaces refer to dummy channels.
    mode : {"guess", "guess_or_read", "read"}, optional
        Default ``"guess"`` uses filenames only. ``"guess_or_read"`` reads
        supported, openable files with unrecognized filenames; ``"read"`` reads
        all supported, openable files.
    """

    data_dummy_instru_path: AbsAnyPath
    obs_spec: ObsSpecField = Field(default_factory=ObsSpec)

    def _find_candidates(self) -> list[Path]:
        return DummyInstruFileStore(self.data_dummy_instru_path).find_candidates(
            self.obs_spec
        )

    def _populate_candidates(
        self, paths: list[Path], *, on_error: SourceInfoErrorPolicyT
    ) -> pd.DataFrame:
        return DummyInstruFileAccessor.from_sources(
            paths, mode=self.mode, on_error=on_error
        )

    @property
    def _obs_spec_context(self) -> ObsSpecContext:
        """Interface labels for the dummy instrument subtree."""
        return DummyInstruFileStore(self.data_dummy_instru_path).get_obs_spec_context()
