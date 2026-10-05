"""Shared interface for configured input discovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import ConfigDict
from tollan.config import FrozenBaseModel

if TYPE_CHECKING:
    import pandas as pd

    from .file import SourceInfoErrorPolicyT


class InputDiscoveryEntry(FrozenBaseModel, ABC):
    """Settings for one input-discovery operation."""

    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def discover(self, *, on_error: SourceInfoErrorPolicyT = "raise") -> pd.DataFrame:
        """Discover inputs using these settings.

        Parameters
        ----------
        on_error : {"raise", "record"}, optional
            Whether source-information failures raise (default) or remain in
            the returned table's ``diagnostics`` column. Invalid settings raise.

        Returns
        -------
        pandas.DataFrame
            Selected file rows in discovery order. No matches yields an empty
            table. Includes the ``source`` and ``source_loc`` columns supplied
            by ``FileAccessor.from_sources()``.
        """
