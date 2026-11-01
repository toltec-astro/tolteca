"""Configured input discovery with built-in LMT and TolTEC entries."""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

from .core.input_discovery import InputDiscovery as _InputDiscovery
from .core.input_discovery import InputDiscoveryConfig
from .core.input_discovery import query_file_store as _query_file_store
from .data_model import DataModelSet as _DataModelSet
from .lmt.input_discovery import LmtInputDiscovery as _LmtInputDiscovery
from .lmt_base.input_discovery import SourcesInputDiscovery as _SourcesInputDiscovery
from .toltec.input_discovery import ToltecInputDiscovery as _ToltecInputDiscovery

if _TYPE_CHECKING:
    import pandas as pd

    from .core.data_model import DataModelSet
    from .core.file import FileTableModeT, SourceErrorPolicyT
    from .core.file_store import FileStoreBase, StoreView
    from .core.obs_spec import ObsSpecBase

__all__ = ["InputDiscovery", "InputDiscoveryConfig", "query_file_store"]

InputDiscoveryConfig.register_input("data_lmt_path", _LmtInputDiscovery)
InputDiscoveryConfig.register_input("data_toltec_path", _ToltecInputDiscovery)
InputDiscoveryConfig.register_input("sources", _SourcesInputDiscovery)


class InputDiscovery(_InputDiscovery):
    """Discover configured files using assembled data-model support.

    Parameters
    ----------
    runtime_context : tollan.config.RuntimeContext
        Runtime containing ``input_discovery`` settings.

    Examples
    --------
    >>> from tollan.config import RuntimeContext
    >>> discovery = InputDiscovery(RuntimeContext({"input_discovery": {"inputs": [
    ...     {"sources": ["https://archive.example/data.nc"], "mode": "guess"},
    ... ]}}))
    >>> len(discovery.discover())
    1
    """

    def discover(
        self,
        *,
        data_models: DataModelSet | None = None,
        on_error: SourceErrorPolicyT = "record",
    ) -> pd.DataFrame:
        """Combine independently selected discovery entries.

        Parameters
        ----------
        data_models : DataModelSet, optional
            Available capabilities. Defaults to built-ins and activated extensions.
            Global and per-entry selectors may narrow this set.
        on_error : {"raise", "record"}, optional
            Source-information failure policy, default "record".

        Returns
        -------
        pandas.DataFrame
            Selected file rows in entry order, keeping the first occurrence of
            each location or local symlink target. Rows without a validated
            location retain their individual traces. No matches yields an empty
            table.
        """
        if data_models is None:
            data_models = _DataModelSet.from_select()
        return super().discover(data_models=data_models, on_error=on_error)


def query_file_store(
    store: FileStoreBase | StoreView,
    obs_spec: ObsSpecBase | str,
    *,
    mode: FileTableModeT = "guess",
    data_models: DataModelSet | None = None,
    on_error: SourceErrorPolicyT = "record",
) -> pd.DataFrame:
    """Query a store using assembled support and return a file table.

    Parameters
    ----------
    store : FileStoreBase or StoreView
        Store to search. Existing view restrictions remain in effect.
    obs_spec : ObsSpecBase or str
        Observation constraints applied to candidates and the populated table.
    mode : {"guess", "guess_or_read", "read"}, optional
        Information to acquire before selection, default "guess".
    data_models : DataModelSet, optional
        Store and file-table capabilities. Defaults to built-ins and activated
        extensions. An empty set searches no store locations.
    on_error : {"raise", "record"}, optional
        Source-information failure policy, default "record".

    Returns
    -------
    pandas.DataFrame
        Selected records in discovery order. Positional selection applies once
        after population. Resources opened for metadata reads are closed before
        return; borrowed objects remain open.
    """
    if data_models is None:
        data_models = _DataModelSet.from_select()
    return _query_file_store(
        store, obs_spec, mode=mode, data_models=data_models, on_error=on_error
    )
