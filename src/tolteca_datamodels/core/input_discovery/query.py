"""Queries combining file-store discovery and file-table selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tollan.utils.log import logger

from ..file import FileAccessorView

if TYPE_CHECKING:
    import pandas as pd

    from ..data_model import DataModelSet
    from ..file import FileTableModeT, SourceErrorPolicyT
    from ..file_store import FileStoreBase, StoreView
    from ..obs_spec import ObsSpecBase

__all__ = ["query_file_store"]


def query_file_store(
    store: FileStoreBase | StoreView,
    obs_spec: ObsSpecBase | str,
    *,
    mode: FileTableModeT = "guess",
    data_models: DataModelSet,
    on_error: SourceErrorPolicyT = "record",
) -> pd.DataFrame:
    """Query a file store and return matching file records as a table.

    Parameters
    ----------
    store : FileStoreBase or StoreView
        Store or bound view to search. Existing location restrictions remain
        in effect. Selected data models further restrict the store locations.
    obs_spec : ObsSpecBase or str
        Observation constraints for candidate discovery and the final table
        query. Text uses the store's and selected workflows' selector classes.
        Positional selection applies once, after all candidates are populated.
    mode : {"guess", "guess_or_read", "read"}, optional
        Information to acquire before selection, default "guess". "guess_or_read"
        attempts metadata reads for sources without identified data-model
        memberships; "read" also reads sources matched by the selected accessor
        types. Sources identified outside these capabilities retain their guesses.
    data_models : tolteca_datamodels.core.data_model.DataModelSet
        Store requirements and file-accessor types to use.
        An empty set searches no locations.
    on_error : {"raise", "record"}, optional
        Source-information failure policy, default "record". Failures remain in
        source_trace unless "raise" is requested. Invalid arguments still raise.

    Returns
    -------
    pandas.DataFrame
        Matching records in discovery order, or an empty table when no records
        match. Filename parsing uses all available registrations. Resources
        opened for metadata reads are closed before returning.

    Notes
    -----
    No runtime configuration is required. The core function uses the supplied
    capability set. Import from ``tolteca_datamodels.input_discovery`` to default
    to assembled LMT, TolTEC and explicitly activated extension support.
    """
    search = store.partial(store_types=data_models.file_store_types)
    candidates = search.glob_obs_spec(obs_spec)
    file_accessor = FileAccessorView.from_sources(
        candidates,
        mode=mode,
        accessor_types=data_models.file_accessor_types,
        on_error=on_error,
    )
    table = file_accessor.table
    selected = table if table.empty else file_accessor.query_obs_spec(obs_spec)
    logger.debug(
        "File-store query: mode={}, obs_spec={}; {} candidates -> {} selected; "
        "data_models={}",
        mode,
        obs_spec,
        len(candidates),
        len(selected),
        list(data_models.data_models),
    )
    return selected
