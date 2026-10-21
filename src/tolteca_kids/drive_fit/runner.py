"""Discovery and sequence invocation of the drive-fit workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.input_discovery import InputDiscovery

from .core import DriveFit
from .input import _from_table

if TYPE_CHECKING:
    from collections.abc import Iterator

    import xarray as xr
    from tollan.config import RuntimeContext

__all__ = ["run"]


def run(runtime: RuntimeContext) -> Iterator[xr.DataTree]:
    """Fit discovered sequences and optionally publish/register their products.

    Parameters
    ----------
    runtime : tollan.config.RuntimeContext
        Runtime with ``input_discovery`` and root ``drive_fit`` settings.
        Select one sequence per master. Inputs may use locations, globs or ObsSpec.

    Yields
    ------
    xarray.DataTree
        One collection per master, with interface-local fits and plots and a
        shared product-output context. Conditional fit/plot settings receive
        master, roach and interface values. Source files are closed on return.
    """
    selected = _discover(runtime, DriveFit(runtime).config.select)
    for master, group in selected.groupby("master", sort=True):
        handler = DriveFit(runtime.with_config(validation_context={"master": master}))
        yield handler.process(_from_table(group))


def _discover(runtime, query):
    candidates = InputDiscovery(runtime).discover(
        data_models=DataModelSet.from_select(selection=["toltec"]),
        on_error="raise",
    )
    if candidates.empty:
        msg = "No target/tune sweeps selected for drive fitting"
        raise ValueError(msg)
    with candidates.toltec_file.open(
        update_from_metadata=True, on_error="raise"
    ) as table:
        selected = table[table["file_suffix"].isin(["targsweep", "tune"])]
        if query is not None:
            selected = selected.query(query)
        selected = selected.drop(columns=["io_obj"], errors="ignore").copy()
    if selected.empty:
        msg = "No target/tune sweeps selected for drive fitting"
        raise ValueError(msg)
    return selected
