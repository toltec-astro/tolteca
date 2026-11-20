"""Bind published models to the recorded calibration sweep."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 -- Sphinx resolves public annotations

from tolteca_datamodels.data_prod import (
    DataProdIndex,  # noqa: TC001 -- public signature
    DataProdSource,  # noqa: TC001 -- public signature
)
from tolteca_datamodels.raw_obs_context import (
    KidsModelBinding,
    SweepCalDataItem,
    SweepCalItem,
)

from .model_fit import open_fit

__all__ = ["prepare_model_binding"]


def prepare_model_binding(  # noqa: PLR0913 -- acquisition and artifact source selectors
    index: DataProdIndex,
    *,
    sweep: SweepCalItem,
    source: str | Path,
    producer: str,
    name: str,
    artifact_source: DataProdSource | None = None,
) -> SweepCalItem:
    """Validate and bind one published fit to a recorded sweep network.

    Parameters
    ----------
    index : DataProdIndex
        Exact processed-acquisition realization. Its source locations must be
        resolvable; for relative locations, read the saved index with
        DataProdIndex.read before preparation.
    sweep : SweepCalItem
        Recorded calibration with exactly one interface/block selection.
    source : str or pathlib.Path
        Original local calibration NetCDF, opened read-only and closed before
        return. Channel metadata is loaded; detector I/Q samples are not loaded.
    producer, name : str
        Exact artifact metadata selectors within the sweep's interface/block
        scope. For a native fit publication, use its recorded producer and name.

    artifact_source : DataProdSource or None, optional
        Exact model artifact source selected from an index or DB table.

    Returns
    -------
    SweepCalItem
        New binding with the same raw acquisition, interface and block, and the
        exact model product reference. Both inputs remain unchanged. Attach it
        with RawObsContext.with_model before preparing workflow-run inputs.

    Raises
    ------
    ValueError
        The selection is ambiguous, product ownership differs, or the native
        schema, acquisition, block, channel order, frequencies, units,
        normalization or completion state do not match the source sweep.

    Notes
    -----
    Supports version 1 ``kids.fit/1`` reports for legacy GainLinTrend and
    physical affine/delay models. It neither fits data nor installs controller
    files. Validation establishes which sweep produced the model; applicability
    to a timestream remains a solver check.
    """
    if len(sweep.data_items) != 1:
        msg = "Prepare one calibration sweep network at a time"
        raise ValueError(msg)
    [selected] = sweep.data_items
    model = KidsModelBinding(
        product=index.product.reference,
        producer=producer,
        name=name,
        source=artifact_source,
    )
    artifact = model.resolve(
        index, raw_obs=sweep.raw_obs, interface=selected.interface, block=selected.block
    )
    with open_fit(source, index.get_source_uri(artifact.source)) as (_, block):
        meta = block.meta
        if (meta.uid_raw_obs, meta.interface, meta.block) != (
            sweep.raw_obs.uid,
            selected.interface,
            selected.block,
        ):
            msg = "Native fit source differs from the recorded calibration sweep"
            raise ValueError(msg)
    return SweepCalItem(
        type="sweep",
        raw_obs=sweep.raw_obs,
        data_items=(
            SweepCalDataItem(
                interface=selected.interface, block=selected.block, model=model
            ),
        ),
    )
