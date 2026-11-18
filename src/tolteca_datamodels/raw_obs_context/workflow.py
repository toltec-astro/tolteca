"""Resolved observation contexts stored in the existing workflow-run inputs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping  # noqa: TC003 -- public signatures
from typing import Any

from pydantic import TypeAdapter

from ..core.data_prod.index import _index_location_label
from ..data_prod import (
    DataProd,
    DataProdBatch,
    DataProdIndex,
    DataProdRef,
    WorkflowRun,
)
from .calibration import ArrayPropTableCalItem, PointingCalItem, SweepCalItem
from .context import RawObsContext

__all__ = ["prepare_run_inputs", "read_run_indexes", "read_run_inputs"]

_contexts_type_adapter = TypeAdapter(list[RawObsContext])


def prepare_run_inputs(
    contexts: Iterable[RawObsContext],
    *,
    raw_products: Mapping[str, DataProd],
    model_indexes: Mapping[DataProdRef, DataProdIndex],
    artifact_indexes: Mapping[DataProdRef, DataProdIndex] | None = None,
) -> list[dict[str, Any]]:
    """Validate resolved contexts for WorkflowRun.context.inputs.

    Parameters
    ----------
    contexts : iterable of RawObsContext
        Nonempty resolved inputs, in consumer order. Every bound calibration
        sweep must have an exact model; inspection-only contexts are insufficient
        for a selected timestream with recorded calibration.
    raw_products : mapping of str to DataProd
        Source acquisitions and their recorded calibration dependencies by UID.
    model_indexes : mapping of DataProdRef to DataProdIndex
        Exact model indexes, keyed by full product and namespace identity.
    artifact_indexes : mapping of DataProdRef to DataProdIndex or None, optional
        Exact indexes for prepared representations and table calibration items.
        None supplies no additional artifact inventory.

    Returns
    -------
    list of dict
        Independent JSON-compatible snapshots. Supply as the ``inputs`` value
        in WorkflowRun.create(context=...). Other run context values are
        unchanged.

    Raises
    ------
    ValueError
        Inputs are empty, lineage differs, a model is unresolved, or its product
        or artifact does not match the recorded sweep.
    KeyError
        A required raw product or exact model index was not supplied.

    Notes
    -----
    No payload I/O, fitting or tag resolution occurs. Native report validation
    belongs to model preparation; physical applicability remains a solver check.
    """
    artifact_indexes = artifact_indexes or {}
    contexts = list(contexts)
    _require_models(contexts)
    for context in contexts:
        for stub in context.stubs:
            stub.resolve(
                artifact_indexes[stub.artifact.product], raw_obs=context.raw_obs
            )
        for binding in _artifact_bindings(context):
            binding.resolve(artifact_indexes[binding.product])
        context.validate_lineage(
            raw_products[context.raw_obs.uid], cal_products=raw_products
        )
        for cal in _calibrations(context):
            for item in cal.data_items:
                model = item.model
                assert model is not None
                model.resolve(
                    model_indexes[model.product],
                    raw_obs=cal.raw_obs,
                    interface=item.interface,
                    block=item.block,
                )
    return _contexts_type_adapter.dump_python(contexts, mode="json", round_trip=True)


def read_run_inputs(run: WorkflowRun) -> tuple[RawObsContext, ...]:
    """Restore the resolved observation selections saved with a workflow run.

    Parameters
    ----------
    run : WorkflowRun
        Reopened run with prepared raw-observation contexts under ``inputs``.

    Returns
    -------
    tuple of RawObsContext
        Typed snapshots, independent of the run's JSON containers. Reading does
        not follow current tags, reopen products, or choose newer calibrations.

    Raises
    ------
    ValueError
        Stored inputs are missing, empty, malformed or contain unresolved models.

    Notes
    -----
    This restores structure and exact identities. Use prepare_run_inputs with
    the pinned indexes to revalidate dependencies before a new execution.
    """
    if "inputs" not in run.context:
        msg = "Workflow run has no saved raw-observation inputs"
        raise ValueError(msg)
    result = _contexts_type_adapter.validate_json(json.dumps(run.context["inputs"]))
    _require_models(result)
    return tuple(result)


def _calibrations(context):
    for cal in context.cal_items:
        if isinstance(cal, SweepCalItem):
            yield cal
    for item in context.data_items:
        for cal in item.cal_items:
            if isinstance(cal, SweepCalItem):
                yield cal


def _require_models(contexts):
    if not contexts:
        msg = "At least one raw-observation context is required"
        raise ValueError(msg)
    if any(
        item.model is None
        for ctx in contexts
        for cal in _calibrations(ctx)
        for item in cal.data_items
    ):
        msg = "Every calibration sweep requires an exact model before execution"
        raise ValueError(msg)


def _artifact_bindings(context):
    for stub in context.stubs:
        yield stub.artifact
    for item in context.data_items:
        if item.representation is not None:
            yield item.representation
    for cal in (
        *context.cal_items,
        *(cal for item in context.data_items for cal in item.cal_items),
    ):
        if isinstance(cal, ArrayPropTableCalItem):
            yield cal.artifact
        elif isinstance(cal, PointingCalItem):
            yield from cal.artifacts


def _absolute_index(index):
    result = index.model_copy(deep=True)
    locations, labels = {}, {}
    for key, value in index.locations.items():
        root = index.resolve_location(key).url
        relative = value.is_relative
        # A document-local self/relative root needs the same stable label used
        # when DPDB imports that original index directly.
        label = _index_location_label(root) if relative or key == "self" else key
        labels[key] = label
        locations[label] = value.model_copy(update={"root": root})
    result.locations = locations
    for item in result.product.data_items:
        item.source.location = labels[item.source.location]
        # Metadata serializers can emit this set in different orders. Inventory
        # snapshots and preparation fingerprints use one stable representation.
        if "data_models" in item.meta:
            item.meta["data_models"] = sorted(item.meta["data_models"])
    # An index's own receipt is replaced when installed at a new location.
    result.product.data_items = [
        item
        for item in result.product.data_items
        if item.meta.get("type") != "data_prod_index"
    ]
    used = {i.source.location for i in result.product.data_items}
    result.locations = {k: v for k, v in result.locations.items() if k in used}
    result._bind_locations(None)
    return result


def read_run_indexes(run: WorkflowRun) -> dict[DataProdRef, DataProdIndex]:
    """Restore exact supporting indexes saved with raw-observation inputs.

    Parameters
    ----------
    run : WorkflowRun
        Record with an input_indexes DataProdBatch snapshot.

    Returns
    -------
    dict of DataProdRef to DataProdIndex
        Independent exact indexes with absolute locations. Version-group
        selectors are omitted; no external index or current tag is read.
    """
    batch = DataProdBatch.model_validate_json(json.dumps(run.context["input_indexes"]))
    return {
        index.product.reference: _absolute_index(index)
        for index in batch.items
        if isinstance(index, DataProdIndex)
        and index.product.meta.data_prod_type != "dp_version_group"
    }


def _snapshot_detector_inputs(contexts, indexes, *, include_apt):
    """Snapshot only the selected detector blocks and their consumed calibrations."""
    types = (SweepCalItem, ArrayPropTableCalItem) if include_apt else (SweepCalItem,)
    consumed = tuple(
        ctx.model_copy(
            update={
                "data_items": tuple(
                    item.model_copy(
                        update={
                            "representation": None,
                            "cal_items": tuple(
                                c for c in item.cal_items if isinstance(c, types)
                            ),
                        }
                    )
                    for item in ctx.data_items
                    if item.block is not None
                ),
                "cal_items": tuple(c for c in ctx.cal_items if isinstance(c, types)),
                "stubs": (),
            }
        )
        for ctx in contexts
    )
    required = {
        ref
        for ctx in consumed
        for ref in (
            ctx.raw_obs,
            *(cal.raw_obs for cal in _calibrations(ctx)),
            *(
                item.model.product
                for cal in _calibrations(ctx)
                for item in cal.data_items
            ),
            *(binding.product for binding in _artifact_bindings(ctx)),
        )
    }
    pinned = {ref: _absolute_index(indexes[ref]) for ref in sorted(required, key=str)}
    return consumed, {
        "inputs": prepare_run_inputs(
            consumed,
            raw_products={
                i.product.uid: i.product
                for i in indexes.values()
                if i.product.meta.data_prod_type == "dp_raw_obs"
            },
            model_indexes=indexes,
            artifact_indexes=indexes,
        ),
        "input_indexes": DataProdBatch(items=list(pinned.values())).model_dump(
            mode="json"
        ),
    }
