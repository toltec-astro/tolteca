"""Publish drive-fit artifacts and reopen them through portable product indexes."""

from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from importlib.metadata import version
from typing import TYPE_CHECKING
from urllib.parse import quote

import xarray as xr
from tollan.pipeline import PIPELINE_CONTEXT_KEY

from tolteca_datamodels.data_prod import (
    DataProdIndex,
    DataProdIndexAssoc,
    DataProdIndexLocation,
    DataProdIndexSource,
)
from tolteca_datamodels.data_prod_store import DataProdStore
from tolteca_kids.config import DataProdOutputConfig
from tolteca_kids.product import _arrays_store, _product_index

from .step import DriveFitStep

if TYPE_CHECKING:
    from pathlib import Path

    from tolteca_datamodels.data_prod_store import DataProdPublication

    from .step import DriveFitStepContext


def make_drive_fit_product_index(data: xr.DataTree) -> DataProdIndex:
    """Describe a sequence product before fitting or reserving its output run.

    Parameters
    ----------
    data : xarray.DataTree
        Prepared collection with input identities under each interface.

    Returns
    -------
    DataProdIndex
        Logical product identity and all raw-acquisition associations. The first
        acquisition in observation order is primary. Membership determines the
        logical identity independently of file location or interface selection.
    """
    members = {}
    for tree in data["interfaces"].children.values():
        inputs = tree["inputs"].to_dataset()
        for row in range(inputs.sizes["input"]):
            uid = str(inputs.uid_raw_obs.values[row])
            members[uid] = {
                "master": inputs.attrs["master"],
                **{
                    name: int(inputs[name].values[row])
                    for name in ("obsnum", "subobsnum", "scannum")
                },
            }
    if not members or len({v["master"] for v in members.values()}) != 1:
        msg = "A drive-fit product requires inputs from one master"
        raise ValueError(msg)
    ordered = sorted(members, key=lambda uid: tuple(members[uid].values()))
    primary = ordered[0]
    digest = sha256("\n".join(ordered).encode()).hexdigest()
    uid = f"{primary}_drive_fit_{digest}"
    return DataProdIndex(
        uid=uid,
        meta={
            "tag": "drivefit",
            "data_prod_type": "dp_drivefit",
            "name": uid,
            **members[primary],
            "n_items": len(members),
            "fit_method": "kid_phase_fit",
        },
        assocs=[
            DataProdIndexAssoc(
                uid=member,
                data_prod_assoc_type="dpa_drivefit_raw_obs",
                is_primary=member == primary,
            )
            for member in ordered
        ],
    )


def write_drive_fit_product(
    data: xr.DataTree,
    store: DataProdStore | Path,
    *,
    output_config: DataProdOutputConfig | None = None,
) -> Path:
    """Publish a completed sequence with scientific and diagnostic artifacts.

    Parameters
    ----------
    data : xarray.DataTree
        Collection with completed fit contexts on its interface scopes.
    store : DataProdStore or pathlib.Path
        Store or output root such as ``dataprod_toltec``.
    output_config : DataProdOutputConfig or None, optional
        Namespace, run and tag settings; None uses defaults. This function writes
        files only. The configured workflow also supports DPDB registration.

    Returns
    -------
    pathlib.Path
        Pinned realization directory containing ``index.yaml``.
    """
    return _write_product(
        data,
        store,
        make_drive_fit_product_index(data),
        output_config or DataProdOutputConfig(),
    )


def _write_product(
    data: xr.DataTree,
    store: DataProdStore | Path,
    index: DataProdIndex,
    config: DataProdOutputConfig,
    publication: DataProdPublication | None = None,
) -> Path:
    if not isinstance(store, DataProdStore):
        store = DataProdStore(store, namespace=config.namespace)
    primary = next(a.uid for a in index.assocs if a.is_primary)
    scope = (
        nullcontext(publication)
        if publication is not None
        else store.publication(
            index, primary_uid=primary, run=config.run, tag=config.tag
        )
    )
    with scope as pending:
        return _write_artifacts(data, pending, index, config)


def _write_artifacts(data, pending, index, config):
    items = []
    for interface, tree in data["interfaces"].children.items():
        records = tree.attrs.get(PIPELINE_CONTEXT_KEY, {})
        scientific = [
            ctx
            for ctx in records.values()
            if ctx.completed and ctx.data is not None and ctx.artifacts
        ]
        if not scientific:
            msg = f"{interface}: no completed scientific result for publication"
            raise ValueError(msg)
        for key, context in records.items():
            spec = context.config.artifacts
            if not context.completed or not context.artifacts or spec is None:
                continue
            path = pending.path / "artifacts" / interface / quote(key, safe="")
            items.extend(
                DataProdIndexSource(
                    source_uri=receipt.path.relative_to(pending.path).as_posix(),
                    location="run",
                    availability_state="available",
                    meta={
                        "type": "artifact",
                        "format": receipt.format,
                        "interface": interface,
                        "run_uid": str(pending.run.uid),
                    },
                )
                for receipt in spec.write(context.artifacts, path=path)
            )
    items.append(
        DataProdIndexSource(
            source_uri="index.yaml",
            location="run",
            availability_state="available",
            meta={
                "type": "artifact",
                "format": "data_prod_index",
                "run_uid": str(pending.run.uid),
            },
        )
    )
    completed = index.model_copy(
        update={
            "data_items": items,
            "locations": {
                "run": DataProdIndexLocation(root_uri=".", location_type="local")
            },
            "provenance": {
                "format": "tolteca.drive_fit.run.v1",
                "config": data.attrs.get("config"),
                "output": config.model_dump(mode="json"),
                "software": {
                    name: version(name)
                    for name in (
                        "tolteca",
                        "tollan",
                        "numpy",
                        "scipy",
                        "astropy",
                        "xarray",
                        "zarr",
                    )
                },
                "runtime_info": pending.runtime_info.model_dump(
                    mode="json", exclude={"config_sources", "validation_context"}
                ),
            },
        }
    )
    return pending.commit(completed)


def open_drive_fit_product(
    path: Path | DataProdIndex,
    *,
    interface: str | None = None,
    location: str | None = None,
    cache: Path | None = None,
) -> xr.DataTree:
    """Open one interface of a published drive fit without accessing raw data.

    Parameters
    ----------
    path : pathlib.Path or DataProdIndex
        Product directory, pinned index, or descriptor obtained from DPDB.
    interface : str or None, optional
        Readout interface. Required when the product contains several interfaces.
    location : str or None, optional
        Registered copy to read when several locations are available.
    cache : pathlib.Path or None, optional
        Cache for immutable remote arrays. None disables persistent caching.

    Returns
    -------
    xarray.DataTree
        Lazy scientific interface tree. Close it after reading. Access results
        with DriveFitStep.view and saved records with read_drive_fit_context.

    Raises
    ------
    ValueError
        The product type or interface selection is invalid.
    """
    index = _product_index(path)
    if index.meta["data_prod_type"] != "dp_drivefit":
        msg = "Expected a dp_drivefit product"
        raise ValueError(msg)
    return _open_interface(index, interface=interface, location=location, cache=cache)


def _open_interface(index, *, interface, location=None, cache=None):
    names = {
        item.meta.get("interface")
        for item in index.data_items
        if item.meta.get("format") == "zarr"
    }
    if interface is None:
        if len(names) != 1:
            msg = f"Select an interface: {sorted(names)}"
            raise ValueError(msg)
        interface = next(iter(names))
    if interface not in names:
        msg = f"Interface {interface!r} is absent from this product"
        raise ValueError(msg)
    selected = index.model_copy(
        update={
            "data_items": [
                item
                for item in index.data_items
                if item.meta.get("interface") == interface
            ],
        }
    )
    return xr.open_datatree(
        _arrays_store(selected, location=location, cache=cache),
        engine="zarr",
        consolidated=True,
        zarr_format=2,
    )


def read_drive_fit_context(
    data: xr.DataTree, *, alias: str | None = None
) -> DriveFitStepContext:
    """Read a saved fit execution record without reconstructing live payloads.

    Parameters
    ----------
    data : xarray.DataTree
        Reopened scientific interface tree.
    alias : str or None, optional
        Fit invocation label; None selects the primary invocation.

    Returns
    -------
    DriveFitStepContext
        Saved configuration, completion and failures, with data=None.
    """
    key = DriveFitStep.get_context_key(alias=alias)
    return DriveFitStep.load_context(data["execution"].attrs[key])
