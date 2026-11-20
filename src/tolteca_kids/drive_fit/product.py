"""Publish drive-fit artifacts and reopen them through portable product indexes."""

from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path  # noqa: TC003 -- Resolve public annotations.
from typing import TYPE_CHECKING

import xarray as xr
from pydantic import TypeAdapter
from tollan.config.types import PathSegment
from tollan.pipeline import PIPELINE_CONTEXT_KEY

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdAssoc,
    DataProdIndex,
    DataProdItem,
    DataProdSource,
    DataProdStore,
    Location,
    WorkflowRun,
)
from tolteca_datamodels.data_prod.metadata import ArtifactMeta, DrivefitMeta
from tolteca_kids.config import DataProdOutputConfig
from tolteca_kids.product import _arrays_store, _product_index

from .step import DriveFitStep

if TYPE_CHECKING:
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
        product=DataProd(
            uid=uid,
            meta=DrivefitMeta(
                tag="drivefit",
                data_prod_type="dp_drivefit",
                name=uid,
                master=str(members[primary]["master"]),
                obsnum=int(members[primary]["obsnum"]),
                subobsnum=int(members[primary]["subobsnum"]),
                scannum=int(members[primary]["scannum"]),
                n_items=len(members),
                fit_method="kid_phase_fit",
            ),
            assocs=[
                DataProdAssoc(
                    uid=member,
                    data_prod_assoc_type="dpa_drivefit_raw_obs",
                    is_primary=member == primary,
                )
                for member in ordered
            ],
        )
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
) -> Path:
    if not isinstance(store, DataProdStore):
        store = DataProdStore(store, namespace=config.namespace)
    with ExitStack() as stack:
        run = index.run
        if run is None:
            run = WorkflowRun.create()
            stack.callback(store.write_run, run)
            stack.enter_context(run)
            index = index.model_copy(update={"run": run})
        pending = stack.enter_context(
            store.materialize(index, run_name=config.run_name, tag=config.tag)
        )
        _write_artifacts(data, pending, config)
    assert pending.base_path is not None
    return pending.base_path


_artifact_path_segment = TypeAdapter(PathSegment)


def _write_artifacts(data, index, config):
    assert index.base_path is not None
    assert index.run is not None
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
            path = (
                index.base_path
                / "artifacts"
                / interface
                / _artifact_path_segment.validate_python(key)
            )
            for receipt in spec.write(context.artifacts, path=path):
                filename = receipt.path.relative_to(index.base_path).as_posix()
                items.append(
                    DataProdItem(
                        meta=ArtifactMeta(
                            type="artifact",
                            format=receipt.format,
                            name=filename if receipt.name is None else receipt.name,
                            producer=key,
                            run_uid=index.run.uid,
                            interface=interface,
                            scope=".",
                        ).model_dump(mode="json"),
                        source=DataProdSource.model_validate(
                            {
                                "path": filename,
                                "location": "self",
                                "availability_state": "available",
                            }
                        ),
                    )
                )
    index.locations["self"] = Location(root=".", location_type="local")
    index.product.data_items.extend(items)
    index.provenance = {
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
        "runtime_info": index.run.runtime_info.model_dump(
            mode="json", exclude={"config_sources", "validation_context"}
        ),
    }


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
    if index.product.meta.data_prod_type != "dp_drivefit":
        msg = "Expected a dp_drivefit product"
        raise ValueError(msg)
    return _open_interface(index, interface=interface, location=location, cache=cache)


def _open_interface(index, *, interface, location=None, cache=None):
    names = {
        item.meta.get("interface")
        for item in index.product.data_items
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
            "product": index.product.model_copy(
                update={
                    "data_items": [
                        item
                        for item in index.product.data_items
                        if item.meta.get("interface") == interface
                    ]
                }
            )
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
