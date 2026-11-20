"""Prepare destination-acquisition model stubs from exact source sweep fits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.pipeline import ArtifactFile, ArtifactSpec

from tolteca_datamodels.data_prod import DataProdItem, DataProdSource, Location
from tolteca_datamodels.data_prod.metadata import ArtifactMeta
from tolteca_datamodels.raw_obs_context import (
    ArtifactBinding,
    StubApplication,
    SweepCalItem,
)
from tolteca_datamodels.raw_obs_context.stub import MODEL_FIELDS
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids import BlockCreate
from tolteca_kids.model_fit import open_fit
from tolteca_kids.product import make_kids_product_index
from tolteca_kids.timestream_solve import _validate_detector_state
from tolteca_kids.tlaloc_output import TLALOC_REPORT_COLUMNS

from ._sources import _source

# These native headers identify the target and its physical detector settings.
_MATCH_FIELDS = tuple(
    f"Header.Toltec.{name}"
    for name in (
        "Master",
        "ObsNum",
        "SubObsNum",
        "ScanNum",
        "RoachIndex",
        "TargSweepObsNum",
        "TargSweepSubObsNum",
        "TargSweepScanNum",
        "AccumLen",
        "LoCenterFreq",
        "DriveAtten",
        "SenseAtten",
        "ToneFreq",
        "ToneAmp",
        "TonePhase",
        "ToneMask",
    )
)


@dataclass(frozen=True)
class AcquisitionStub:
    """Loaded partial acquisition, independent of the raw and fitted input files."""

    interface: str
    filename: str
    data: xr.Dataset


class ModelStubArtifact(ArtifactSpec[tuple[AcquisitionStub, ...]]):
    """Write one model stub per selected network in a consuming acquisition."""

    def __call__(self, context, *, indexes):
        """Prepare model payloads without writing or modifying any input files.

        Parameters
        ----------
        context : RawObsContext
            Destination acquisition with exact per-block sweep model bindings.
        indexes : mapping of DataProdRef to DataProdIndex
            Raw acquisitions and pinned fitted-model realizations. Recorded
            lineage and the sweep/target tone and power settings must agree.

        Returns
        -------
        tuple of AcquisitionStub
            One loaded partial NetCDF dataset per selected detector interface.
            Source sweep block numbers select fits; destination block numbers
            select header slots. Ancillary interfaces produce no model stubs.

        Raises
        ------
        ValueError
            Lineage, model binding, detector state or native layout is invalid.
            Existing detector stub applications must be cleared before preparing models.
            Telescope stubs are retained.
        """
        if any(s.interface.startswith("toltec") for s in context.stubs):
            msg = "Model preparation requires a context without stub applications"
            raise ValueError(msg)
        raw = indexes[context.raw_obs]
        context.validate_lineage(
            raw.product,
            cal_products={
                i.product.uid: i.product
                for i in indexes.values()
                if i.product.meta.data_prod_type == "dp_raw_obs"
            },
        )
        result = []
        for interface in dict.fromkeys(
            i.interface for i in context.data_items if i.block is not None
        ):
            items = [i for i in context.data_items if i.interface == interface]
            source = _source(raw, interface)
            with open_toltec_kids(source) as data:
                meta = data.toltec_kids.meta
                if (
                    not isinstance(meta, ToltecRawObsMetadata)
                    or meta.uid_raw_obs != context.raw_obs.uid
                    or f"toltec{meta.roach}" != interface
                ):
                    msg = "Raw source identity differs from the consuming acquisition"
                    raise ValueError(msg)
                blocks = BlockCreate()(ensure_datatree(data)).toltec_kids.blocks
                reports = {}
                for item in items:
                    target = blocks[item.block]
                    if not target.data_kind & ToltecDataKind.timestream:
                        msg = "Model stubs require selected timestream blocks"
                        raise ValueError(msg)
                    candidates = [
                        (cal, selected)
                        for cal in (*context.cal_items, *item.cal_items)
                        if isinstance(cal, SweepCalItem)
                        for selected in cal.data_items
                        if selected.interface == interface
                    ]
                    if len(candidates) != 1:
                        msg = "Each target block requires one recorded sweep"
                        raise ValueError(msg)
                    cal, selected = candidates[0]
                    model = selected.model
                    if model is None:
                        msg = "Model stub requires an exact fitted realization"
                        raise ValueError(msg)
                    fit_index = indexes[model.product]
                    model.resolve(
                        fit_index,
                        raw_obs=cal.raw_obs,
                        interface=interface,
                        block=selected.block,
                    )
                    binding = ArtifactBinding(
                        product=model.product,
                        producer=model.producer,
                        name=model.name,
                        source=model.source,
                        interface=interface,
                        scope=f"blocks/{selected.block}",
                    )
                    with open_fit(
                        _source(indexes[cal.raw_obs], interface),
                        binding.resolve_path(indexes[binding.product]),
                    ) as (table, sweep):
                        if (
                            sweep.meta.block != selected.block
                            or sweep.meta.uid_raw_obs != cal.raw_obs.uid
                        ):
                            msg = "Fit payload identifies another sweep block"
                            raise ValueError(msg)
                        _validate_detector_state(sweep, target)
                        reports[item.block] = table
                stub = _pack_models(data, reports)
            result.append(
                AcquisitionStub(interface, f"{source.stem}_kids_model_stub.nc", stub)
            )
        return tuple(result)

    def write(self, data, *, path):
        """Write the prepared network payloads into a product directory.

        Parameters
        ----------
        data : tuple of AcquisitionStub
            Loaded results returned by this artifact specification.
        path : pathlib.Path
            Materialization destination supplied by the data-product store.

        Returns
        -------
        list of ArtifactFile
            NetCDF receipts, named by interface. No source fit is rewritten.
        """
        receipts = []
        for stub in data:
            target = path / stub.filename
            stub.data.to_netcdf(target)
            receipts.append(ArtifactFile(path=target, format="nc", name=stub.interface))
        return receipts


def _model_columns(raw, reports):
    """Union model-owned columns while preserving unselected header slots."""
    params, header, _ = MODEL_FIELDS
    names = []
    original_names = []
    if params in raw:
        original_names = [
            v.decode() if isinstance(v, bytes) else str(v) for v in raw[header].values
        ]
        if len(set(original_names)) != len(original_names) or any(
            not n for n in original_names
        ):
            msg = "Acquisition model header has empty or duplicate fields"
            raise ValueError(msg)
        names.extend(original_names)
    for table in reports.values():
        required = [
            "f_in",
            "f_out",
            "flag",
            *[p["name"] for p in table.meta["model_schema"]["parameters"]],
        ]
        if table.meta["model_schema"]["name"] == "S21WithGainLinTrend":
            required.extend(TLALOC_REPORT_COLUMNS)
        names.extend(n for n in required if n not in names)
    return names, original_names


def _pack_models(raw, reports):
    params, header, normalization = MODEL_FIELDS
    stub = raw[list(_MATCH_FIELDS)].load().copy(deep=True)
    if (params in raw) != (header in raw):
        msg = "Acquisition contains an incomplete model header group"
        raise ValueError(msg)
    names, original_names = _model_columns(raw, reports)
    block_dim, chan_dim = raw["Header.Toltec.ToneFreq"].dims
    values = np.full((raw.sizes[block_dim], len(names), raw.sizes[chan_dim]), np.nan)
    attrs = {}
    if params in raw:
        previous = raw[params].values
        values[:, : len(original_names), :] = previous
        attrs = dict(raw[params].attrs)
    stub[params] = xr.DataArray(
        values, dims=(block_dim, "modelParamsNum", chan_dim), attrs=attrs
    )
    width = max(32, *map(len, names))
    stub[header] = xr.DataArray(
        np.asarray(names, dtype=f"S{width}"), dims="modelParamsNum"
    )
    stub[header].encoding["char_dim_name"] = "modelParamsHeaderLen"
    block_dim = stub[params].dims[0]
    count = stub.sizes[block_dim]
    if normalization in raw:
        values = np.broadcast_to(raw[normalization].values, (count,)).copy()
    else:
        # Unspecified legacy normalization stays unknown, never guessed from
        # the consuming timestream's (possibly different) accumulation length.
        values = np.full(count, np.nan)
    for block, table in reports.items():
        if block >= count:
            msg = "Acquired block has no matching native model header slot"
            raise ValueError(msg)
        stub[params].values[block, :, :] = np.nan
        for column, name in enumerate(names):
            if name not in table.colnames and name != "f_centered":
                continue
            values_col = (
                table["f_out"] - raw["Header.Toltec.LoCenterFreq"].item()
                if name == "f_centered"
                else table[name]
            )
            stub[params].values[block, column, :] = np.asarray(values_col)
        stub[params].attrs[f"model_schema_{block}"] = json.dumps(
            table.meta["model_schema"]
        )
        # Keep the complete group payload in the same atomic model variable.
        # Replacing a slot must also remove a previous slot's grouped response.
        stub[params].attrs.pop(f"joint_models_{block}", None)
        if "joint_models" in table.meta:
            stub[params].attrs[f"joint_models_{block}"] = json.dumps(
                table.meta["joint_models"], allow_nan=False
            )
        values[block] = table.meta["Header.Toltec.AccumLen"]
    stub[normalization] = xr.DataArray(values, dims=block_dim)
    # Preserve acquisition channel identity for the same native model decoder,
    # without adding detector I/Q or an artificial time axis.
    stub = stub.assign_coords(iqlen=raw["iqlen"].values)
    stub.encoding = {}
    stub.attrs = {
        "tolteca_stub_fields": json.dumps(MODEL_FIELDS),
        "tolteca_stub_match": json.dumps(_MATCH_FIELDS),
    }
    return stub


def materialize_model_stubs(context, *, indexes, store, run, mode="override"):
    """Prepare and persist model stubs owned by their consuming acquisition.

    Parameters
    ----------
    context : RawObsContext
        Resolved destination acquisition without preselected detector stubs.
    indexes : mapping of DataProdRef to DataProdIndex
        Exact raw and model dependencies, retained in the preparation run.
    store : DataProdStore
        Destination namespace. Store run naming is automatic.
    run : WorkflowRun
        Caller-owned preparation execution, shared across acquisitions.
    mode : {"override", "default"}, optional
        Explicit application policy saved in the returned context; default override.

    Returns
    -------
    tuple of (RawObsContext, DataProdIndex)
        Context selecting the newly written stubs and their bound processed index.
        The index's primary raw relation points to the consuming acquisition;
        provenance retains exact source sweep/model and target block selections.
    """
    if mode not in {"override", "default"}:
        msg = f"Unknown stub mode: {mode}"
        raise ValueError(msg)
    spec = ModelStubArtifact()
    data = spec(context, indexes=indexes)
    if not data:
        msg = "No detector blocks selected for model stub preparation"
        raise ValueError(msg)
    return _materialize_stubs(
        context,
        indexes=indexes,
        store=store,
        run=run,
        mode=mode,
        spec=spec,
        data=data,
        producer="raw_obs_context.models",
        name="kids_model",
    )


def _materialize_stubs(  # noqa: PLR0913 -- artifact materialization boundary
    context, *, indexes, store, run, mode, spec, data, producer, name
):
    """Materialize prepared acquisition stubs and append their exact applications."""
    meta = indexes[context.raw_obs].product.meta
    index = make_kids_product_index(
        meta.master, meta.obsnum, meta.subobsnum, meta.scannum, run=run
    )
    index.product.assocs[0].namespace_uid = context.raw_obs.namespace_uid
    index.provenance = {"raw_obs_context": context.model_dump(mode="json")}
    with store.materialize(index, tag=None) as bound:
        bound.locations["self"] = Location(root=".", location_type="local")
        for receipt in spec.write(data, path=bound.base_path):
            with receipt.path.open("rb") as handle:
                checksum = "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()
            bound.product.data_items.append(
                DataProdItem(
                    meta=ArtifactMeta(
                        type="artifact",
                        format="nc",
                        name=name,
                        producer=producer,
                        run_uid=run.uid,
                        interface=receipt.name,
                        kind="acquisition_stub",
                    ).model_dump(mode="json"),
                    source=DataProdSource.model_validate(
                        {
                            "location": "self",
                            "path": receipt.path.name,
                            "checksum": checksum,
                            "size": receipt.path.stat().st_size,
                            "availability_state": "available",
                        }
                    ),
                )
            )
    applications = tuple(
        StubApplication(
            interface=stub.interface,
            mode=mode,
            artifact=ArtifactBinding(
                product=bound.product.reference,
                producer=producer,
                name=name,
                interface=stub.interface,
            ),
        )
        for stub in data
    )
    return context.model_copy(update={"stubs": (*context.stubs, *applications)}), bound
