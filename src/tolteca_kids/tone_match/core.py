"""Configured tone matching from saved KIDs blocks to portable products."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path  # noqa: TC003 -- public dataclass signature introspection.
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
import zarr
from astropy.table import QTable
from tollan.config import ConfigHandler, SubConfigKeyTransformer
from tollan.pipeline import ArtifactFile

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdAssoc,
    DataProdBatch,
    DataProdIndex,
    DataProdItem,
    DataProdSource,
    DataProdStore,
    Location,
    WorkflowRun,
    resolve_index,
)
from tolteca_datamodels.data_prod.metadata import ArtifactMeta, ToneMatchMeta
from tolteca_datamodels.raw_obs_context.workflow import _absolute_index
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids.kids_find import KidsFind
from tolteca_kids.match1d_result import _make_match_result, _table_dataset
from tolteca_kids.product import open_kids_product

from .compute import match_tones
from .config import ToneMatchConfig
from .result import TONE_MATCH_PRODUCER, ToneMatchView

if TYPE_CHECKING:
    from astropy.units import Quantity

    from .config import ToneMatchInput


@dataclass
class ToneMatchRun:
    """Full per-interface result tree and the optional publication directory."""

    data: xr.DataTree
    path: Path | None = None

    @property
    def tables(self) -> dict[str, QTable]:
        """Derived reference projections for each interface."""
        return {name: self.view(name).table for name in self.data.children}

    def view(self, interface: str) -> ToneMatchView:
        """Access the full result for one interface.

        Parameters
        ----------
        interface : str
            Readout interface present in this run, for example ``toltec0``.

        Returns
        -------
        ToneMatchView
            Input tables, pair correspondences and diagnostics for the interface.
        """
        return ToneMatchView(self.data[interface])


@dataclass
class _PreparedInput:
    selection: ToneMatchInput
    index: DataProdIndex
    index_path: Path
    table: QTable
    frequency: Quantity
    order: float
    identity: dict


def _load_input(selection):
    index_path = resolve_index(selection.source)
    index = DataProdIndex.read(index_path)
    if index.product.meta.data_prod_type != "dp_raw_obs_proc" or index.run is None:
        msg = "Tone matching requires published processed-acquisition realizations"
        raise ValueError(msg)
    with open_kids_product(index, interface=selection.interface) as tree:
        block = tree.toltec_kids.blocks[selection.block]
        meta = tree.toltec_kids.meta
        if block.data_kind not in ToltecDataKind.raw_sweep:
            msg = "Tone matching requires saved sweep blocks"
            raise ValueError(msg)
        if KidsFind.result_path() in block.tree.children:
            tables = KidsFind.view(block.tree).export_tables()
            table = tables[
                "kids_find"
                if block.data_kind == ToltecDataKind.vnasweep
                else "chan_prop"
            ]
        elif block.data_kind == ToltecDataKind.vnasweep:
            msg = "VNA tone matching requires saved KidsFind detections"
            raise ValueError(msg)
        else:
            axis = block.sweep.chan_axis_table
            axis = axis[np.asarray(axis["tone_mask"], dtype=bool)]
            table = QTable(
                {
                    "idx_chan": axis["chan"],
                    "f_chan": axis["f_chan"],
                    "amp_tone": axis["tone_amp"],
                }
            )
        table.meta["uid_raw_obs"] = meta.uid_raw_obs
        identity = {
            "product": index.product.reference.model_dump(mode="json"),
            "run_uid": str(index.run.uid),
            "interface": selection.interface,
            "block": selection.block,
            "uid_raw_obs": meta.uid_raw_obs,
            "f_lo_center": block.meta.f_lo_center,
            "frequency_column": "f_det"
            if block.data_kind == ToltecDataKind.vnasweep
            else "f_chan",
        }
        table.meta["input"] = identity
        frequency = table[
            "f_det" if block.data_kind == ToltecDataKind.vnasweep else "f_chan"
        ].copy()
        order = float(meta.t0.unix)
        return _PreparedInput(
            selection, index, index_path, table, frequency, order, identity
        )


def _prepare(config):
    groups = defaultdict(list)
    for selection in config.inputs:
        groups[selection.interface].append(_load_input(selection))
    for interface, rows in groups.items():
        rows.sort(
            key=lambda row: (row.order, row.selection.block, str(row.index.run.uid))
        )
        identities = [
            (row.index.product.reference, row.selection.block) for row in rows
        ]
        if len(set(identities)) != len(rows):
            msg = f"{interface}: duplicate block/result selection"
            raise ValueError(msg)
        references = [row for row in rows if row.selection.reference]
        if not rows[1:] or len(references) > 1:
            msg = f"{interface}: select at least two inputs and at most one reference"
            raise ValueError(msg)
        reference = references[0] if references else rows[0]
        groups[interface] = [reference, *(row for row in rows if row is not reference)]
        labels = Counter(
            (row.identity["uid_raw_obs"], row.selection.block) for row in rows
        )
        for i, row in enumerate(groups[interface]):
            label = f"{row.identity['uid_raw_obs']} · b{row.selection.block}"
            if labels[row.identity["uid_raw_obs"], row.selection.block] > 1:
                label += f" · {str(row.index.run.uid)[:8]}"
            row.identity.update(
                key=f"d{i:04d}",
                label=label,
            )
    return groups


def _index(uid, primary, inputs):
    refs = {row.index.product.reference for rows in inputs.values() for row in rows}
    return DataProdIndex(
        product=DataProd(
            uid=uid,
            meta=ToneMatchMeta(
                tag="tone_match",
                name=uid,
                data_prod_type="dp_tone_match",
                n_inputs=sum(map(len, inputs.values())),
            ),
            assocs=[
                DataProdAssoc(
                    uid=primary,
                    data_prod_assoc_type="dpa_tone_match_raw_obs",
                    is_primary=True,
                ),
                *(
                    DataProdAssoc(
                        **ref.model_dump(),
                        data_prod_assoc_type="dpa_tone_match_data_prod",
                    )
                    for ref in sorted(refs, key=lambda x: (x.uid, str(x.namespace_uid)))
                ),
            ],
        )
    )


def _write_result(data, path):
    for node in data.subtree:
        for array in node.data_vars.values():
            # Long VNA correlation vectors otherwise create thousands of tiny
            # chunks per array. Keep matrix tiles small for alignment inspection.
            chunk = 65536 if array.ndim == 1 else 256
            array.encoding["chunks"] = tuple(min(size, chunk) for size in array.shape)
    # DataTree otherwise reconsolidates the growing store at every node.
    data.to_zarr(path, mode="w-", zarr_format=2, consolidated=False)
    zarr.consolidate_metadata(path)


def _save_artifact(pending, table, interface, name, *, figure=None, viz=None):  # noqa: PLR0913 -- destination, scientific scope and optional figure export.
    if pending is None:
        return []
    folder = pending.base_path / "artifacts" / interface
    folder.mkdir(parents=True, exist_ok=True)
    if viz is None:
        target = folder / (name + ".ecsv")
        table.write(target, format="ascii.ecsv")
        receipts = [ArtifactFile(target, "ecsv", name=name)]
    else:
        receipts = viz.write(figure, path=folder / name)
    return [
        DataProdItem(
            meta=ArtifactMeta(
                type="artifact",
                format=receipt.format,
                name=name,
                producer=TONE_MATCH_PRODUCER,
                run_uid=pending.run.uid,
                interface=interface,
            ).model_dump(mode="json"),
            source=DataProdSource.model_validate(
                {
                    "path": receipt.path.relative_to(pending.base_path).as_posix(),
                    "location": "self",
                    "availability_state": "available",
                }
            ),
        )
        for receipt in receipts
    ]


def _calculate(config, inputs, pending):
    items, data = [], xr.DataTree()

    for interface, rows in inputs.items():
        reference = rows[0]
        ref_table, ref_frequency = reference.table, reference.frequency
        ref_uid = reference.identity["label"]
        data[interface] = xr.DataTree()
        node = data[interface]
        node.attrs.update(
            reference=reference.identity, inputs=[row.identity for row in rows]
        )
        for row in rows:
            node[f"inputs/{row.identity['key']}"] = xr.DataTree(
                dataset=_table_dataset(row.table)
            )
        collated = ref_table.copy()
        collated.meta["reference"] = reference.identity
        collated.meta["inputs"] = [row.identity for row in rows]
        for row in rows[1:]:
            result = match_tones(
                config.matcher,
                row.table,
                row.frequency,
                ref_table,
                ref_frequency,
                match_shift_max=config.match_shift_max,
                match_shift_step=config.match_shift_step,
                query_key=row.identity["key"],
            )
            query_uid = row.identity["key"]
            result.query.meta["reference"] = reference.identity
            pair = xr.DataTree(
                dataset=xr.Dataset(
                    attrs={
                        "query": row.identity["key"],
                        "reference": reference.identity["key"],
                        "role": "reference",
                    }
                )
            )
            pair["query"] = xr.DataTree(dataset=_table_dataset(result.query))
            pair["reference"] = xr.DataTree(dataset=_table_dataset(result.reference))
            pair["matched"] = _make_match_result(result.matched)
            node[f"pairs/{query_uid}"] = pair
            items.extend(
                _save_artifact(pending, result.query, interface, f"matched_{query_uid}")
            )
            for column in result.reference.colnames:
                if column.endswith(query_uid):
                    collated[column] = result.reference[column]
            collated.meta.update(result.reference.meta)
            if config.plot is not None and pending is not None:
                items.extend(
                    _save_artifact(
                        pending,
                        None,
                        interface,
                        f"matched_{query_uid}",
                        figure=config.plot(
                            result,
                            query_uid=row.identity["label"],
                            reference_uid=ref_uid,
                        ),
                        viz=config.plot,
                    )
                )
        node["lineage"] = xr.DataTree(dataset=_table_dataset(collated))
        items.extend(_save_artifact(pending, collated, interface, "lineage"))
        if config.lineage_plot is not None and pending is not None:
            items.extend(
                _save_artifact(
                    pending,
                    None,
                    interface,
                    "lineage",
                    figure=config.lineage_plot(ToneMatchView(node).overview),
                    viz=config.lineage_plot,
                )
            )
    data.attrs["config"] = config.model_dump(mode="json")
    if pending is not None:
        _write_result(data, pending.base_path / "result.zarr")
        items.append(
            DataProdItem(
                meta=ArtifactMeta(
                    type="artifact",
                    format="zarr",
                    name="tone_match",
                    producer=TONE_MATCH_PRODUCER,
                    run_uid=pending.run.uid,
                ).model_dump(mode="json"),
                source=DataProdSource.model_validate(
                    {
                        "path": "result.zarr",
                        "location": "self",
                        "availability_state": "available",
                    }
                ),
            )
        )
    return data, items


class ToneMatch(
    SubConfigKeyTransformer[Literal["tone_match"]], ConfigHandler[ToneMatchConfig]
):
    """Match saved sweep blocks using the root ``tone_match`` configuration.

    Parameters
    ----------
    runtime_context : tollan.config.RuntimeContext
        Input selections, matching parameters and output actions. No raw files
        are opened and no sweep checks or finding computations are repeated.
    """

    def __call__(self) -> ToneMatchRun:
        """Match each interface and optionally publish and register its product.

        Returns
        -------
        ToneMatchRun
            Full in-memory result tree and a pinned publication directory when output
            is enabled. Includes every pair and its alignment diagnostics.

        Raises
        ------
        ValueError
            Inputs are not sweeps, a required finding is absent, or reference and
            block/result selections are ambiguous.
        """
        cfg = self.config
        inputs = _prepare(cfg)
        first = inputs[min(inputs)][0]
        primary = first.identity["uid_raw_obs"]
        index = _index(cfg.uid or primary + "_tone_match", primary, inputs)
        path = None
        with ExitStack() as stack:
            pending = None
            if cfg.output.enabled:
                store = DataProdStore(cfg.output.path, namespace=cfg.output.namespace)
                run = WorkflowRun.create(runtime_info=self._rc.runtime_info)
                # Consumers need the exact input editions even after a tag moves.
                indexes = {
                    row.index.product.reference: row.index
                    for rows in inputs.values()
                    for row in rows
                }
                run.context["input_indexes"] = DataProdBatch(
                    items=[_absolute_index(index) for index in indexes.values()]
                ).model_dump(mode="json")
                stack.callback(store.write_run, run)
                stack.enter_context(run)
                pending = stack.enter_context(
                    store.materialize(
                        index.model_copy(update={"run": run}),
                        run_name=cfg.output.run_name,
                        tag=cfg.output.tag,
                    )
                )
            data, items = _calculate(cfg, inputs, pending)
            if pending is not None:
                pending.locations["self"] = Location(root=".", location_type="local")
                pending.product.data_items.extend(items)
                pending.provenance = {
                    "config": cfg.model_dump(mode="json"),
                    "inputs": [
                        row.identity for rows in inputs.values() for row in rows
                    ],
                    "software": {
                        name: version(name)
                        for name in (
                            "tolteca",
                            "tollan",
                            "numpy",
                            "astropy",
                            "dtw-python",
                        )
                    },
                }
        if pending is not None:
            path = pending.base_path
        if path is not None and cfg.output.registration:
            from tolteca_db.core import DataProdHandler  # noqa: PLC0415

            published = DataProdIndex.read(path)
            assert published.run is not None
            registrar = DataProdHandler(self._rc).index
            # Input runs may belong to another store/namespace. Ingest their
            # pinned indexes separately from this output namespace descriptor;
            # a descriptor's run filter is local to its own run registry.
            registrar.ingest(
                list(
                    dict.fromkeys(
                        row.index_path for rows in inputs.values() for row in rows
                    )
                ),
                run_uids=list(
                    {row.index.run.uid for rows in inputs.values() for row in rows}
                ),
            )
            registrar.ingest(
                [
                    path / "index.yaml",
                    path.parent / "index.yaml",
                    store.namespace_path / "namespace.yaml",
                ],
                run_uids=[published.run.uid],
            )
        return ToneMatchRun(data=data, path=path)
