"""Prepare recorded calibration contexts for the configured KIDs consumer."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ConfigDict, Field
from pydantic_core import to_json
from tollan.config import FrozenBaseModel
from tollan.config.types import AbsAnyPath, PathSegment
from tollan.utils.fileloc import fileloc
from tollan.utils.yaml import yaml_load

from tolteca_datamodels.data_prod import (
    DataProdBatch,
    DataProdIndex,
    Location,
    WorkflowRun,
    build_raw_obs_batch,
    resolve_index,
)
from tolteca_datamodels.data_prod.lineage import collect_cal_refs
from tolteca_datamodels.raw_obs_context import (
    RawObsContext,
    RawObsDataItem,
    prepare_run_inputs,
    read_run_inputs,
)
from tolteca_datamodels.raw_obs_context.workflow import _artifact_bindings

from .model_binding import prepare_model_binding

__all__ = ["KidsContextConfig", "KidsModelSelection"]


class KidsModelSelection(FrozenBaseModel):
    """Select a fitted model from a product index.

    Parameters
    ----------
    index : pathlib.Path
        Exact model index, or a version-group index when tag is supplied.
    tag : str or None, optional
        Group tag to resolve once during preparation. None requires an exact
        realization. Replay uses the saved product identity instead of this tag.
    producer, name : str, optional
        Artifact selectors, default ``kidscpp.fit`` and ``model``.
    """

    model_config = ConfigDict(extra="forbid")
    index: AbsAnyPath
    tag: PathSegment | None = None
    producer: str = "kidscpp.fit"
    name: str = "model"


class KidsContextConfig(FrozenBaseModel):
    """Raw inventories and fitted models used for calibrated timestream reduction.

    Parameters
    ----------
    locations : dict of str to Location, optional
        Named absolute archive roots for discovered raw files.
    raw_indexes : list of pathlib.Path, optional
        Existing raw and observation indexes. Their complete inventory is retained
        when files enrich it. These may also supply the CLI's input files.
    models : list of KidsModelSelection, optional
        Candidate model indexes; exactly one must match each recorded sweep.
    replay : pathlib.Path or None, optional
        Saved run.yaml whose resolved contexts are reused. The selected blocks
        must agree. Models are found by saved identity in the supplied indexes;
        current tags are ignored. None prepares a new selection.
    """

    model_config = ConfigDict(extra="forbid")
    locations: dict[str, Location] = Field(default_factory=dict)
    raw_indexes: list[AbsAnyPath] = Field(default_factory=list)
    models: list[KidsModelSelection] = Field(default_factory=list)
    replay: AbsAnyPath | None = None


def _raw_index_sources(config):
    """Read the configured raw inventories and return their distinct sources."""
    sources = []
    for path in config.raw_indexes:
        index = DataProdIndex.read(path)
        if index.product.meta.data_prod_type != "dp_raw_obs":
            continue
        for item in index.product.data_items:
            if "interface" not in item.meta:
                continue
            source = index.get_source_uri(item.source)
            if source not in sources:
                sources.append(source)
    return sources


def _raw_source(index, interface):
    """Resolve the single file belonging to an acquisition interface."""
    items = [
        item
        for item in index.product.data_items
        if item.meta.get("interface") == interface
    ]
    if len(items) != 1:
        msg = f"Expected one raw file for {index.product.uid}/{interface}"
        raise ValueError(msg)
    location = fileloc(index.get_source_uri(items[0].source))
    if location.is_remote():
        msg = "Native solving requires locally accessible raw files"
        raise ValueError(msg)
    return location.path


@dataclass
class _ContextInputs:
    """Prepared inputs owned by one workflow invocation."""

    batch: DataProdBatch
    contexts: tuple[RawObsContext, ...]
    raw_indexes: dict
    model_indexes: dict
    artifact_indexes: dict = field(default_factory=dict)

    @property
    def inputs(self):
        return prepare_run_inputs(
            self.contexts,
            raw_products={
                uid: index.product for uid, index in self.raw_indexes.items()
            },
            model_indexes=self.model_indexes,
            artifact_indexes=self.artifact_indexes,
        )

    def calibration(self, uid, interface, block):
        context = next(ctx for ctx in self.contexts if ctx.raw_obs.uid == uid)
        item = next(
            item
            for item in context.data_items
            if (item.interface, item.block) == (interface, block)
        )
        return item.cal_items[0]

    def stub_sources(self, uid, interface):
        context = next(c for c in self.contexts if c.raw_obs.uid == uid)
        result = []
        for application in context.stubs:
            if application.interface != interface:
                continue
            index = self.artifact_indexes[application.artifact.product]
            application.resolve(index, raw_obs=context.raw_obs)
            result.append(
                {
                    "filepath": str(application.artifact.resolve_path(index)),
                    "mode": application.mode,
                }
            )
        return result

    def snapshot_indexes(self):
        indexes = {
            i.product.reference: i
            for i in (
                *self.raw_indexes.values(),
                *self.model_indexes.values(),
                *self.artifact_indexes.values(),
            )
        }
        copies = []
        for index in indexes.values():
            copy = index.model_copy(deep=True)
            copy.locations = {
                key: location.model_copy(
                    update={"root": index.resolve_location(key).url}
                )
                for key, location in index.locations.items()
            }
            copies.append(copy)
        return DataProdBatch(items=copies).model_dump(mode="json")


def _prepare_context_inputs(available, selected, config):
    """Prepare contexts without narrowing the available acquisition inventory."""
    batch = _raw_inventory(available, config)
    raw_indexes = {
        index.product.uid: index
        for index in batch.items
        if index.product.meta.data_prod_type == "dp_raw_obs"
    }
    products = {uid: index.product for uid, index in raw_indexes.items()}
    contexts = _select_contexts(products, selected)
    if config.replay is not None:
        contexts = _replay_contexts(contexts, config.replay)
    candidates = _model_candidates(config, replay=config.replay is not None)
    models, resolved = {}, []
    for initial in contexts:
        context = initial
        for item in initial.data_items:
            if len(item.cal_items) != 1:
                msg = "Every timestream block requires one recorded sweep"
                raise ValueError(msg)
            sweep = item.cal_items[0]
            target = sweep.data_items[0]
            index, producer, name = _select_model(
                sweep, candidates, replay=config.replay is not None
            )
            bound = prepare_model_binding(
                index,
                sweep=sweep,
                source=_raw_source(raw_indexes[sweep.raw_obs.uid], target.interface),
                producer=producer,
                name=name,
                artifact_source=None if target.model is None else target.model.source,
            )
            context = context.with_model(
                bound, interface=item.interface, block=item.block
            )
            models[index.product.reference] = index
        resolved.append(context)
    artifacts = {}
    required = {b.product for ctx in resolved for b in _artifact_bindings(ctx)}
    if required:
        saved = WorkflowRun.model_validate(yaml_load(config.replay))
        inventory = DataProdBatch.model_validate_json(
            to_json(saved.context["input_indexes"])
        )
        artifacts = {
            i.product.reference: i
            for i in inventory.items
            if isinstance(i, DataProdIndex) and i.product.reference in required
        }
        if set(artifacts) != required:
            msg = "Replay is missing exact artifact indexes"
            raise ValueError(msg)
    result = _ContextInputs(batch, tuple(resolved), raw_indexes, models, artifacts)
    _ = result.inputs  # Validate lineage and exact identities before computation.
    return result


def _raw_inventory(available, config):
    indexes = [DataProdIndex.read(path) for path in config.raw_indexes]
    existing = DataProdBatch(items=[*indexes])
    locations = dict(config.locations)
    for index in indexes:
        archive_locations = {
            item.source.location
            for item in index.product.data_items
            if "interface" in item.meta
        }
        for key in archive_locations:
            location = index.locations[key]
            resolved = location.model_copy(
                update={"root": index.resolve_location(key).url}
            )
            if key in locations and locations[key].resolve().url != resolved.root:
                msg = f"Conflicting storage location {key!r}"
                raise ValueError(msg)
            locations[key] = resolved
    sources = [
        {
            "source": row.io_obj if row.io_obj is not None else row.source,
            "source_loc": row.source,
        }
        for row in available.itertuples(index=False)
    ]
    return build_raw_obs_batch(sources, locations=locations, existing=existing)


def _select_contexts(products, selected):
    contexts = []
    for uid, rows in selected.groupby("uid_raw_obs", sort=False):
        refs = collect_cal_refs(products[uid])
        items = []
        for interface in rows.interface:
            ref = refs.get(interface)
            if ref is None or ref.meta.block_history is None:
                msg = f"Missing detector block history: {uid}/{interface}"
                raise ValueError(msg)
            blocks = ref.meta.block_history.blocks
            if any(block.kind != "timestream" for block in blocks):
                msg = (
                    "Select complete timestream acquisitions for calibrated solving; "
                    "sweeps remain available as calibration dependencies"
                )
                raise ValueError(msg)
            items.extend(
                RawObsDataItem(interface=interface, block=i) for i in range(len(blocks))
            )
        contexts.append(
            RawObsContext.from_product(
                products[uid], data_items=tuple(items), cal_products=products
            )
        )
    return contexts


def _replay_contexts(contexts, path):
    saved = read_run_inputs(WorkflowRun.model_validate(yaml_load(path)))

    def key(ctx):
        return ctx.raw_obs, frozenset(
            (item.interface, item.block) for item in ctx.data_items
        )

    if {key(ctx) for ctx in saved} != {key(ctx) for ctx in contexts}:
        msg = "Replay selection differs from the saved raw-observation contexts"
        raise ValueError(msg)
    return saved


def _select_model(sweep, candidates, *, replay):
    target = sweep.data_items[0]
    matches = []
    for index, selection in candidates:
        if replay:
            if target.model is None or index.product.reference != target.model.product:
                continue
            producer, name = target.model.producer, target.model.name
        else:
            producer, name = selection.producer, selection.name
        if not any(
            assoc.is_primary and assoc.uid == sweep.raw_obs.uid
            for assoc in index.product.assocs
        ):
            continue
        expected = {
            "interface": target.interface,
            "scope": f"blocks/{target.block}",
            "producer": producer,
            "name": name,
        }
        if any(
            all(artifact.meta.get(k) == v for k, v in expected.items())
            for artifact in index.product.data_items
        ):
            matches.append((index, producer, name))
    if len(matches) != 1:
        msg = (
            f"Expected one exact model for {sweep.raw_obs.uid}/{target.interface}/"
            f"blocks/{target.block}; found {len(matches)}"
        )
        raise ValueError(msg)
    return matches[0]


def _model_candidates(config, *, replay):
    result = []
    seen = set()
    for selection in config.models:
        index = DataProdIndex.read(selection.index)
        if index.run is not None:
            if selection.tag is not None:
                msg = "A tag requires a version-group index"
                raise ValueError(msg)
            members = [index]
        elif replay:
            members = index.products
        else:
            if selection.tag is None:
                msg = "Select a tag when supplying a model version group"
                raise ValueError(msg)
            path = resolve_index(selection.index, tag=selection.tag)
            members = [DataProdIndex.read(path)]
        for member in members:
            key = (member.product.reference, selection.producer, selection.name)
            if key not in seen:
                result.append((member, selection))
                seen.add(key)
    return result
