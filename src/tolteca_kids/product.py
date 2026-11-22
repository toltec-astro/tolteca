"""Publish and reopen acquisition-level KIDs products.

Products use ``dp_raw_obs_proc/index.yaml`` in a `DataProdStore`. Each run
contains one Zarr tree and a YAML index with publication provenance.
The YAML associations identify the primary raw product by UID.

Each interface's acquisition/block tree occupies ``<interface>`` directly under
the Zarr root, including products with only one interface.
`open_sweep_product` and `open_timestream_product` select an interface and return
its scientific tree. Readers also accept a `DataProdIndex` from DPDB without
opening YAML. Supply ``interface`` when more than one is present.
"""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 -- Resolve public annotations.
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from http import HTTPStatus
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

import fsspec
import xarray as xr
from aiohttp import ClientError
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from fsspec.implementations.cached import SimpleCacheFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.utils import atomic_write
from pydantic import ConfigDict, TypeAdapter
from tollan.accessor import NameMapping, Schema
from tollan.accessor.xarray import XarrayMapper
from tollan.config import FrozenBaseModel, RuntimeInfo
from tollan.config.types import PathSegment
from tollan.pipeline import (
    PIPELINE_CONTEXT_KEY,
    ArtifactContext,
    ArtifactFile,
    ArtifactSpec,
    Pipeline,
)
from tollan.utils.fileloc import FileLoc, fileloc
from zarr.storage import FsspecStore, LocalStore

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdAssoc,
    DataProdIndex,
    DataProdItem,
    DataProdRef,
    DataProdSource,
    DataProdStore,
    Location,
    WorkflowRun,
    make_raw_obs_product_uid,
    resolve_index,
)
from tolteca_datamodels.data_prod.metadata import ArtifactMeta, RawObsProcMeta
from tolteca_datamodels.toltec.kids.core import ToltecKidsIOMapper

from ._reduction import select_block
from .accessor import ToltecKidsTreeMapper
from .blocks import NODE_TYPE
from .config import DataProdOutputConfig, KidsConfig
from .sweep import ToltecSweepMapper
from .sweep_check_view import SweepCheckMapper
from .sweep_reduce import SweepReduce, SweepReduceContext
from .timestream_reduce import TimestreamReduce, TimestreamReduceContext

if TYPE_CHECKING:
    from tollan.accessor import Mapping as SchemaMapping

__all__ = [
    "KidsDataArtifact",
    "KidsInputProvenance",
    "KidsRunProvenance",
    "make_kids_product_index",
    "open_kids_product",
    "open_kids_product_tree",
    "open_sweep_product",
    "open_timestream_product",
    "read_product_interfaces",
    "read_product_provenance",
    "read_sweep_context",
    "read_timestream_context",
    "write_kids_product",
    "write_sweep_product",
    "write_timestream_product",
]


class KidsInputProvenance(FrozenBaseModel):
    """Selected acquisition file and its resolved workflow configuration.

    Parameters
    ----------
    uid_raw_obs_file : str
        Canonical identity of the selected acquisition file.
    source : FileLoc or None, optional
        Input requested by the caller. None when no input location was supplied.
    source_loc : FileLoc or None, optional
        Resolved input location used by the file reader, when known.
    config : KidsConfig or None, optional
        Full workflow settings after defaults, overrides and conditions resolve.
        None for independently invoked steps without a workflow snapshot; their
        saved execution records retain the individual step settings.
    """

    model_config = ConfigDict(extra="forbid")
    uid_raw_obs_file: str
    source: FileLoc | None = None
    source_loc: FileLoc | None = None
    config: KidsConfig | None = None


class KidsRunProvenance(FrozenBaseModel):
    """Inputs, publication settings and environment of one KIDs publication.

    Parameters
    ----------
    runtime_info : RuntimeInfo
        Publication runtime, including host and process information.
    software : dict of str to str
        Installed package versions at publication.
    output : DataProdOutputConfig
        Settings used to publish this run.
    inputs : dict of str to KidsInputProvenance
        Selected acquisition files and their workflow settings, keyed by interface.
    format : str, optional
        Provenance format identifier; defaults to ``tolteca.kids.run.v1``.
    """

    model_config = ConfigDict(extra="forbid")
    format: Literal["tolteca.kids.run.v1"] = "tolteca.kids.run.v1"
    runtime_info: RuntimeInfo
    software: dict[str, str]
    output: DataProdOutputConfig
    inputs: dict[str, KidsInputProvenance]


@dataclass
class _ProductSchema(Schema):
    """Acquisition collection in a published scientific tree."""

    interfaces: SchemaMapping = NameMapping.path(".")


class _ProductMapper(XarrayMapper[_ProductSchema]):
    """Interface locations beneath a product root."""

    def get_interfaces(self, tree):
        nodes = self.get_node(tree, self.schema.interfaces).children.values()
        return {node.toltec_kids.meta.interface: node.path for node in nodes}


class KidsDataArtifact(ArtifactSpec[xr.DataTree]):
    """Scientific acquisition trees and saved step records in ``array.zarr``."""

    def __call__(
        self, data: Mapping[str, xr.DataTree], *, config: DataProdOutputConfig
    ) -> xr.DataTree:
        """Prepare an acquisition's interface trees for persistence.

        Parameters
        ----------
        data : mapping of str to xarray.DataTree
            Scientific trees keyed by interface. Present raw data and results
            are retained; explicitly prepared trees without raw data remain
            without it. Input trees and their live contexts are unchanged.
        config : DataProdOutputConfig
            Chunk sizes for the prepared arrays.

        Returns
        -------
        xarray.DataTree
            Product tree with one child per interface and serialized step
            records. Lazy sources must stay open until writing finishes. The
            same prepared tree can be written to multiple destinations.
        """
        output = xr.DataTree()
        for name, tree in data.items():
            output[name] = _product_tree(tree, config)
        return output

    def write(self, data: xr.DataTree, *, path: Path) -> list[ArtifactFile]:
        """Write a prepared scientific tree and return its directory receipt.

        Parameters
        ----------
        data : xarray.DataTree
            Prepared product tree. Keep lazy sources open until this returns.
        path : pathlib.Path
            Destination directory for ``array.zarr``. An existing array store
            is rejected. The input tree is unchanged.

        Returns
        -------
        list of ArtifactFile
            One Zarr directory receipt named ``Scientific data``.
        """
        target = path / "array.zarr"
        data.to_zarr(target, mode="w-", zarr_format=2, consolidated=True)
        return [ArtifactFile(target, "zarr", name="Scientific data")]


def read_product_provenance(path: Path | DataProdIndex) -> KidsRunProvenance:
    """Read a completed KIDs run's provenance without opening arrays.

    Parameters
    ----------
    path : pathlib.Path or DataProdIndex
        DB descriptor, product directory, main index, or pinned run index.

    Returns
    -------
    KidsRunProvenance
        Actual workflow snapshots, selected inputs and publication environment.

    Raises
    ------
    pydantic.ValidationError
        The index does not contain valid KIDs run provenance.
    """
    index = _product_index(path)
    return KidsRunProvenance.model_validate(index.provenance)


def _product_index(source):
    return (
        source
        if isinstance(source, DataProdIndex)
        else DataProdIndex.read(resolve_index(source))
    )


class _PublishedHTTPFileSystem(HTTPFileSystem):
    """Publish complete cached HTTP objects and retain confirmed absences."""

    def __init__(self, *args, missing_cache, **kwargs):
        self._missing_cache = missing_cache
        self._missing_cache.mkdir(parents=True, exist_ok=True)
        super().__init__(*args, **kwargs)

    @override
    async def _get_file(self, rpath, lpath, *args, **kwargs):
        # Simplecache considers any final cache filename readable. Stage the
        # body beside it so concurrent readers never open an incomplete object.
        with atomic_write(lpath) as target:
            await super()._get_file(rpath, target, *args, **kwargs)

    @override
    async def _exists(self, path, strict=True, **kwargs):
        marker = self._missing_cache / sha256(path.encode()).hexdigest()
        if marker.exists():
            return False
        # Simplecache checks existence before downloading the same object.
        # HEAD avoids transferring its body twice; GET-only hosts retain the
        # original fsspec check. Service errors must not become fill chunks.
        session = await self.set_session()
        try:
            async with session.head(
                self.encode_url(path), **(self.kwargs | kwargs)
            ) as response:
                if response.status in (
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    HTTPStatus.NOT_IMPLEMENTED,
                ):
                    exists = await super()._exists(path, strict=strict, **kwargs)
                else:
                    self._raise_not_found_for_status(response, path)
                    exists = True
        except FileNotFoundError:
            exists = False
        except ClientError:
            if strict:
                raise
            return False
        if not exists and strict:
            # Zarr omits all-fill chunks. Their confirmed absence is part of a
            # cached selection, and remains valid for this completed run URI.
            marker.touch()
        return exists


def _arrays_store(index, *, location=None, cache=None):
    items = [
        item
        for item in index.product.data_items
        if item.meta.get("format") == "zarr"
        and item.meta.get("name") == "Scientific data"
        and (location is None or item.source.location == location)
    ]
    if len(items) != 1:
        msg = "Select one registered Zarr source with location"
        raise ValueError(msg)
    uri = index.get_source_uri(items[0].source)
    fs, path = fsspec.core.url_to_fs(uri)
    if isinstance(fs, LocalFileSystem):
        # url_to_fs strips file:// but retains URI escapes. FileLoc decodes
        # exactly once, including literal '%' characters in artifact names.
        return LocalStore(fileloc(uri).path, read_only=True)
    if cache is None:
        return FsspecStore.from_url(uri, read_only=True)
    if isinstance(fs, HTTPFileSystem):
        fs = _PublishedHTTPFileSystem(
            missing_cache=Path(cache) / "missing", **fs.storage_options
        )
    # Simplecache uses synchronous target methods. Wrap the completed cache,
    # rather than enabling asynchronous methods on its HTTP target.
    cached = SimpleCacheFileSystem(fs=fs, cache_storage=str(cache))
    return FsspecStore(
        AsyncFileSystemWrapper(cached, asynchronous=True), path=path, read_only=True
    )


def _open_product_tree(
    source, *, location=None, cache=None, group=None, create_default_indexes=False
):
    index = _product_index(source)
    KidsRunProvenance.model_validate(index.provenance)
    return xr.open_datatree(
        _arrays_store(index, location=location, cache=cache),
        engine="zarr",
        consolidated=None,
        zarr_format=2,
        create_default_indexes=create_default_indexes,
        group=group,
    )


def open_kids_product_tree(
    path: Path | DataProdIndex,
    *,
    location: str | None = None,
    cache: Path | None = None,
) -> xr.DataTree:
    """Open all published interface subtrees with lazy coordinates and arrays.

    Parameters
    ----------
    path : pathlib.Path or DataProdIndex
        Product directory, pinned run index or DB descriptor.
    location : str or None, optional
        Registered location key, required for descriptors with several copies.
    cache : pathlib.Path or None, optional
        Persistent cache for immutable remote Zarr objects.

    Returns
    -------
    xarray.DataTree
        Acquisition root with one child per interface. Use as a context manager
        to share metadata across interface selections and close all stores.
        For a single interface, use `open_kids_product` to limit metadata reads.
    """
    return _open_product_tree(path, location=location, cache=cache)


def read_product_interfaces(
    path: Path | DataProdIndex,
    *,
    location: str | None = None,
    cache: Path | None = None,
) -> dict[str, str]:
    """Locate saved interfaces using scientific metadata without reading arrays.

    Parameters
    ----------
    path : pathlib.Path or DataProdIndex
        Product directory, pinned run index, or descriptor obtained from DPDB.
    location : str or None, optional
        Registered location key. Required when several Zarr copies are listed.
    cache : pathlib.Path or None, optional
        Persistent cache for immutable remote Zarr objects. None reads remotely
        without a disk cache. Local artifacts are read directly.

    Returns
    -------
    dict of str to str
        Interface names and their paths within the Zarr tree.
    """
    with open_kids_product_tree(path, location=location, cache=cache) as tree:
        return _ProductMapper.from_data_source(tree).get_interfaces(tree)


def read_sweep_context(
    data: xr.DataTree, *, block: int = -1, alias: str | None = None
) -> SweepReduceContext:
    """Read saved sweep execution metadata without restoring live context data.

    Parameters
    ----------
    data : xarray.DataTree
        Open interface or block tree.
    block : int, optional
        Block to inspect in an interface tree; -1 selects the last block.
        An execution record on the supplied node is read directly.
    alias : str or None, optional
        Invocation label; None selects the primary invocation.

    Returns
    -------
    SweepReduceContext
        Detached execution metadata with data=None.
    """
    key = SweepReduce.get_context_key(alias=alias)
    if "tolteca_kids" not in data.children or key not in data["tolteca_kids"].attrs:
        data = select_block(data, block).tree
    return SweepReduce.load_context(data["tolteca_kids"].attrs[key])


def read_timestream_context(
    data: xr.DataTree, *, block: int = -1, alias: str | None = None
) -> TimestreamReduceContext:
    """Read saved timestream execution metadata without restoring live data.

    Parameters
    ----------
    data : xarray.DataTree
        Open interface or block tree.
    block : int, optional
        Block to inspect in an interface tree; -1 selects the last block.
        An execution record on the supplied node is read directly.
    alias : str or None, optional
        Invocation label; None selects the primary invocation.

    Returns
    -------
    TimestreamReduceContext
        Detached execution metadata with data=None.
    """
    key = TimestreamReduce.get_context_key(alias=alias)
    if "tolteca_kids" not in data.children or key not in data["tolteca_kids"].attrs:
        data = select_block(data, block).tree
    return TimestreamReduce.load_context(data["tolteca_kids"].attrs[key])


def _product_tree(data, config):
    from .dataprod_output import DataProdOutput  # noqa: PLC0415 -- import cycle

    product = data.toltec_kids._get_export_tree(include_raw=True)
    for node, records in _step_records(data):
        target = product[node.relative_to(data)]
        # A post-processing step may start from a reopened scientific product.
        # Preserve earlier execution records while replacing steps rerun here.
        saved = {
            key: value
            for key, value in (
                target["tolteca_kids"].attrs.items()
                if "tolteca_kids" in target.children
                else ()
            )
            if ":" in key and isinstance(value, str)
        }
        target["tolteca_kids"] = xr.DataTree(
            dataset=xr.Dataset(
                attrs=saved
                | {
                    key: context.step_cls.dump_context(context)
                    for key, context in records.items()
                    if context.step_cls is not DataProdOutput
                }
            )
        )
    for node in product.subtree:
        node.dataset = node.dataset.drop_encoding()
        node.attrs.pop(PIPELINE_CONTEXT_KEY, None)
    _set_product_chunks(product, config)
    return product


def _set_product_chunks(tree, config):
    """Apply output chunk sizes through the acquisition and sweep mappings."""
    check_mapper = SweepCheckMapper.from_defaults()
    check_schema = check_mapper.schema
    for node in tree.subtree:
        if check_mapper.has_attr(node, check_schema.config):
            chan_dim = check_mapper.get_name(check_schema.chan_dim)
            for field in (
                check_schema.samples_subpath,
                check_schema.channels_subpath,
                check_schema.chunks_subpath,
            ):
                _set_array_chunks(
                    check_mapper.get_node(node, field),
                    {chan_dim: config.n_chans_per_chunk},
                    chan_dim=chan_dim,
                )
        kind = node.attrs.get(NODE_TYPE)
        if kind in {"acquisition", "block"}:
            tree_mapper = ToltecKidsTreeMapper.from_data_source(node)
            raw = tree_mapper.get_node(node, tree_mapper.schema.acquisition)
            mapper = ToltecKidsIOMapper.from_data_source(raw)
            chan_dim = mapper.get_name(mapper.schema.n_chans)
            _set_array_chunks(
                raw,
                {
                    chan_dim: config.n_chans_per_chunk,
                    mapper.get_name(mapper.schema.n_times): config.n_samples_per_chunk,
                },
                chan_dim=chan_dim,
            )
        elif kind == "sweep":
            mapper = ToltecSweepMapper.from_data_source(node)
            chan_dim = mapper.get_arr(node, mapper.schema.f_chan).dims[0]
            _set_array_chunks(
                node, {chan_dim: config.n_chans_per_chunk}, chan_dim=chan_dim
            )


def _set_array_chunks(node, chunks, *, chan_dim):
    """Set physical chunks without fragmenting small channel coordinates."""
    ds = node.to_dataset(inherit=False)
    for name, variable in ds.variables.items():
        if not set(variable.dims).intersection(chunks):
            continue
        if variable.dims == (chan_dim,):
            continue
        sizes = {
            dim: min(size, chunks.get(dim, size))
            for dim, size in variable.sizes.items()
        }
        if variable.chunks is not None:
            ds[name] = ds[name].chunk(sizes)
        ds[name].encoding["chunks"] = tuple(sizes.values())
    node.dataset = ds


def _input_provenance(data, workflow_config):
    attrs = data["tolteca_kids"].attrs if "tolteca_kids" in data.children else {}
    return KidsInputProvenance(
        uid_raw_obs_file=data.toltec_kids.meta.uid_raw_obs_file,
        source=attrs.get("source"),
        source_loc=attrs.get("source_loc"),
        config=attrs.get("config", workflow_config),
    )


def _step_records(tree):
    """Visit execution records without adding contexts to scientific leaves."""
    for node in tree.subtree:
        records = node.attrs.get(PIPELINE_CONTEXT_KEY)
        if records:
            yield node, records


def _validate_saved(original, saved):
    for node, records in _step_records(original):
        target = saved[node.relative_to(original)]
        for key, context in records.items():
            if key not in target["tolteca_kids"].attrs:
                continue
            restored = context.step_cls.load_context(target["tolteca_kids"].attrs[key])
            if restored.model_dump() != context.model_dump():
                msg = "Step context did not round-trip"
                raise ValueError(msg)
            if context.completed and isinstance(
                context, (SweepReduceContext, TimestreamReduceContext)
            ):
                assert context.data is not None
                view = context.step_cls.view(
                    target, block=context.block, alias=context.config.alias
                )
                if dict(view.data_source.sizes) != dict(context.data.data_source.sizes):
                    msg = "Reduced array dimensions did not round-trip"
                    raise ValueError(msg)


def make_kids_product_index(
    master: str,
    obsnum: int,
    subobsnum: int,
    scannum: int,
    *,
    run: WorkflowRun | None = None,
) -> DataProdIndex:
    """Describe a processed acquisition before reserving its output run.

    Parameters
    ----------
    master : str
        Acquisition master, such as ``tcs`` or ``ics``.
    obsnum, subobsnum, scannum : int
        Acquisition quartet values from the input metadata.
    run : WorkflowRun or None, optional
        Producing execution. Required before materializing the index.

    Returns
    -------
    DataProdIndex
        Processed product identity and primary raw-product association, without
        artifact sources. The output layer supplies the producing run before
        materialization.
    """
    uid = make_raw_obs_product_uid(master, obsnum, subobsnum, scannum)
    return DataProdIndex(
        run=run,
        product=DataProd(
            uid=f"{uid}_proc",
            meta=RawObsProcMeta(
                tag="raw_obs_proc",
                data_prod_type="dp_raw_obs_proc",
                name=f"{uid}_proc",
                master=master,
                obsnum=obsnum,
                subobsnum=subobsnum,
                scannum=scannum,
            ),
            assocs=[
                DataProdAssoc(
                    data_prod_assoc_type="dpa_raw_obs_proc_raw_obs",
                    uid=uid,
                    is_primary=True,
                )
            ],
        ),
    )


def write_kids_product(  # noqa: PLR0913 -- Independent publication inputs.
    data: xr.DataTree | Mapping[str, xr.DataTree],
    store: DataProdStore | Path,
    *,
    output_config: DataProdOutputConfig | None = None,
    output_index: DataProdIndex | None = None,
    run: WorkflowRun | None = None,
    workflow_config: KidsConfig | None = None,
    artifacts: ArtifactContext | None = None,
) -> Path:
    """Publish selected interfaces of one acquisition as a processed product.

    Parameters
    ----------
    data : xarray.DataTree or mapping of str to xarray.DataTree
        Prepared scientific trees, optionally keyed by interface. The caller
        keeps their input resources open until publication finishes.
    store : DataProdStore or pathlib.Path
        Store or root for the default namespace.
    output_config : DataProdOutputConfig or None, optional
        Output settings; defaults to automatic allocation and tag ``main``.
    output_index : DataProdIndex or None, optional
        Bound index from an active store.materialize scope. None opens a scope
        for export. With a supplied index, the caller completes materialization
        and the returned path is its temporary directory until scope exit.
    run : WorkflowRun or None, optional
        Caller-owned execution when output_index is omitted. The caller saves
        its final outcome. None creates a standalone execution.
    workflow_config : KidsConfig or None, optional
        Resolved workflow used for these inputs. Per-input snapshots retained by
        the runner take precedence. None records no full workflow snapshot for
        independently invoked steps; their execution records remain available.
    artifacts : ArtifactContext or None, optional
        Prepared acquisition artifacts. Exported in the same atomic publication.

    Returns
    -------
    pathlib.Path
        Pinned run directory containing array.zarr and
        index.yaml. The product inventory records the run and requested tag.

    Raises
    ------
    ValueError
        Interfaces describe different acquisitions or saved data fails validation.
    FileExistsError
        The requested run UID, name or counter value is already reserved.
    """
    config = output_config or DataProdOutputConfig()
    if not isinstance(store, DataProdStore):
        store = DataProdStore(store, namespace=config.namespace)
    config = config.model_copy(
        update={"path": store.path.resolve(), "namespace": store.namespace}
    )
    trees = (
        {data.toltec_kids.meta.interface: data}
        if isinstance(data, xr.DataTree)
        else dict(data)
    )
    metadata = {name: tree.toltec_kids.meta for name, tree in trees.items()}
    uids = {
        make_raw_obs_product_uid(m.master, m.obsnum, m.subobsnum, m.scannum)
        for m in metadata.values()
    }
    if len(uids) != 1 or any(name != meta.interface for name, meta in metadata.items()):
        msg = "A KIDs product requires interfaces of one acquisition"
        raise ValueError(msg)
    first = next(iter(metadata.values()))
    with ExitStack() as stack:
        if output_index is None:
            if run is None:
                run = WorkflowRun.create()
                stack.callback(store.write_run, run)
                stack.enter_context(run)
            index = make_kids_product_index(
                first.master, first.obsnum, first.subobsnum, first.scannum, run=run
            )
            output_index = stack.enter_context(
                store.materialize(index, run_name=config.run_name, tag=config.tag)
            )
        _write_artifacts(
            trees,
            output_index,
            config,
            workflow_config=workflow_config,
            prepared=artifacts,
        )
    assert output_index.base_path is not None
    return output_index.base_path


def tone_prep_inputs(tree):
    """Return the exact saved DriveFit input used by a block's TonePrep.

    Parameters
    ----------
    tree : xarray.DataTree
        Computed block containing an optional TonePrep context.

    Returns
    -------
    list of dict
        Product identity and resolved index URI, or an empty list when no
        DriveFit was consumed. A moved selection is rejected, not reselected.
    """
    from .tone_prep import TonePrep  # noqa: PLC0415 -- configuration cycle

    records = []
    for context in Pipeline.get_contexts(tree).values():
        if context.step_cls is not TonePrep or context.data is None:
            continue
        reference = context.data.data.attrs.get("reference_product")
        if reference is None:
            continue
        reference = DataProdRef.model_validate(reference)
        source = resolve_index(context.config.drive_fit_reference)
        if DataProdIndex.read(source).product.reference != reference:
            msg = "DriveFit selection changed after TonePrep; retain the exact input"
            raise ValueError(msg)
        record = {
            "product": reference.model_dump(mode="json"),
            "source": fileloc(source).url,
        }
        if record not in records:
            records.append(record)
    return records


def _write_artifacts(trees, index, config, *, workflow_config, prepared):
    """Write arrays and artifact receipts into a destination-bound index."""
    path = index.base_path
    assert path is not None
    assert index.run is not None
    arrays = "array.zarr"
    artifacts = []
    for relative, spec, data, meta in _collect_artifacts(trees, prepared, config):
        for item in spec.write(data, path=path / relative):
            filename = item.path.relative_to(path).as_posix()
            artifacts.append(
                (
                    filename,
                    {
                        "format": item.format,
                        "name": filename if item.name is None else item.name,
                        **meta,
                    },
                )
            )
    with xr.open_datatree(path / arrays, engine="zarr", consolidated=True) as saved:
        groups = _ProductMapper.from_data_source(saved).get_interfaces(saved)
        for name, tree in trees.items():
            _validate_saved(tree, saved[groups[name]])

    # A component reference selects the saved analysis, including joint fits.
    # These receipts address subtrees of the existing store; no arrays are copied.
    for interface, tree in trees.items():
        blocks = (
            [tree.toltec_kids.block]
            if tree.attrs.get(NODE_TYPE) == "block"
            else tree.toltec_kids.blocks
        )
        for block in blocks:
            for record in tone_prep_inputs(block.tree):
                edge = DataProdAssoc(
                    **record["product"],
                    data_prod_assoc_type="dpa_raw_obs_proc_data_prod",
                )
                if edge not in index.product.assocs:
                    index.product.assocs.append(edge)
            scope = block.tree.relative_to(tree)
            for name, node in block.tree.children.items():
                if name.startswith("model_fit") and "plan_source" in node.attrs:
                    artifacts.append(
                        (
                            (Path(arrays) / interface / scope / name).as_posix(),
                            {
                                "format": "zarr",
                                "name": name,
                                "producer": "tolteca_kids.model_fit:ModelFit",
                                "interface": interface,
                                "scope": str(scope),
                            },
                        )
                    )
    provenance = KidsRunProvenance(
        inputs={
            name: _input_provenance(tree, workflow_config)
            for name, tree in trees.items()
        },
        output=config,
        software={
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
        runtime_info=index.run.runtime_info.model_copy(
            update={"config_sources": None, "validation_context": {}},
        ),
    )
    index.provenance = provenance.model_dump(mode="json", round_trip=True)
    index.locations["self"] = Location(root=".", location_type="local")
    index.product.data_items.extend(
        DataProdItem(
            meta=ArtifactMeta.model_validate(
                {"type": "artifact", **meta, "run_uid": index.run.uid}
            ).model_dump(mode="json"),
            source=DataProdSource.model_validate(
                {
                    "path": filename,
                    "location": "self",
                    "availability_state": "available",
                }
            ),
        )
        for filename, meta in artifacts
    )


_artifact_path_segment = TypeAdapter(PathSegment)


def _collect_artifacts(trees, prepared, config):
    """Collect retained objects and effective writers with their output scopes."""
    scientific = KidsDataArtifact()
    yield (
        Path(),
        scientific,
        scientific(trees, config=config),
        {"producer": KidsDataArtifact.get_context_key()},
    )
    for interface, tree in trees.items():
        for node, records in _step_records(tree):
            for key, context in records.items():
                spec = context.config.artifacts
                if not context.completed or not context.artifacts or spec is None:
                    continue
                relative = (
                    Path("artifacts")
                    / interface
                    / node.relative_to(tree)
                    / _artifact_path_segment.validate_python(key)
                )
                yield (
                    relative,
                    spec,
                    context.artifacts,
                    {
                        "interface": interface,
                        "scope": str(node.relative_to(tree)),
                        "producer": key,
                    },
                )
    for key, artifact in (prepared or {}).items():
        yield (
            Path("artifacts") / _artifact_path_segment.validate_python(key),
            artifact.spec,
            artifact.data,
            {"producer": key},
        )


def write_sweep_product(
    data: xr.DataTree,
    path: DataProdStore | Path,
    *,
    output_config: DataProdOutputConfig | None = None,
) -> Path:
    """Publish a completed sweep through a data-product store.

    Parameters
    ----------
    data : xarray.DataTree
        Tree with a completed SweepReduce invocation.
    path : DataProdStore or pathlib.Path
        Store or root; the acquisition determines the product directory.
    output_config : DataProdOutputConfig or None, optional
        Namespace and run settings, or defaults when None.

    Returns
    -------
    pathlib.Path
        Pinned run directory.
    """
    scopes = (
        [data]
        if SweepReduce.has_context(data)
        else [block.tree for block in data.toltec_kids.blocks]
    )
    for scope in scopes:
        context = SweepReduce.get_context(scope)
        if not context.completed or context.data is None:
            msg = "Sweep reduction is not completed"
            raise ValueError(msg)
    return write_kids_product(data, path, output_config=output_config)


def write_timestream_product(
    data: xr.DataTree,
    path: DataProdStore | Path,
    *,
    output_config: DataProdOutputConfig | None = None,
) -> Path:
    """Publish a completed timestream through a data-product store.

    Parameters
    ----------
    data : xarray.DataTree
        Tree with a completed TimestreamReduce invocation.
    path : DataProdStore or pathlib.Path
        Store or root; the acquisition determines the product directory.
    output_config : DataProdOutputConfig or None, optional
        Namespace and run settings, or defaults when None.

    Returns
    -------
    pathlib.Path
        Pinned run directory.
    """
    scopes = (
        [data]
        if TimestreamReduce.has_context(data)
        else [block.tree for block in data.toltec_kids.blocks]
    )
    for scope in scopes:
        context = TimestreamReduce.get_context(scope)
        if not context.completed or context.data is None:
            msg = "Timestream reduction is not completed"
            raise ValueError(msg)
    return write_kids_product(data, path, output_config=output_config)


def open_kids_product(
    path: Path | DataProdIndex,
    *,
    interface: str | None = None,
    location: str | None = None,
    cache: Path | None = None,
    create_default_indexes: bool = True,
) -> xr.DataTree:
    """Open one published interface tree with lazy arrays.

    Parameters
    ----------
    path : pathlib.Path or DataProdIndex
        DB descriptor, product directory or YAML run index.
    interface : str or None, optional
        Interface to select; required when the product has more than one.
    location : str or None, optional
        Registered location key. Required when several Zarr copies are listed.
    cache : pathlib.Path or None, optional
        Persistent cache for immutable remote Zarr objects. None disables the
        disk cache. Local artifacts are read directly.
        Separate processes can share a local HTTP cache. Interrupted downloads
        are retried on the next read; cached selections remain available offline.
    create_default_indexes : bool, optional
        Build coordinate indexes for label-based selection, default True. False
        keeps dimension coordinates lazy; use positional selection or explicitly
        create an index before label-based selection.

    Returns
    -------
    xarray.DataTree
        Scientific tree. Close it after reading the required arrays.
    """
    index = _product_index(path)
    if interface is None:
        groups = read_product_interfaces(index, location=location, cache=cache)
        if len(groups) != 1:
            msg = f"Select an interface: {tuple(groups)}"
            raise ValueError(msg)
        interface = next(iter(groups))
    return _open_product_tree(
        index,
        location=location,
        cache=cache,
        group=interface,
        create_default_indexes=create_default_indexes,
    )


def _open_reduced_product(path, interface, step, load_context, **options):
    tree = open_kids_product(path, interface=interface, **options)
    try:
        context = load_context(tree)
        if not context.completed:
            msg = "Incomplete reduction"
            raise ValueError(msg)  # noqa: TRY301 -- close tree on failure
        _ = step.view(tree, block=context.block, alias=context.config.alias).data_source
    except BaseException:
        tree.close()
        raise
    return tree


def open_sweep_product(
    path: Path | DataProdIndex,
    *,
    interface: str | None = None,
    location: str | None = None,
    cache: Path | None = None,
) -> xr.DataTree:
    """Open a published sweep and validate its saved reduction metadata.

    Parameters
    ----------
    path : pathlib.Path or DataProdIndex
        DB descriptor, product directory or run index.
    interface : str or None, optional
        Interface to select for multi-interface products.
    location : str or None, optional
        Registered location key when the descriptor lists several Zarr copies.
    cache : pathlib.Path or None, optional
        Persistent cache for immutable remote Zarr objects; None disables it.

    Returns
    -------
    xarray.DataTree
        Lazy scientific tree; the caller closes it after use.
    """
    return _open_reduced_product(
        path, interface, SweepReduce, read_sweep_context, location=location, cache=cache
    )


def open_timestream_product(
    path: Path | DataProdIndex,
    *,
    interface: str | None = None,
    location: str | None = None,
    cache: Path | None = None,
) -> xr.DataTree:
    """Open a published timestream and validate its saved reduction metadata.

    Parameters
    ----------
    path : pathlib.Path or DataProdIndex
        DB descriptor, product directory or run index.
    interface : str or None, optional
        Interface to select for multi-interface products.
    location : str or None, optional
        Registered location key when the descriptor lists several Zarr copies.
    cache : pathlib.Path or None, optional
        Persistent cache for immutable remote Zarr objects; None disables it.

    Returns
    -------
    xarray.DataTree
        Lazy scientific tree; the caller closes it after use.
    """
    return _open_reduced_product(
        path,
        interface,
        TimestreamReduce,
        read_timestream_context,
        location=location,
        cache=cache,
    )
