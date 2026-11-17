"""Materialize and select data products in a local namespace.

Write artifacts within a destination-bound index::

    with WorkflowRun.create() as run:
        index = make_index(run=run)
        with store.materialize(index) as output_index:
            write_artifacts(output_index)
        path = output_index.base_path
    store.write_run(run)

A run records completion or failure when its context exits. Materialization holds
the requested tag until its context exits. The store declares the generated index;
producers declare their artifact items. An installed version is immutable.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp
from typing import TYPE_CHECKING
from uuid import UUID

from filelock import FileLock, Timeout
from pydantic import TypeAdapter
from tollan.config import RuntimeInfo
from tollan.config.types import PathSegment
from tollan.utils.fileloc import fileloc

from ..core.data_prod import (
    CounterRunNameGenerator,
    DataProdAssoc,
    DataProdAssocRun,
    DataProdItem,
    DataProdNamespaceIndex,
    DataProdRef,
    DataProdSource,
    DataProdStoreBase,
    DataProdTag,
    Location,
    RunName,
)
from ..core.data_prod._io import _read_yaml, _sync_directory, _write_yaml
from ..core.data_prod._primary import _select_primary_assoc
from ..core.data_prod.store import _audit, _select_namespace_actor
from ._version_publication import _append_version_member, _compact_version_group
from ._versions import GroupHeader, parse_index_document
from .identity import _normalize_entity_uid, normalize_primary_uid
from .index import DataProd, DataProdIndex
from .metadata import VersionedProductMeta, VersionGroupMeta
from .types import (
    ASSOC_TARGET_TYPES,
    LOGICAL_PRODUCT_TYPES,
    PRIMARY_PRODUCT_TYPES,
    VERSION_GROUP_PRIMARY_ASSOC_TYPES,
    NamedGroupPath,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["DataProdStore", "DataProdTagBusyError"]

_DESCRIPTOR_PRODUCT_TYPES = LOGICAL_PRODUCT_TYPES - {"dp_version_group"}


_validate_path_segment = TypeAdapter(PathSegment).validate_python
_validate_run_name = TypeAdapter(RunName).validate_python


class DataProdStore(DataProdStoreBase):
    """Materialize products under one relocatable namespace.

    Parameters
    ----------
    path : str or pathlib.Path
        Product root, usually ``dataprod_toltec``.
    namespace : str or pathlib.PurePosixPath, optional
        Relative namespace path, default ``scratch``. Construction creates no files.
    """

    def __init__(
        self, path: str | Path, *, namespace: str | PurePosixPath = "scratch"
    ) -> None:
        super().__init__(path, namespace=namespace)

    def get_product_path(
        self, primary_uid: str, data_prod_type: str, *, uid: str | None = None
    ) -> Path:
        """Locate a product directory without creating it.

        Parameters
        ----------
        primary_uid : str
            Canonical observation, acquisition or named-collection UID.
            Observation directories omit TCS. A collection UUID resolves through
            the namespace's saved primary_paths.
        data_prod_type : str
            Managed product type.
        uid : str or None, optional
            Logical group UID for grouped products. Observation and raw-processed
            products use their type name. Named-primary results omit the
            ``<primary UUID>_`` prefix; other supplied UIDs name the directory.

        Returns
        -------
        pathlib.Path
            Namespace, primary label and product directory.

        Raises
        ------
        ValueError
            The primary identifier is invalid or its named-group path is absent.
        """
        primary_uid = normalize_primary_uid(primary_uid)
        primary, named = self._primary_path(primary_uid)
        name = data_prod_type
        if uid is not None and data_prod_type not in {
            "dp_raw_obs_proc",
            "dp_raw_obs",
            "dp_obs",
            "dp_named_group",
        }:
            name = (
                uid.removeprefix(primary_uid + "_")
                if named
                else uid.replace(primary_uid, primary)
            )
        return self.namespace_path / primary / _validate_path_segment(name)

    def _primary_path(self, primary_uid):
        # A primary is a named collection when the namespace records its path;
        # otherwise it must be an observation or acquisition.
        try:
            paths = self.read_namespace().primary_paths
        except FileNotFoundError:
            paths = {}
        recorded = {str(uid): path for uid, path in paths.items()}
        if primary_uid in recorded:
            path = TypeAdapter(NamedGroupPath).validate_python(recorded[primary_uid])
            return path, True
        try:
            uid = _normalize_entity_uid(primary_uid)
        except ValueError:
            msg = (
                f"Unknown primary {primary_uid!r}: write the named-group descriptor "
                "with primary_path first"
            )
            raise ValueError(msg) from None
        return uid.removeprefix("tcs-"), False

    def _bind_primary_path(self, primary_uid, path):
        path = TypeAdapter(NamedGroupPath).validate_python(path)
        self.ensure_namespace()
        with FileLock(self.namespace_path / ".namespace.lock"):
            descriptor = self.read_namespace()
            uid = UUID(primary_uid)
            current = descriptor.primary_paths.get(uid)
            if current is not None and current != path:
                msg = (
                    "Primary path already assigned; use a new projection to relocate it"
                )
                raise ValueError(msg)
            updated = DataProdNamespaceIndex.model_validate(
                {
                    **descriptor.model_dump(),
                    "primary_paths": {**descriptor.primary_paths, uid: path},
                }
            )
            if current is None and (self.namespace_path / path).exists():
                msg = f"Primary path already exists: {path}"
                raise FileExistsError(msg)
            _write_yaml(updated, self.namespace_path / "namespace.yaml", overwrite=True)

    def get_index_path(self, primary_uid, data_prod_type, *, uid=None, run_name=None):
        """Locate a descriptor, version-group index or realized-product index.

        Parameters
        ----------
        primary_uid : str
            Canonical observation, acquisition or named-collection UID.
        data_prod_type : str
            Managed product type. Logical descriptors (dp_obs, dp_raw_obs,
            dp_cal_group and dp_named_group) use ``<type>.yaml``; other types use
            ``<type>/index.yaml`` beneath the primary directory.
        uid : str or None, optional
            Logical result UID when its directory is named by the producer.
        run_name : str or None, optional
            Realization directory beneath the product directory, default None.
            Only processed products have run directories.

        Returns
        -------
        pathlib.Path
            Destination without creating directories.

        Raises
        ------
        ValueError
            A run name is supplied for a logical descriptor.
        """
        path = self.get_product_path(primary_uid, data_prod_type, uid=uid)
        if data_prod_type in _DESCRIPTOR_PRODUCT_TYPES:
            if run_name is not None:
                msg = "Logical descriptors do not have run indexes"
                raise ValueError(msg)
            return path.with_name(path.name + ".yaml")
        if run_name is not None:
            return path / _validate_run_name(run_name) / "index.yaml"
        return path / "index.yaml"

    def write_index(self, index, *, primary_uid, primary_path=None, overwrite=False):
        """Save an existing product's index without copying its payloads.

        Logical descriptors (dp_obs, dp_raw_obs, dp_cal_group and dp_named_group) write
        ``<type>.yaml`` beneath the primary directory. Version groups write
        ``index.yaml`` in their group directory; processed versions write it
        beneath their run name.

        Parameters
        ----------
        index : DataProdIndex
            Product description and its supporting records. Processed versions
            must already carry their namespace and producing run.
        primary_uid : str
            Terminal observation, acquisition or named-collection UID.
        primary_path : str or None, optional
            Readable directory for a named-group descriptor's first write. Later
            writes reuse its recorded placement. Display-name changes do not move
            files. To relocate, change the placement in a new namespace projection.
            None is the default; observations use their established directory names.
        overwrite : bool, optional
            Replace the destination YAML file atomically when True. False
            (default) raises FileExistsError if it already exists. Replacement
            does not merge index contents or replace artifact payloads.

        Returns
        -------
        pathlib.Path
            Written index filename. Product identity is preserved.

        Raises
        ------
        FileExistsError
            The destination index exists and overwrite is False.
        ValueError
            A processed product lacks its namespace or producing run, or the
            primary placement is absent, invalid or already assigned differently.
        """
        kind = index.product.meta.data_prod_type
        if primary_path is not None and kind != "dp_named_group":
            msg = "primary_path is only supplied with a named-group descriptor"
            raise ValueError(msg)
        if (
            kind in PRIMARY_PRODUCT_TYPES
            and normalize_primary_uid(primary_uid) != index.product.uid
        ):
            msg = "A logical primary descriptor must be stored under its own UID"
            raise ValueError(msg)
        if primary_path is not None:
            self._bind_primary_path(index.product.uid, primary_path)
        if kind in _DESCRIPTOR_PRODUCT_TYPES:
            destination = self.get_index_path(primary_uid, kind)
        else:
            if kind != "dp_version_group" and (
                index.namespace is None or index.run is None
            ):
                msg = "Use materialize() before exporting a product version"
                raise ValueError(msg)
            if index.run is None:
                group_uid = index.product.uid
            elif (
                isinstance(index.product.meta, VersionedProductMeta)
                and index.product.meta.version_group is not None
            ):
                group_uid = index.product.meta.version_group.uid
            else:
                msg = "A product version must record its version group"
                raise ValueError(msg)
            destination = self.get_product_path(
                primary_uid,
                getattr(index.product.meta, "content_data_prod_type", kind),
                uid=group_uid,
            )
            if index.run is not None:
                destination /= index.store_run.name
        snapshot = index.model_copy(deep=True)
        name = destination.name if destination.suffix == ".yaml" else "index.yaml"
        namespace_uid = self.ensure_namespace().namespace.uid
        if kind in _DESCRIPTOR_PRODUCT_TYPES:
            _descriptor_item(snapshot, destination, self.namespace_path, namespace_uid)
        else:
            _include_index_item(snapshot, name, namespace_uid)
        return snapshot.write(destination, overwrite=overwrite)

    @contextmanager
    def materialize(
        self,
        index: DataProdIndex,
        *,
        run_name: RunName | CounterRunNameGenerator | None = None,
        tag: str | None = "main",
    ) -> Iterator[DataProdIndex]:
        """Write artifacts and an index as one completed product version.

        Parameters
        ----------
        index : DataProdIndex
            Reusable product description with a producing run and primary
            association. Its UID identifies the version group. Relative artifact
            locations are resolved in the temporary output directory. The input
            object remains unchanged; retained artifact objects stay with the
            workflow output step.
        run_name : str or CounterRunNameGenerator or None, optional
            Namespace-local naming policy. None uses index.run.uid. Reservations
            remain consumed after failure.
        tag : str or None, optional
            Group selection to hold and advance, default "main". None leaves all
            selections unchanged.

        Yields
        ------
        DataProdIndex
            Independent destination-bound index. Write artifacts beneath its
            base_path and add their receipts as data items. Successful exit writes
            index.yaml, installs the output and advances the tag. Its base_path
            then resolves the installed directory. There is no explicit commit.

        Raises
        ------
        ValueError
            The index describes a logical product or a version group, execution
            or primary identity is missing, the supplied namespace conflicts with
            the store, or bound identity changes inside the scope.
        DataProdTagBusyError
            Another writer owns this product's requested tag.
        FileExistsError
            The local name or product realization is already reserved.

        Notes
        -----
        Keep lazy artifact sources open until the scope exits. An exception
        discards temporary output and leaves the prior tag selection unchanged.
        A normal exit materializes the supplied description, including an empty
        artifact inventory when that is the intended product. Output steps must
        signal incomplete computation before normal exit.
        """
        kind = index.product.meta.data_prod_type
        if kind in LOGICAL_PRODUCT_TYPES:
            msg = (
                "A version group forms from its materialized members"
                if kind == "dp_version_group"
                else "Logical descriptors use write_index(), without a producing run"
            )
            raise ValueError(msg)
        if index.run is None:
            msg = "Materialization requires index.run"
            raise ValueError(msg)
        primary_uid, primary_type = _primary(index)
        run = self._reserve_run(index.run, run_name)
        descriptor = self.read_namespace()
        if (
            index.namespace is not None
            and index.namespace.uid != descriptor.namespace.uid
        ):
            msg = "Index namespace differs from the destination store"
            raise ValueError(msg)
        product = self.get_product_path(
            primary_uid, kind, uid=index.product.uid
        ).resolve()
        product.mkdir(parents=True, exist_ok=True)
        with _tag_lock(product, None if tag is None else _validate_path_segment(tag)):
            run._request(index.product.uid, kind, primary_uid, tag)
            with FileLock(product / ".publish.lock"):
                root_path = product / "index.yaml"
                current = (
                    GroupHeader.load(root_path)
                    if root_path.exists()
                    else GroupHeader.from_index(
                        _version_group(
                            index, primary_uid, primary_type, descriptor, run.actor
                        )
                    )
                )
                if (
                    current.product.uid != index.product.uid
                    or not isinstance(current.product.meta, VersionGroupMeta)
                    or current.product.meta.content_data_prod_type != kind
                ):
                    msg = "Materialization would replace a different version group"
                    raise ValueError(msg)
                _compact_version_group(current, product)
                if tag is not None and tag not in current.product.tags:
                    current.product.tags[tag] = DataProdTag(audit=_audit(run.actor))
                current.write(root_path, overwrite=True)
            staging = product / ".staging"
            staging.mkdir(exist_ok=True)
            path = Path(mkdtemp(dir=staging))
            pending = _Materialization(path, product, index, run, tag)
            try:
                yield pending.index
                pending.finish()
            except BaseException as exc:
                if not pending.completed:
                    run._record(index.product.uid, "failed", str(exc))
                raise
            finally:
                if path.exists():
                    shutil.rmtree(path)
                if not pending.completed:
                    pending.index._bind_locations(None)

    def reconcile_publication(self, run, group_uid, group_directory):
        """Recover one interrupted publication from its committed group membership.

        The caller must have exclusive ownership of the interrupted run.
        A retry can acquire that ownership with the workflow's collection lock.
        Return the committed member path, or None when publication did not commit.
        The caller owns finalization of the execution, which can have other outputs.
        """
        directory = Path(group_directory)
        root = directory / "index.yaml"
        path = None
        if root.is_file():
            group = DataProdIndex.read(root, run_uids=[run.uid])
            if group.product.uid != group_uid:
                msg = "Recovery directory belongs to another product group"
                raise ValueError(msg)
            members = [p for p in group.products if p.run and p.run.uid == run.uid]
            if members:
                (member,) = members
                if member.store_run is None:
                    msg = "Committed member has no store run"
                    raise ValueError(msg)
                path = directory / member.store_run.name / "index.yaml"
        if run.status == "running":
            descriptor = self.read_namespace_index(run_uids=[run.uid])
            stored = descriptor.store_runs.get(run.uid)
            publication = stored.publications.get(group_uid) if stored else None
            if publication is not None and publication.status == "pending":
                writer = self._reserve_run(run)
                writer._record(group_uid, "completed" if path else "failed")
        return path

    def set_tag(self, primary_uid, data_prod_type, run_name, *, uid=None, tag="main"):
        """Select a completed realization without changing its artifacts.

        Attribute the change to the current environment's actor, reusing a
        matching actor recorded in the namespace.

        Parameters
        ----------
        primary_uid, data_prod_type : str
            Product placement identifiers.
        run_name : str
            Completed realization's namespace-unique run name.
        uid : str or None, optional
            Logical result UID; required for a producer-named result directory.
        tag : str, optional
            Selection label, default ``main``.

        Returns
        -------
        pathlib.Path
            Selected realization index.
        """
        product = self.get_product_path(primary_uid, data_prod_type, uid=uid).resolve()
        with (
            _tag_lock(product, _validate_path_segment(tag)),
            FileLock(product / ".publish.lock"),
        ):
            path = product / "index.yaml"
            index = GroupHeader.load(path)
            _compact_version_group(index, product)
            member_path = product / _validate_run_name(run_name) / "index.yaml"
            if not member_path.is_file():
                raise KeyError(run_name)
            member = DataProdIndex.read(member_path)
            if (
                member.run is None
                or index.lookup(product, member.product.uid).name != run_name
            ):
                raise KeyError(run_name)
            # Promotion attribution uses the current environment, not the producer.
            with FileLock(self.namespace_path / ".namespace.lock"):
                descriptor = self.read_namespace()
                actor = _select_namespace_actor(
                    RuntimeInfo(), descriptor.actors.values()
                )
                if actor.uid not in descriptor.actors:
                    descriptor.actors[actor.uid] = actor
                    _write_yaml(
                        descriptor,
                        self.namespace_path / "namespace.yaml",
                        overwrite=True,
                    )
            index.actors[actor.uid] = actor
            previous = index.product.tags.get(tag)
            index.product.tags[tag] = DataProdTag(
                current=DataProdRef(
                    uid=member.product.uid, namespace_uid=member.product.namespace_uid
                ),
                audit=_audit(actor, previous.audit if previous else None),
            )
            index.product.audit = _audit(actor, index.product.audit)
            index.write(path, overwrite=True)
        return product / run_name / "index.yaml"

    @staticmethod
    def resolve_index(path, *, tag="main"):
        """Select a version-group tag or retain an exact product index.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML index or directory containing one.
        tag : str, optional
            Group selection, default ``main``.

        Returns
        -------
        pathlib.Path
            Pinned product index.

        Raises
        ------
        KeyError
            The tag has no completed selection.
        """
        path = Path(path)
        if path.is_dir():
            path /= "index.yaml"
        document = parse_index_document(_read_yaml(path))
        if isinstance(document, GroupHeader):
            group = document
            current = group.product.tags[tag].current
            if current is None:
                raise KeyError(tag)
            member = group.lookup(path.parent, current.uid)
            if (member.edge.uid, member.edge.namespace_uid) != (
                current.uid,
                current.namespace_uid,
            ):
                msg = "Tag target differs from version membership"
                raise ValueError(msg)
            return path.parent / member.name / "index.yaml"
        index = document
        index._bind_locations(path.parent.resolve())
        if index.product.meta.data_prod_type != "dp_version_group":
            return path
        selected = index.product.tags[tag].current
        product = next(
            (
                p
                for p in index.products
                if selected is not None
                and p.product.uid == selected.uid
                and p.product.namespace_uid == selected.namespace_uid
            ),
            None,
        )
        if product is None:
            raise KeyError(tag)
        item = next(
            i
            for i in product.product.data_items
            if i.meta.get("format") == "data_prod_index"
        )
        return fileloc(product.get_source_uri(item.source)).path


def _primary(index):
    """Resolve a terminal UID and type from records or typed acquisition edges."""
    products = {p.product.reference: p.product for p in index.products}
    product, visited = index.product, set()
    while product.reference not in visited:
        visited.add(product.reference)
        kind = product.meta.data_prod_type
        if kind in PRIMARY_PRODUCT_TYPES:
            return product.uid, kind
        association = _select_primary_assoc(product.uid, product.assocs)
        reference = DataProdRef(
            uid=association.uid, namespace_uid=association.namespace_uid
        )
        try:
            target_type = ASSOC_TARGET_TYPES[association.data_prod_assoc_type]
        except KeyError:
            msg = (
                "Unregistered primary association type "
                f"{association.data_prod_assoc_type!r}"
            )
            raise ValueError(msg) from None
        if reference in products:
            product = products[reference]
            if target_type is not None and product.meta.data_prod_type != target_type:
                msg = "Primary relationship disagrees with its target type"
                raise ValueError(msg)
            continue
        # Observation/acquisition edges specify their terminal type and canonical
        # identity without requiring a raw-file inventory. Named collections need
        # their supplied record; arbitrary UID text never establishes their type.
        if target_type in {"dp_obs", "dp_raw_obs"} and reference.namespace_uid is None:
            return _normalize_entity_uid(reference.uid, target_type), target_type
        msg = f"Primary placement requires associated product {reference}"
        raise ValueError(msg)
    msg = "Primary placement contains a cycle"
    raise ValueError(msg)


def _version_group(index, primary_uid, primary_type, descriptor, actor):
    return DataProdIndex(
        product=DataProd(
            uid=index.product.uid,
            meta=VersionGroupMeta(
                tag="version_group",
                name=index.product.uid,
                data_prod_type="dp_version_group",
                content_data_prod_type=index.product.meta.data_prod_type,
            ),
            audit=_audit(actor),
            assocs=[
                DataProdAssoc(
                    uid=primary_uid,
                    data_prod_assoc_type=VERSION_GROUP_PRIMARY_ASSOC_TYPES[
                        primary_type
                    ],
                    is_primary=True,
                    audit=_audit(actor),
                )
            ],
            data_items=[
                DataProdItem(
                    meta={
                        "type": "artifact",
                        "format": "data_prod_index",
                        "namespace_uid": str(descriptor.namespace.uid),
                    },
                    audit=_audit(actor),
                    source=DataProdSource.model_validate(
                        {
                            "path": "index.yaml",
                            "location": "self",
                            "availability_state": "available",
                        }
                    ),
                )
            ],
        ),
        actors=descriptor.actors,
        locations={
            "self": Location(root=".", location_type="local", audit=_audit(actor))
        },
    )


class DataProdTagBusyError(RuntimeError):
    """Another writer owns the requested product tag."""

    def __init__(self, path: Path, tag: str) -> None:
        self.path, self.tag = path, tag
        super().__init__(f"Product tag {tag!r} is busy: {path}")


@contextmanager
def _tag_lock(product, tag):
    if tag is None:
        yield
        return
    lock = FileLock(product / f".tag-{tag}.lock", timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise DataProdTagBusyError(product, tag) from exc
    try:
        yield
    finally:
        lock.release()


class _Materialization:
    """Store-owned filesystem scope; callers receive only the bound index."""

    def __init__(self, path, product_path, index, run, tag):
        descriptor = run.store.read_namespace()
        self.path, self.product_path = path, product_path
        self.group_uid = index.product.uid
        self._run_writer, self.tag = run, tag
        self.completed = False
        self.index = index.model_copy(deep=True)
        self.index.namespace = descriptor.namespace
        # The namespace may have been created by an earlier actor. Its audit
        # references must remain portable with this standalone product index.
        self.index.actors.update(descriptor.actors)
        self.index.store_run = run.store_run
        self.index.product.uid = f"{index.product.uid}/{run.run.uid}"
        self.index.product.namespace_uid = descriptor.namespace.uid
        self.index.product.meta.version_group = DataProdRef(uid=index.product.uid)
        self.index.product.tags = {}
        self.identity = self.index.product.reference
        self.primary = _primary(index)
        self.kind = index.product.meta.data_prod_type
        self.index._bind_locations(path)

    def finish(self):
        index = self.index
        if (
            index.product.reference != self.identity
            or index.product.meta.data_prod_type != self.kind
            or index.product.meta.version_group != DataProdRef(uid=self.group_uid)
            or index.run is None
            or index.run.uid != self._run_writer.run.uid
            or _primary(index) != self.primary
        ):
            msg = "Materialization identity changed inside its scope"
            raise ValueError(msg)
        actor = self._run_writer.actor
        descriptor = self._run_writer.store.read_namespace()
        if index.provenance is not None:
            self._run_writer._provenance(self.group_uid, index.provenance)
        snapshot = index
        # Supporting indexes resolve placement; each version describes its own files.
        snapshot.products = []
        snapshot.store_run = self._run_writer.store_run
        snapshot.product.audit = _audit(actor)
        _include_index_item(snapshot, "index.yaml", descriptor.namespace.uid)
        snapshot.product.data_items = [
            item.model_copy(update={"audit": _audit(actor)})
            for item in snapshot.product.data_items
        ]
        snapshot.locations = {
            key: loc.model_copy(update={"audit": _audit(actor)})
            for key, loc in snapshot.locations.items()
        }
        snapshot.product.assocs = [
            a.model_copy(
                update={
                    "runs": {
                        self._run_writer.run.uid: DataProdAssocRun(
                            created_at=datetime.now(UTC), created_by=actor.uid
                        )
                    },
                    "audit": _audit(actor),
                }
            )
            for a in snapshot.product.assocs
        ]
        DataProdIndex.model_validate(snapshot.model_dump())
        _sync_tree(self.path)
        with FileLock(self.product_path / ".publish.lock"):
            root_path = self.product_path / "index.yaml"
            current = GroupHeader.load(root_path)
            snapshot.write(self.path / "index.yaml")
            destination = self.product_path / self._run_writer.store_run.name
            if destination.exists():
                msg = f"Run destination already exists: {destination}"
                raise FileExistsError(msg)
            self.path.rename(destination)
            _sync_directory(self.product_path)
            member = snapshot
            _append_version_member(
                current,
                self.product_path,
                self._run_writer.store_run.name,
                DataProdAssoc(
                    uid=member.product.uid,
                    namespace_uid=member.product.namespace_uid,
                    data_prod_assoc_type="dpa_version_group_data_prod",
                    runs={
                        self._run_writer.run.uid: DataProdAssocRun(
                            created_at=datetime.now(UTC), created_by=actor.uid
                        )
                    },
                    audit=_audit(actor),
                ),
            )
            current.actors[actor.uid] = actor
            current.product.audit = _audit(actor, current.product.audit)
            if self.tag is not None:
                current.product.tags[self.tag] = DataProdTag(
                    current=DataProdRef(
                        uid=member.product.uid,
                        namespace_uid=member.product.namespace_uid,
                    ),
                    audit=_audit(actor, current.product.tags[self.tag].audit),
                )
            current.write(root_path, overwrite=True)
            self.completed = True
        self._run_writer._record(self.group_uid, "completed")
        snapshot.store_run = self._run_writer.store_run
        snapshot._bind_locations(destination)
        return destination


def _sync_tree(path):
    for directory, _, filenames in os.walk(path, topdown=False):
        for name in filenames:
            with (Path(directory) / name).open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(Path(directory))


def _descriptor_item(index, destination, namespace_path, namespace_uid):
    """Project a logical descriptor using one storage location per namespace."""
    root_uri = namespace_path.resolve().as_uri()
    relative_root = os.path.relpath(namespace_path, destination.parent)
    filename = destination.relative_to(namespace_path).as_posix()
    key = next(
        (k for k, loc in index.locations.items() if loc.root == root_uri),
        None,
    )
    if key is None:
        key = "_store"
        while key in index.locations and index.locations[key].root != relative_root:
            key += "_"
        index.locations.setdefault(
            key, Location(root=relative_root, location_type="local")
        )
    index.locations[key].root = relative_root
    # Preserve audit data on re-export and references to the same logical
    # product published in other namespaces. Reprojection replaces only self.
    retained = []
    removed = set()
    has_self = False
    for item in index.product.data_items:
        if item.meta.get("format") == "data_prod_index" and item.meta.get(
            "namespace_uid"
        ) == str(namespace_uid):
            removed.add(item.source.location)
            if item.source.location == key and item.source.path.as_posix() == filename:
                retained.append(item)
                has_self = True
        else:
            retained.append(item)
    if not has_self:
        retained.append(
            DataProdItem(
                meta={
                    "type": "artifact",
                    "format": "data_prod_index",
                    "namespace_uid": str(namespace_uid),
                },
                source=DataProdSource.model_validate(
                    {
                        "path": filename,
                        "location": key,
                        "availability_state": "available",
                    }
                ),
            )
        )
    index.product.data_items = retained
    for unused in removed - {i.source.location for i in retained}:
        del index.locations[unused]


def _include_index_item(index, filename, namespace_uid):
    """Declare the descriptor written by this store alongside producer artifacts."""
    location = next(
        (key for key, value in index.locations.items() if value.root == "."),
        "_index",
    )
    index.locations[location] = Location(root=".", location_type="local")
    index.product.data_items = [
        item
        for item in index.product.data_items
        if item.meta.get("format") != "data_prod_index"
    ]
    index.product.data_items.append(
        DataProdItem(
            meta={
                "type": "artifact",
                "format": "data_prod_index",
                "namespace_uid": str(namespace_uid),
                **({"run_uid": str(index.run.uid)} if index.run is not None else {}),
            },
            source=DataProdSource.model_validate(
                {
                    "path": filename,
                    "location": location,
                    "availability_state": "available",
                }
            ),
        )
    )
