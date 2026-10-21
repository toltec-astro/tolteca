"""Place, publish and select data products in a local namespace.

Use an invocation to publish several products with one run UID::

    request = RunRequest(name=CounterRunNameGenerator(key="nightly"))
    with store.invocation(run=request) as invocation:
        with store.publication(index, primary_uid=uid, invocation=invocation) as output:
            # Write and close payloads beneath output.path.
            output.commit(completed_index)

Omit ``invocation`` for a single-product run. Publication holds the requested tag
until context exit. A committed realization is immutable; changing a tag selects
another completed realization.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING
from uuid import uuid4

from filelock import FileLock, Timeout
from pydantic import TypeAdapter
from tollan.config import RuntimeInfo
from tollan.utils.fileloc import fileloc
from tollan.utils.yaml import yaml_load

from .data_prod import (
    DataProdIndex,
    DataProdIndexAssoc,
    DataProdIndexAssocRun,
    DataProdIndexLocation,
    DataProdIndexSource,
    DataProdRef,
    DataProdTag,
    _sync_directory,
    _write_yaml,
    normalize_primary_uid,
)
from .data_prod_namespace import (
    DataProdActor,
    DataProdAudit,
    DataProdNamespace,
    DataProdNamespaceIndex,
    NamespacePath,
)
from .data_prod_run import (
    CounterRunNameState,
    DataProdRun,
    PathComponent,
    PublicationRequest,
    RunName,
    RunRequest,
    prepare_run_request,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

__all__ = [
    "DataProdInvocation",
    "DataProdPublication",
    "DataProdStore",
    "DataProdTagBusyError",
]


def _path_component(value):
    return TypeAdapter(PathComponent).validate_python(value)


def _audit(actor, previous=None):
    now = datetime.now(UTC)
    return DataProdAudit(
        created_at=now if previous is None else previous.created_at,
        created_by=actor.uid if previous is None else previous.created_by,
        updated_at=now,
        updated_by=actor.uid,
    )


class DataProdStore:
    """Publish products under one relocatable namespace.

    Parameters
    ----------
    path : str or pathlib.Path
        Product root, usually ``dataprod_toltec``.
    namespace : str, optional
        Relative namespace path, default ``scratch``. Construction creates no files.
    """

    def __init__(self, path: str | Path, *, namespace: str = "scratch") -> None:
        self.path = Path(path)
        self.namespace = TypeAdapter(NamespacePath).validate_python(namespace)

    @property
    def namespace_path(self):
        """Directory containing the namespace descriptor and product projections."""
        return self.path / self.namespace

    def read_namespace(self) -> DataProdNamespaceIndex:
        """Read namespace identity, actors and invocation records."""
        return DataProdNamespaceIndex.model_validate(
            yaml_load(self.namespace_path / "namespace.yaml")
        )

    def ensure_namespace(
        self, *, runtime_info: RuntimeInfo | None = None
    ) -> DataProdNamespaceIndex:
        """Create or read the namespace descriptor without reserving a run.

        Parameters
        ----------
        runtime_info : RuntimeInfo or None, optional
            Actor environment. None captures the current environment.

        Returns
        -------
        DataProdNamespaceIndex
            Portable namespace identity, actors and existing invocation records.
        """
        self.namespace_path.mkdir(parents=True, exist_ok=True)
        with FileLock(self.namespace_path / ".namespace.lock"):
            descriptor, _ = self._namespace_with_actor(runtime_info or RuntimeInfo())
            _write_yaml(
                descriptor, self.namespace_path / "namespace.yaml", overwrite=True
            )
        return descriptor

    def _namespace_with_actor(self, runtime):
        descriptor = (
            self.read_namespace()
            if (self.namespace_path / "namespace.yaml").exists()
            else None
        )
        stable = runtime.model_dump(
            mode="json",
            include={"hostname", "username", "platform", "python_prefix", "timezone"},
        )
        actor = (
            next((a for a in descriptor.actors.values() if a.meta == stable), None)
            if descriptor
            else None
        )
        if actor is None:
            actor = DataProdActor(
                uid=uuid4(),
                name=f"{runtime.username}@{runtime.hostname}",
                meta=stable,
                created_at=datetime.now(UTC),
            )
        if descriptor is None:
            descriptor = DataProdNamespaceIndex(
                namespace=DataProdNamespace(
                    uid=uuid4(), path=self.namespace, audit=_audit(actor)
                )
            )
        descriptor.actors[actor.uid] = actor
        return descriptor, actor

    def get_product_path(
        self, primary_uid: str, data_prod_type: str, *, uid: str | None = None
    ) -> Path:
        """Locate a product directory without creating it.

        Parameters
        ----------
        primary_uid : str
            Canonical observation or acquisition UID. Directory labels omit TCS.
        data_prod_type : str
            Managed product type.
        uid : str or None, optional
            Logical group UID for grouped products. Observation and raw-processed
            products use their type name; other supplied UIDs name the directory.

        Returns
        -------
        pathlib.Path
            Namespace, primary label and product directory.
        """
        primary = normalize_primary_uid(primary_uid).removeprefix("tcs-")
        name = data_prod_type
        if uid is not None and data_prod_type not in {
            "dp_raw_obs_proc",
            "dp_raw_obs",
            "dp_obs",
        }:
            name = uid.replace(normalize_primary_uid(primary_uid), primary)
        return self.namespace_path / primary / _path_component(name)

    def get_index_path(
        self, primary_uid, data_prod_type, *, directory=True, run_name=None
    ):
        """Locate a descriptor, version-group index or realized-product index.

        Parameters
        ----------
        primary_uid, data_prod_type : str
            Primary UID and product directory name.
        directory : bool, optional
            False selects ``<type>.yaml``; True selects ``<type>/index.yaml``.
        run_name : str or None, optional
            Realization directory beneath the product directory.

        Returns
        -------
        pathlib.Path
            Destination without creating directories.
        """
        path = self.get_product_path(primary_uid, data_prod_type)
        if run_name is not None:
            if not directory:
                msg = "Run indexes require a product directory"
                raise ValueError(msg)
            return path / TypeAdapter(RunName).validate_python(run_name) / "index.yaml"
        return path / "index.yaml" if directory else path.with_name(path.name + ".yaml")

    def write_index(self, index, *, primary_uid, directory=True, overwrite=False):
        """Write an inventory whose payloads already exist.

        Parameters
        ----------
        index : DataProdIndex
            Product inventory.
        primary_uid : str
            Terminal observation or acquisition UID.
        directory, overwrite : bool, optional
            Directory layout and atomic replacement controls.

        Returns
        -------
        pathlib.Path
            Written descriptor.
        """
        kind = index.meta["data_prod_type"]
        if kind in {"dp_obs", "dp_raw_obs", "dp_cal_group"}:
            destination = self.get_index_path(primary_uid, kind, directory=directory)
        else:
            if kind != "dp_version_group" and (
                index.namespace is None or index.run is None
            ):
                msg = "Use publication() before exporting a proposed product"
                raise ValueError(msg)
            group_uid = (
                index.uid
                if index.run is None
                else index.uid.removesuffix(f"/{index.run.uid}")
            )
            destination = self.get_product_path(
                primary_uid,
                index.meta.get("content_data_prod_type", kind),
                uid=group_uid,
            )
            if index.run is not None:
                destination /= index.run.name
        return index.write(destination, overwrite=overwrite)

    @contextmanager
    def invocation(
        self,
        *,
        run: RunRequest | None = None,
        runtime_info: RuntimeInfo | None = None,
        context: dict[str, Any] | None = None,
    ) -> Iterator[DataProdInvocation]:
        """Reserve one namespace run for all outputs of an invocation.

        Parameters
        ----------
        run : RunRequest or None, optional
            Identity and name policy; None generates a UUID and uses it as the name.
        runtime_info : RuntimeInfo or None, optional
            Invocation environment. None captures the current environment.
        context : dict or None, optional
            Resolved workflow configuration and exact selected inputs.

        Yields
        ------
        DataProdInvocation
            Shared reservation. Failed and abandoned reservations remain consumed.
        """
        runtime = runtime_info or RuntimeInfo()
        request = prepare_run_request(
            RunRequest.model_validate({} if run is None else run), uuid4()
        )
        root = self.namespace_path
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(root / ".namespace.lock"):
            path = root / "namespace.yaml"
            descriptor, actor = self._namespace_with_actor(runtime)
            actors = descriptor.actors
            state = {
                "counter": CounterRunNameState(next=descriptor.namespace.counter_next)
            }

            if request.uid in descriptor.runs:
                msg = f"Run UID {request.uid} is already reserved in {root}"
                raise FileExistsError(msg)
            names = {record.name for record in descriptor.runs.values()}

            def available(value):
                return value not in names

            if isinstance(request.name, str):
                if not available(request.name):
                    msg = f"Run name {request.name!r} is already reserved in {root}"
                    raise FileExistsError(msg)
                run_name, generator = request.name, None
            else:
                run_name, generator = request.name._allocate(
                    state, descriptor.runs, available
                )
            reserved = DataProdRun(
                uid=request.uid,
                name=run_name,
                namespace_uid=descriptor.namespace.uid,
                actor_uid=actor.uid,
                generator=generator,
                time_created=datetime.now(UTC),
                context=context or {},
                meta={
                    "runtime_info": runtime.model_dump(
                        mode="json", exclude={"config_sources", "validation_context"}
                    )
                },
            )
            namespace = descriptor.namespace.model_copy(
                update={
                    "path": self.namespace,
                    "counter_next": state["counter"].next,
                    "audit": _audit(actor, descriptor.namespace.audit),
                }
            )
            descriptor = descriptor.model_copy(
                update={
                    "namespace": namespace,
                    "actors": actors,
                    "runs": {**descriptor.runs, reserved.uid: reserved},
                }
            )
            _write_yaml(descriptor, path, overwrite=True)
        invocation = DataProdInvocation(self, reserved, actor, runtime)
        try:
            yield invocation
        except BaseException as exc:
            invocation._finish(
                "cancelled"
                if isinstance(exc, (KeyboardInterrupt, GeneratorExit))
                else "failed"
            )
            raise
        else:
            failed = any(
                p.status != "completed" for p in invocation.run.publications.values()
            )
            invocation._finish("failed" if failed else "completed")

    @contextmanager
    def publication(  # noqa: PLR0913 -- publication identity and invocation options
        self,
        index,
        *,
        primary_uid,
        run=None,
        tag="main",
        runtime_info=None,
        invocation=None,
    ):
        """Hold a tag while creating one immutable product realization.

        Parameters
        ----------
        index : DataProdIndex
            Proposed product metadata and input associations. Its UID identifies
            the version group; commit appends the invocation's run UID.
        primary_uid : str
            Terminal observation or acquisition UID.
        run : RunRequest or None, optional
            Single-product run request. Omit when supplying an invocation.
        tag : str or None, optional
            Selection to hold and advance, default ``main``. None publishes untagged.
        runtime_info : RuntimeInfo or None, optional
            Environment for a single-product invocation.
        invocation : DataProdInvocation or None, optional
            Shared run reserved by this store. None creates a single-product run.

        Yields
        ------
        DataProdPublication
            Private staging directory. Call commit after validating closed payloads.

        Raises
        ------
        DataProdTagBusyError
            Another writer holds this product directory's tag. No computation
            should start before entry.
        FileExistsError
            The run or realization is already reserved.
        """
        if invocation is not None and (
            run is not None
            or invocation.store.namespace_path.resolve()
            != self.namespace_path.resolve()
        ):
            msg = (
                "Publication requires this store's invocation and no second run request"
            )
            raise ValueError(msg)
        kind = index.meta["data_prod_type"]
        if kind in {"dp_obs", "dp_raw_obs", "dp_cal_group"}:
            msg = "Intrinsic products use write_index(), without a producing run"
            raise ValueError(msg)
        product = self.get_product_path(primary_uid, kind, uid=index.uid).resolve()
        product.mkdir(parents=True, exist_ok=True)
        with _tag_lock(product, None if tag is None else _path_component(tag)):
            scope = (
                nullcontext(invocation)
                if invocation
                else self.invocation(run=run, runtime_info=runtime_info)
            )
            with scope as active:
                active._request(index.uid, kind, primary_uid, tag)
                with FileLock(product / ".publish.lock"):
                    root_path = product / "index.yaml"
                    descriptor = self.read_namespace()
                    current = (
                        DataProdIndex.model_validate(yaml_load(root_path))
                        if root_path.exists()
                        else _version_group(
                            index, primary_uid, descriptor, active.actor
                        )
                    )
                    if (
                        current.uid != index.uid
                        or current.meta["content_data_prod_type"] != kind
                    ):
                        msg = "Publication would replace a different version group"
                        raise ValueError(msg)
                    if tag is not None and tag not in current.tags:
                        current.tags[tag] = DataProdTag(audit=_audit(active.actor))
                    current.write(root_path, overwrite=True)
                staging = product / ".staging"
                staging.mkdir(exist_ok=True)
                path = Path(mkdtemp(dir=staging))
                pending = DataProdPublication(path, product, index, active, tag)
                try:
                    yield pending
                except BaseException as exc:
                    if not pending._committed:
                        active._record(index.uid, "failed", str(exc))
                    raise
                finally:
                    if not pending._committed:
                        active._record(
                            index.uid, "failed", "Publication did not complete"
                        )
                    if path.exists():
                        shutil.rmtree(path)

    def set_tag(self, primary_uid, data_prod_type, run_name, *, tag="main"):
        """Select a completed realization without changing its artifacts.

        Parameters
        ----------
        primary_uid, data_prod_type : str
            Product placement identifiers.
        run_name : str
            Completed realization's namespace-unique run name.
        tag : str, optional
            Selection label, default ``main``.

        Returns
        -------
        pathlib.Path
            Selected realization index.
        """
        product = self.get_product_path(primary_uid, data_prod_type).resolve()
        with (
            _tag_lock(product, _path_component(tag)),
            FileLock(product / ".publish.lock"),
        ):
            path = product / "index.yaml"
            index = DataProdIndex.model_validate(yaml_load(path))
            member = next((p for p in index.products if p.run.name == run_name), None)
            if member is None:
                raise KeyError(run_name)
            # Promotion attribution uses the current environment, not the producer.
            descriptor = self.read_namespace()
            runtime = RuntimeInfo()
            stable = runtime.model_dump(
                mode="json",
                include={
                    "hostname",
                    "username",
                    "platform",
                    "python_prefix",
                    "timezone",
                },
            )
            with FileLock(self.namespace_path / ".namespace.lock"):
                descriptor = self.read_namespace()
                actor = next(
                    (a for a in descriptor.actors.values() if a.meta == stable), None
                )
                if actor is None:
                    actor = DataProdActor(
                        uid=uuid4(),
                        name=f"{runtime.username}@{runtime.hostname}",
                        meta=stable,
                        created_at=datetime.now(UTC),
                    )
                    descriptor.actors[actor.uid] = actor
                    _write_yaml(
                        descriptor,
                        self.namespace_path / "namespace.yaml",
                        overwrite=True,
                    )
            index.actors[actor.uid] = actor
            previous = index.tags.get(tag)
            index.tags[tag] = DataProdTag(
                current=DataProdRef(uid=member.uid, namespace_uid=member.namespace.uid),
                audit=_audit(actor, previous.audit if previous else None),
            )
            index.audit = _audit(actor, index.audit)
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
        index = DataProdIndex.read(path)
        if index.meta["data_prod_type"] != "dp_version_group":
            return path
        selected = index.tags[tag].current
        product = next(
            (
                p
                for p in index.products
                if selected is not None
                and p.uid == selected.uid
                and p.namespace.uid == selected.namespace_uid
            ),
            None,
        )
        if product is None:
            raise KeyError(tag)
        item = next(
            i for i in product.data_items if i.meta.get("format") == "data_prod_index"
        )
        return fileloc(
            item.source_uri,
            local_parent_path=fileloc(product.locations[item.location].root_uri).path,
        ).path


def _version_group(index, primary_uid, descriptor, actor):
    terminal_type = "obs" if primary_uid.count("-") == 1 else "raw_obs"
    return DataProdIndex(
        uid=index.uid,
        meta={
            "tag": "version_group",
            "name": index.uid,
            "data_prod_type": "dp_version_group",
            "content_data_prod_type": index.meta["data_prod_type"],
        },
        actors=descriptor.actors,
        audit=_audit(actor),
        assocs=[
            DataProdIndexAssoc(
                uid=primary_uid,
                data_prod_assoc_type=f"dpa_version_group_{terminal_type}",
                is_primary=True,
                audit=_audit(actor),
            )
        ],
        locations={
            "self": DataProdIndexLocation(
                root_uri=".", location_type="local", audit=_audit(actor)
            )
        },
        data_items=[
            DataProdIndexSource(
                source_uri="index.yaml",
                location="self",
                meta={
                    "type": "artifact",
                    "format": "data_prod_index",
                    "namespace_uid": str(descriptor.namespace.uid),
                },
                availability_state="available",
                audit=_audit(actor),
            )
        ],
    )


class DataProdInvocation:
    """Shared namespace reservation obtained from ``DataProdStore.invocation``."""

    def __init__(self, store, run, actor, runtime_info):
        self.store, self.run, self.actor, self.runtime_info = (
            store,
            run,
            actor,
            runtime_info,
        )

    def _update(self, **values):
        with FileLock(self.store.namespace_path / ".namespace.lock"):
            descriptor = self.store.read_namespace()
            self.run = descriptor.runs[self.run.uid].model_copy(update=values)
            descriptor.runs[self.run.uid] = self.run
            _write_yaml(
                descriptor, self.store.namespace_path / "namespace.yaml", overwrite=True
            )

    def _request(self, uid, kind, primary_uid, tag):
        if uid in self.run.publications:
            msg = f"Product {uid!r} already requested in run {self.run.uid!r}"
            raise FileExistsError(msg)
        request = PublicationRequest(
            uid=uid, data_prod_type=kind, primary_uid=primary_uid, tag=tag
        )
        self._update(publications={**self.run.publications, uid: request})

    def _record(self, uid, status, message=None):
        request = self.run.publications[uid].model_copy(
            update={"status": status, "message": message}
        )
        self._update(publications={**self.run.publications, uid: request})

    def _finish(self, status):
        self._update(status=status, time_completed=datetime.now(UTC))


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


class DataProdPublication:
    """Private staging area and shared invocation for one realized product."""

    def __init__(self, path, product_path, index, invocation, tag):
        self.path, self.product_path, self.index = path, product_path, index
        self.invocation, self.tag = invocation, tag
        self.runtime_info = invocation.runtime_info
        self._committed = False

    @property
    def run(self):
        """Current invocation record."""
        return self.invocation.run

    def commit(self, index: DataProdIndex) -> Path:
        """Publish closed payloads and advance this product's requested tag.

        Parameters
        ----------
        index : DataProdIndex
            Completed artifact inventory with the proposal's UID. Local roots
            must be relative to the staging directory.

        Returns
        -------
        pathlib.Path
            Immutable realization directory.

        Raises
        ------
        ValueError
            The product differs from the proposal.
        RuntimeError
            This publication has already committed.
        FileExistsError
            The realization destination already exists.
        """
        if self._committed:
            msg = "Publication has already been committed"
            raise RuntimeError(msg)
        if index.uid != self.index.uid:
            msg = "Publication would replace a different product"
            raise ValueError(msg)
        actor = self.invocation.actor
        descriptor = self.invocation.store.read_namespace()
        # Each index retains the exact resolved workflow for its own interfaces.
        if index.provenance is not None:
            self.invocation._update(
                context={
                    **self.run.context,
                    "outputs": {
                        **self.run.context.get("outputs", {}),
                        index.uid: index.provenance,
                    },
                }
            )
        snapshot = index.model_copy(
            deep=True,
            update={
                "uid": f"{index.uid}/{self.run.uid}",
                "namespace": descriptor.namespace,
                "actors": descriptor.actors,
                "audit": _audit(actor),
                "run": self.run,
                "products": [],
                "tags": {},
            },
        )
        snapshot.data_items = [
            item.model_copy(update={"audit": _audit(actor)})
            for item in snapshot.data_items
        ]
        snapshot.locations = {
            key: loc.model_copy(update={"audit": _audit(actor)})
            for key, loc in snapshot.locations.items()
        }
        snapshot.assocs = [
            a.model_copy(
                update={
                    "runs": {
                        self.run.uid: DataProdIndexAssocRun(
                            created_at=datetime.now(UTC), created_by=actor.uid
                        )
                    },
                    "audit": _audit(actor),
                }
            )
            for a in snapshot.assocs
        ]
        _sync_tree(self.path)
        with FileLock(self.product_path / ".publish.lock"):
            root_path = self.product_path / "index.yaml"
            current = DataProdIndex.model_validate(yaml_load(root_path))
            snapshot.write(self.path / "index.yaml")
            destination = self.product_path / self.run.name
            if destination.exists():
                msg = f"Run destination already exists: {destination}"
                raise FileExistsError(msg)
            self.path.rename(destination)
            _sync_directory(self.product_path)
            member = snapshot.model_copy(deep=True)
            member.locations = {
                key: loc.model_copy(
                    update={"root_uri": (Path(self.run.name) / loc.root_uri).as_posix()}
                )
                for key, loc in member.locations.items()
            }
            current.products.append(member)
            current.assocs.append(
                DataProdIndexAssoc(
                    uid=member.uid,
                    namespace_uid=snapshot.namespace.uid,
                    data_prod_assoc_type="dpa_version_group_data_prod",
                    runs={
                        self.run.uid: DataProdIndexAssocRun(
                            created_at=datetime.now(UTC), created_by=actor.uid
                        )
                    },
                    audit=_audit(actor),
                )
            )
            current.actors.update(descriptor.actors)
            current.audit = _audit(actor, current.audit)
            if self.tag is not None:
                current.tags[self.tag] = DataProdTag(
                    current=DataProdRef(
                        uid=member.uid, namespace_uid=member.namespace.uid
                    ),
                    audit=_audit(actor, current.tags[self.tag].audit),
                )
            current.write(root_path, overwrite=True)
            self._committed = True
        self.invocation._record(index.uid, "completed")
        return destination


def _sync_tree(path):
    for directory, _, filenames in os.walk(path, topdown=False):
        for name in filenames:
            with (Path(directory) / name).open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(Path(directory))
