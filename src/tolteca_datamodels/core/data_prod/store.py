"""Namespace attribution and run reservations for filesystem product stores."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from filelock import FileLock
from pydantic import TypeAdapter
from pydantic_core import to_json
from tollan.config import RuntimeInfo
from tollan.config.types import SubPath

from ._io import _read_yaml, _sync_directory, _write_yaml
from .actor import Actor, _environment_actor
from .namespace import (
    DataProdAudit,
    DataProdNamespace,
    DataProdNamespaceIndex,
)
from .run import (
    CounterRunNameGenerator,
    CounterRunNameState,
    PublicationRequest,
    RunName,
    StoreRun,
    WorkflowRun,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["DataProdStoreBase"]

_validate_namespace_path = TypeAdapter(SubPath).validate_python


def _audit(actor, previous=None):
    now = datetime.now(UTC)
    return DataProdAudit(
        created_at=now if previous is None else previous.created_at,
        created_by=actor.uid if previous is None else previous.created_by,
        updated_at=now,
        updated_by=actor.uid,
    )


def _select_namespace_actor(runtime: RuntimeInfo, actors: Iterable[Actor]) -> Actor:
    """Reuse a recorded environment actor before choosing the process default."""
    default = _environment_actor(runtime)
    return next((actor for actor in actors if actor.meta == default.meta), default)


class DataProdStoreBase:
    """Manage namespace metadata and run records at one filesystem location.

    Parameters
    ----------
    path : str or pathlib.Path
        Root directory containing namespaces.
    namespace : str or pathlib.PurePosixPath
        Relative namespace path. Construction creates no files.

    Notes
    -----
    Use ``write_run`` to save an execution record supplied by the caller.
    The execution owner determines its outcome.
    Product naming and publication are supplied by the concrete store.
    """

    def __init__(self, path: str | Path, *, namespace: str | PurePosixPath) -> None:
        self.path = Path(path)
        self.namespace = _validate_namespace_path(namespace)

    @property
    def namespace_path(self):
        """Directory containing namespace metadata and product projections."""
        return self.path / self.namespace

    def read_namespace(self) -> DataProdNamespaceIndex:
        """Read namespace metadata and actors without loading run history.

        Returns
        -------
        DataProdNamespaceIndex
            Namespace document with an empty run collection.
        """
        return DataProdNamespaceIndex.model_validate_json(
            to_json(_read_yaml(self.namespace_path / "namespace.yaml"))
        )

    def ensure_namespace(
        self, *, runtime_info: RuntimeInfo | None = None
    ) -> DataProdNamespaceIndex:
        """Ensure namespace metadata and the current actor exist without a run.

        Parameters
        ----------
        runtime_info : RuntimeInfo or None, optional
            Actor environment. None captures the current environment.
            Reuse a matching recorded actor, or this process's default actor
            when the namespace has no match.

        Returns
        -------
        DataProdNamespaceIndex
            Namespace metadata and actors, with an empty run collection.
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
        actor = _select_namespace_actor(
            runtime, descriptor.actors.values() if descriptor is not None else ()
        )
        if descriptor is None:
            descriptor = DataProdNamespaceIndex(
                namespace=DataProdNamespace(
                    uid=uuid4(), path=self.namespace, audit=_audit(actor)
                )
            )
        descriptor.actors[actor.uid] = actor
        return descriptor, actor

    def write_run(
        self,
        run: WorkflowRun,
        *,
        run_name: RunName | CounterRunNameGenerator | None = None,
    ) -> Path:
        """Save an execution snapshot in this namespace without owning its lifecycle.

        Parameters
        ----------
        run : WorkflowRun
            Execution to record. Its identity and provenance remain unchanged.
        run_name : str, CounterRunNameGenerator or None, optional
            Reserve a namespace-local name, even when the run produces no
            products. None writes the snapshot without allocating a name.
            Repeated writes reuse a successfully saved reservation.
            Later materialization of this execution reuses that reservation.

        Returns
        -------
        pathlib.Path
            Saved run.yaml path.

        Raises
        ------
        ValueError
            The execution, its actor, completed outcome or naming policy
            conflicts with a saved record.
        FileExistsError
            The requested name or counter is already claimed, including by
            an interrupted reservation that was not saved.

        Notes
        -----
        A saved completed execution is retained when its earlier running
        snapshot is supplied. Runtime capture is not required to save a run
        or reserve its name.

        Interrupted allocation can leave counter gaps and claimed names.
        Automatic naming advances past them; fixed names remain unavailable.
        """
        if run_name is not None:
            self._reserve_run(run, run_name)
            return self.namespace_path / ".runs/by_uid" / str(run.uid) / "run.yaml"
        root = self.namespace_path
        root.mkdir(parents=True, exist_ok=True)
        path = root / ".runs" / "by_uid" / str(run.uid) / "run.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with (
            FileLock(root / ".namespace.lock"),
            FileLock(path.with_suffix(".lock")),
        ):
            if path.exists():
                previous = WorkflowRun.model_validate_json(to_json(_read_yaml(path)))
                stable = {"time_completed", "status"}
                if previous.model_dump(exclude=stable) != run.model_dump(
                    exclude=stable
                ):
                    msg = f"Conflicting execution {run.uid}"
                    raise ValueError(msg)
                if previous.time_completed is not None:
                    if run.time_completed is None:
                        return path
                    if previous != run:
                        msg = f"Conflicting completed execution {run.uid}"
                        raise ValueError(msg)
            descriptor = (
                self.read_namespace()
                if (root / "namespace.yaml").exists()
                else DataProdNamespaceIndex(
                    namespace=DataProdNamespace(
                        uid=uuid4(), path=self.namespace, audit=_audit(run.actor)
                    )
                )
            )
            actor = descriptor.actors.get(run.actor_uid)
            if actor is not None and actor != run.actor:
                msg = f"Conflicting actor {run.actor_uid}"
                raise ValueError(msg)
            descriptor.actors[run.actor_uid] = run.actor
            _write_yaml(descriptor, root / "namespace.yaml", overwrite=True)
            return _write_yaml(run, path, overwrite=True)

    def _reserve_run(self, run, name=None):
        # Reservation saves the snapshot only; a concrete store's write_run may
        # also act on a finished run, which must not happen during publication.
        DataProdStoreBase.write_run(self, run)
        root = self.namespace_path
        records = root / ".runs"
        path = records / "by_uid" / str(run.uid) / "store_run.yaml"
        with FileLock(root / ".namespace.lock"):
            if path.exists():
                store_run = StoreRun.model_validate_json(to_json(_read_yaml(path)))
                _validate_store_run_name(store_run, name)
                return _StoreRunWriter(self, run, store_run)
            descriptor = self.read_namespace()
            state = {
                "counter": CounterRunNameState(next=descriptor.namespace.counter_next)
            }
            name = str(run.uid) if name is None else name

            def available(value):
                return not (records / "by_name" / value).exists()

            def value_available(value):
                return not (records / "by_counter" / str(value)).exists()

            if isinstance(name, str):
                if not available(name):
                    msg = f"Run name {name!r} is already reserved"
                    raise FileExistsError(msg)
                store_run = StoreRun(run_uid=run.uid, name=name)
            else:
                if name.key is None:
                    name = name.model_copy(update={"key": str(run.uid)})
                resolved, generator = name._allocate(state, value_available, available)
                store_run = StoreRun(
                    run_uid=run.uid, name=resolved, generator=generator
                )
            descriptor = descriptor.model_copy(
                update={
                    "namespace": descriptor.namespace.model_copy(
                        update={
                            "counter_next": state["counter"].next,
                            "audit": _audit(run.actor, descriptor.namespace.audit),
                        }
                    )
                }
            )
            _write_yaml(descriptor, root / "namespace.yaml", overwrite=True)
            claims = []
            if store_run.generator is not None:
                claims.append(records / "by_counter" / str(store_run.generator.value))
            # Claim the counter first so a leftover name also consumes its value.
            claims.append(records / "by_name" / store_run.name)
            for target in claims:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.mkdir()
                _sync_directory(target.parent)
            _write_yaml(store_run, path)
        return _StoreRunWriter(self, run, store_run)

    def read_namespace_index(
        self, *, run_uids: Iterable[UUID] | None = None
    ) -> DataProdNamespaceIndex:
        """Read namespace information and selected runs for export.

        Parameters
        ----------
        run_uids : iterable of UUID or None, optional
            Runs to include. None includes all saved run records; an empty
            iterable reads only namespace metadata and actors.

        Returns
        -------
        DataProdNamespaceIndex
            Self-contained document for registration or transport.
        """
        return DataProdNamespaceIndex.read(
            self.namespace_path / "namespace.yaml", run_uids=run_uids
        )

    def read_run(self, uid: UUID) -> WorkflowRun:
        """Read one execution without loading other runs.

        Parameters
        ----------
        uid : UUID
            Portable run identity.

        Returns
        -------
        WorkflowRun
            Latest saved execution snapshot, with or without a reserved name.
        """
        return WorkflowRun.model_validate_json(
            to_json(
                _read_yaml(
                    self.namespace_path / ".runs" / "by_uid" / str(uid) / "run.yaml"
                )
            )
        )

    def write_namespace_index(self, index: DataProdNamespaceIndex) -> Path:
        """Create a namespace projection with independent run records.

        Parameters
        ----------
        index : DataProdNamespaceIndex
            Namespace and selected runs exported from an inventory.

        Returns
        -------
        pathlib.Path
            Namespace metadata filename. Run records are written separately.

        Raises
        ------
        FileExistsError
            A destination record already exists. Use a new projection location;
            this method does not merge an existing namespace.
        """
        root = self.namespace_path
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(root / ".namespace.lock"):
            descriptor = DataProdNamespaceIndex(
                namespace=index.namespace,
                actors=index.actors,
                primary_paths=index.primary_paths,
            )
            path = _write_yaml(descriptor, root / "namespace.yaml")
            for run in index.runs.values():
                directory = root / ".runs"
                _write_yaml(run, directory / "by_uid" / str(run.uid) / "run.yaml")
                store_run = index.store_runs.get(run.uid)
                if store_run is not None:
                    _write_yaml(
                        store_run,
                        directory / "by_uid" / str(run.uid) / "store_run.yaml",
                    )
                    names = [directory / "by_name" / store_run.name]
                    if store_run.generator is not None:
                        names.append(
                            directory / "by_counter" / str(store_run.generator.value)
                        )
                    for name in names:
                        name.parent.mkdir(parents=True, exist_ok=True)
                        name.mkdir(exist_ok=True)
                        _sync_directory(name.parent)
            return path


def _validate_store_run_name(store_run, name):
    """Require repeated publication to use the reserved destination name."""
    if isinstance(name, str) and name != store_run.name:
        msg = "Conflicting output name for the same execution"
        raise ValueError(msg)
    if isinstance(name, CounterRunNameGenerator):
        expected = name.model_dump(exclude={"key", "value"})
        actual = store_run.generator
        if (
            actual is None
            or actual.model_dump(exclude={"key", "value"}) != expected
            or (name.value is not None and name.value != actual.value)
            or (name.key is not None and name.key != actual.key)
        ):
            msg = "Conflicting store-run naming policy"
            raise ValueError(msg)


class _StoreRunWriter:
    """Write namespace-local run records; the caller owns execution finalization."""

    def __init__(self, store, run, store_run):
        self.store, self.run, self.store_run = store, run, store_run
        self.actor = run.actor

    def _change(self, uid, *, request=None, status=None, message=None):
        path = (
            self.store.namespace_path
            / ".runs"
            / "by_uid"
            / str(self.run.uid)
            / "store_run.yaml"
        )
        with FileLock(path.with_suffix(".lock")):
            current = StoreRun.model_validate_json(to_json(_read_yaml(path)))
            publications = dict(current.publications)
            if request is not None:
                if uid in publications:
                    msg = f"Product {uid!r} already requested in run {self.run.uid}"
                    raise FileExistsError(msg)
                publications[uid] = request
            else:
                publications[uid] = publications[uid].model_copy(
                    update={"status": status, "message": message}
                )
            self.store_run = current.model_copy(update={"publications": publications})
            _write_yaml(self.store_run, path, overwrite=True)

    def _provenance(self, uid, provenance):
        path = (
            self.store.namespace_path
            / ".runs"
            / "by_uid"
            / str(self.run.uid)
            / "store_run.yaml"
        )
        with FileLock(path.with_suffix(".lock")):
            current = StoreRun.model_validate_json(to_json(_read_yaml(path)))
            self.store_run = current.model_copy(
                update={"outputs": {**current.outputs, uid: provenance}}
            )
            _write_yaml(self.store_run, path, overwrite=True)

    def _request(self, uid, kind, primary_uid, tag):
        self._change(
            uid,
            request=PublicationRequest(
                uid=uid, data_prod_type=kind, primary_uid=primary_uid, tag=tag
            ),
        )

    def _record(self, uid, status, message=None):
        self._change(uid, status=status, message=message)
