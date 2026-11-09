"""Portable namespace identity and attribution for product indexes.

A namespace has a stable UUID and a relocatable directory path. Actors identify
reusable environments; run records carry invocation-specific information.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    ConfigDict,
    Field,
    NonNegativeInt,
    field_validator,
    model_validator,
)
from pydantic_core import to_json
from tollan.config import FrozenBaseModel
from tollan.config.types import PathSegment, SubPath

from ._io import _read_yaml
from .actor import _ActorsByUid
from .run import RunUid, StoreRun, WorkflowRun

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "DataProdAudit",
    "DataProdNamespace",
    "DataProdNamespaceIndex",
]


class DataProdAudit(FrozenBaseModel):
    """Creation and latest modification times with independently known actors.

    Use None for an unknown creator or modifier. The timestamps and any known
    actor remain available even when the other actor is unknown.
    """

    model_config = ConfigDict(extra="forbid")
    created_at: AwareDatetime = Field(strict=False)
    created_by: UUID | None = Field(
        default=None, strict=False, description="Creator UUID, or None if unknown."
    )
    updated_at: AwareDatetime = Field(strict=False)
    updated_by: UUID | None = Field(
        default=None,
        strict=False,
        description="Latest modifier UUID, or None if unknown.",
    )


class DataProdNamespace(FrozenBaseModel):
    """Stable namespace identity, projection path and run-name counter."""

    model_config = ConfigDict(extra="forbid")
    uid: UUID = Field(strict=False, description="Stable portable UUID.")
    path: SubPath = Field(description="Current relative namespace directory.")
    counter_next: NonNegativeInt = Field(
        default=0, description="Next automatic run counter candidate."
    )
    meta: dict[str, Any] = Field(default_factory=dict)
    audit: DataProdAudit


class DataProdNamespaceIndex(FrozenBaseModel):
    """Portable namespace metadata, actors and optional selected run records.

    An empty run collection is omitted when serialized. A store's namespace
    metadata file contains no run history; exports can include selected runs.
    Actor and run keys must match their records' UUIDs. Store-run keys must match
    their records' execution UUIDs. Each store run must reference an execution
    included in ``runs``. Reserved names and assigned counter values must be
    unique within the namespace.
    """

    model_config = ConfigDict(extra="forbid")
    namespace: DataProdNamespace
    actors: _ActorsByUid = Field(default_factory=dict)
    runs: dict[RunUid, WorkflowRun] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    store_runs: dict[RunUid, StoreRun] = Field(default_factory=dict)
    primary_paths: dict[Annotated[UUID, Field(strict=False)], PathSegment] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description="Readable directories for UUID primaries in this namespace.",
    )

    @field_validator("runs")
    @classmethod
    def _validate_run_mapping_keys(
        cls, runs: dict[UUID, WorkflowRun]
    ) -> dict[UUID, WorkflowRun]:
        for key, run in runs.items():
            if key != run.uid:
                msg = f"Run mapping key {key} differs from execution UID {run.uid}"
                raise ValueError(msg)
        return runs

    @field_validator("store_runs")
    @classmethod
    def _validate_store_runs(
        cls, store_runs: dict[UUID, StoreRun]
    ) -> dict[UUID, StoreRun]:
        names: set[str] = set()
        counter_values: set[int] = set()
        for key, store_run in store_runs.items():
            if key != store_run.run_uid:
                msg = (
                    f"Store-run mapping key {key} differs from execution UID "
                    f"{store_run.run_uid}"
                )
                raise ValueError(msg)
            if store_run.name in names:
                msg = f"Run name {store_run.name!r} is reserved by multiple executions"
                raise ValueError(msg)
            names.add(store_run.name)
            counter = None if store_run.generator is None else store_run.generator.value
            if counter is not None:
                if counter in counter_values:
                    msg = f"Counter value {counter} is reserved by multiple executions"
                    raise ValueError(msg)
                counter_values.add(counter)
        return store_runs

    @model_validator(mode="after")
    def _validate_store_run_references(self) -> Self:
        for run_uid in self.store_runs:
            if run_uid not in self.runs:
                msg = f"Store run {run_uid} must reference an included execution"
                raise ValueError(msg)
        return self

    @field_validator("primary_paths")
    @classmethod
    def _validate_primary_paths(cls, value: dict[UUID, str]) -> dict[UUID, str]:
        paths = list(value.values())
        if len(set(paths)) != len(paths):
            msg = "Primary paths must be unique within the namespace"
            raise ValueError(msg)
        if {"namespace.yaml", ".namespace.lock", ".runs"}.intersection(paths):
            msg = "Primary path is reserved by the namespace store"
            raise ValueError(msg)
        return value

    @classmethod
    def read(cls, path: str | Path, *, run_uids: Iterable[UUID] | None = None) -> Self:
        """Read a filesystem namespace projection and selected run records.

        Parameters
        ----------
        path : str or pathlib.Path
            Store's namespace metadata filename, without embedded run history.
        run_uids : iterable of UUID or None, optional
            Runs to include. None reads all saved run records; an empty
            iterable reads only namespace metadata and actors.

        Returns
        -------
        DataProdNamespaceIndex
            Self-contained namespace document for registration or transport.
        """
        path = Path(path)
        index = cls.model_validate_json(to_json(_read_yaml(path)))
        directory = path.parent / ".runs" / "by_uid"
        paths = (
            directory.glob("*/run.yaml")
            if run_uids is None
            else (directory / str(uid) / "run.yaml" for uid in run_uids)
        )
        paths = list(paths)
        runs = [WorkflowRun.model_validate_json(to_json(_read_yaml(p))) for p in paths]
        store_runs = [
            StoreRun.model_validate_json(
                to_json(_read_yaml(p.with_name("store_run.yaml")))
            )
            for p in paths
            if p.with_name("store_run.yaml").exists()
        ]
        return cls(
            namespace=index.namespace,
            actors=index.actors,
            runs={run.uid: run for run in runs},
            store_runs={a.run_uid: a for a in store_runs},
            primary_paths=index.primary_paths,
        )
