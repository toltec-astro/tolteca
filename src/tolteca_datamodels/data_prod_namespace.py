"""Portable namespace identity and attribution for product indexes.

A namespace has a stable UUID and a relocatable directory path. Actors identify
reusable environments; run records carry invocation-specific information.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, ConfigDict, Field, NonNegativeInt
from tollan.config import FrozenBaseModel

from .data_prod_run import DataProdRun, RunUid

__all__ = [
    "DataProdActor",
    "DataProdAudit",
    "DataProdNamespace",
    "DataProdNamespaceIndex",
    "NamespacePath",
]


def _namespace_path(value: str) -> str:
    if (
        not value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(char in value for char in "\\\x00")
    ):
        msg = "Namespace requires a relative path without dot components"
        raise ValueError(msg)
    return value


type NamespacePath = Annotated[str, AfterValidator(_namespace_path)]
"""Relative namespace path, with optional hierarchical components."""


class DataProdActor(FrozenBaseModel):
    """Stable environment identity shared by multiple invocations."""

    model_config = ConfigDict(extra="forbid")
    uid: UUID = Field(strict=False, description="Stable portable UUID.")
    name: str = Field(description="Readable actor name, independent of identity.")
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime = Field(strict=False)


class DataProdAudit(FrozenBaseModel):
    """Original creation and most recent modification attribution."""

    model_config = ConfigDict(extra="forbid")
    created_at: AwareDatetime = Field(strict=False)
    created_by: UUID = Field(strict=False)
    updated_at: AwareDatetime = Field(strict=False)
    updated_by: UUID = Field(strict=False)


class DataProdNamespace(FrozenBaseModel):
    """Stable namespace identity, projection path and allocation frontier."""

    model_config = ConfigDict(extra="forbid")
    uid: UUID = Field(strict=False, description="Stable portable UUID.")
    path: NamespacePath = Field(description="Current relative namespace directory.")
    counter_next: NonNegativeInt = Field(
        default=0, description="Next automatic run counter candidate."
    )
    meta: dict[str, Any] = Field(default_factory=dict)
    audit: DataProdAudit


class DataProdNamespaceIndex(FrozenBaseModel):
    """Namespace descriptor with actors and reserved invocation records."""

    model_config = ConfigDict(extra="forbid")
    namespace: DataProdNamespace = Field(
        description="Namespace identity and allocation state."
    )
    actors: dict[Annotated[UUID, Field(strict=False)], DataProdActor] = Field(
        default_factory=dict, description="Actors referenced by this descriptor."
    )
    runs: dict[RunUid, DataProdRun] = Field(
        default_factory=dict,
        description="Reserved invocations, including failed attempts.",
    )
