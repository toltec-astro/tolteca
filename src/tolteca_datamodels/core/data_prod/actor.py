"""Portable execution actors."""

import json
from datetime import UTC, datetime
from functools import lru_cache
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import AfterValidator, AwareDatetime, ConfigDict, Field
from tollan.config import FrozenBaseModel, RuntimeInfo


class Actor(FrozenBaseModel):
    """Stable environment identity shared by multiple invocations."""

    model_config = ConfigDict(extra="forbid")
    uid: UUID = Field(strict=False, description="Stable portable UUID.")
    name: str = Field(description="Readable actor name, independent of identity.")
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime = Field(strict=False)


def _validate_actor_mapping_keys(actors: dict[UUID, Actor]) -> dict[UUID, Actor]:
    for key, actor in actors.items():
        if key != actor.uid:
            msg = f"Actor mapping key {key} differs from actor UID {actor.uid}"
            raise ValueError(msg)
    return actors


_ActorsByUid = Annotated[
    dict[Annotated[UUID, Field(strict=False)], Actor],
    AfterValidator(_validate_actor_mapping_keys),
]


def _environment_actor(runtime: RuntimeInfo) -> Actor:
    """Return this process's default actor for the stable runtime facts."""
    metadata = runtime.model_dump(
        mode="json",
        include={"hostname", "username", "platform", "python_prefix", "timezone"},
    )
    return _cached_environment_actor(
        f"{runtime.username}@{runtime.hostname}", json.dumps(metadata, sort_keys=True)
    )


@lru_cache(maxsize=128)
def _cached_environment_actor(name: str, metadata: str) -> Actor:
    return Actor(
        uid=uuid4(), name=name, meta=json.loads(metadata), created_at=datetime.now(UTC)
    )
