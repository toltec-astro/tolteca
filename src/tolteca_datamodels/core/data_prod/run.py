"""Run identifiers, configurable generators and product tag selections.

Configure an optional UID and a readable name or counter policy::

    name = "nightly"
    name = CounterRunNameGenerator(value=42, key="nightly")

Counter values belong to one output namespace. A reservation consumes
its UID, name and counter value even when processing does not complete.
"""

from __future__ import annotations

from datetime import UTC, datetime
from string import Formatter
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    TypeAdapter,
)
from tollan.config import FrozenBaseModel, RuntimeInfo
from tollan.config.types import PathSegment

from .actor import Actor, _environment_actor

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "CounterRunNameFormat",
    "CounterRunNameGenerator",
    "CounterRunNameState",
    "PublicationRequest",
    "RunName",
    "RunUid",
    "StoreRun",
    "WorkflowRun",
]


def _validate_run_name(value: str) -> str:
    if value in {
        "index.yaml",
        ".staging",
        ".members",
        ".inline-collect.lock",
        ".publish.lock",
    } or (value.startswith(".tag-") and value.endswith(".lock")):
        msg = f"Run name is reserved by the product store: {value!r}"
        raise ValueError(msg)
    return value


type RunName = Annotated[PathSegment, AfterValidator(_validate_run_name)]
"""Namespace-unique run name usable as a product subdirectory name."""
_validate_run_name_value = TypeAdapter(RunName).validate_python


def _validate_counter_format(value: str) -> str:
    fields = {name for _, name, _, _ in Formatter().parse(value) if name is not None}
    if "value" not in fields or fields - {"value", "key"}:
        msg = "Counter format requires {value} and permits only {value} and {key}"
        raise ValueError(msg)
    try:
        _validate_run_name_value(value.format(value=0, key="run"))
    except (KeyError, ValueError, IndexError) as exc:
        msg = f"Invalid counter format: {value!r}"
        raise ValueError(msg) from exc
    return value


type CounterRunNameFormat = Annotated[str, AfterValidator(_validate_counter_format)]
"""Run-name format containing ``{value}`` and optionally ``{key}``."""


class CounterRunNameState(FrozenBaseModel):
    """Next automatic counter value for one namespace."""

    model_config = ConfigDict(extra="forbid")
    next: NonNegativeInt = 0


class CounterRunNameGenerator(FrozenBaseModel):
    """Generate a run name from a namespace's persistent counter.

    Parameters
    ----------
    generator : {'counter'}, optional
        Generator selection in configuration dictionaries.
    value : nonnegative int or None, optional
        Exact unused value. None allocates the next available value (default).
    format : str, optional
        Name format, default ``{value:04d}-{key}``. Requires ``{value}`` and
        optionally includes ``{key}``; standard Python format specs are supported.
    key : str or None, optional
        Naming suffix. None uses the invocation's UID.
    """

    model_config = ConfigDict(extra="forbid")
    generator: Literal["counter"] = "counter"
    value: NonNegativeInt | None = None
    format: CounterRunNameFormat = "{value:04d}-{key}"
    key: PathSegment | None = None

    def _allocate(
        self,
        state: dict[str, CounterRunNameState],
        is_value_available: Callable[[int], bool],
        is_available: Callable[[str], bool],
    ) -> tuple[str, Self]:
        counter = state.get("counter", CounterRunNameState())
        value = counter.next if self.value is None else self.value
        key = self.key if self.key is not None else str(uuid4())
        while True:
            run_name = _validate_run_name_value(
                self.format.format(value=value, key=key)
            )
            if is_value_available(value) and is_available(run_name):
                break
            if self.value is not None:
                msg = (
                    f"Counter value {value} or run name {run_name!r} "
                    "is already reserved"
                )
                raise FileExistsError(msg)
            value += 1
        state["counter"] = CounterRunNameState(next=max(counter.next, value + 1))
        return run_name, self.model_copy(update={"value": value, "key": key})


type RunUid = Annotated[UUID, Field(strict=False)]
"""Stable portable UUID of one invocation, independent of its name and location."""


class PublicationRequest(FrozenBaseModel):
    """Outcome of one product materialization in a destination."""

    model_config = ConfigDict(extra="forbid")
    uid: str
    data_prod_type: str
    primary_uid: str
    tag: PathSegment | None = None
    status: Literal["pending", "completed", "failed"] = "pending"
    message: str | None = None


class StoreRun(FrozenBaseModel):
    """A store's namespace-local record of an execution and its outputs.

    The containing product or namespace index supplies the namespace identity.
    One record can describe several products sharing the same local run name.
    The store creates it when a run name is requested and updates it during
    materialization. An execution with no products has an empty output record.

    Parameters
    ----------
    run_uid : UUID
        Identity of the execution recorded by WorkflowRun.
    name : str
        Reserved local name used beneath each product directory. This is a path
        component, not a directory containing all products from the execution.
    generator : CounterRunNameGenerator or None, optional
        Resolved counter policy and value, if counter naming was requested.
    publications : dict of str to PublicationRequest, optional
        Requested product identities and their materialization outcomes.
    outputs : dict of str to Any, optional
        Provenance supplied by each product's output writer.
    """

    model_config = ConfigDict(extra="forbid")
    run_uid: RunUid
    name: RunName
    generator: CounterRunNameGenerator | None = None
    publications: dict[str, PublicationRequest] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)


class WorkflowRun(BaseModel):
    """One execution, independent of its output namespaces and destinations.

    Use ``create`` to capture execution provenance. A context scope records
    completion, failure or cancellation without performing filesystem or DB I/O.
    Call ``fail`` for failures reported without an exception.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
        validate_assignment=True,
    )
    uid: RunUid = Field(frozen=True)
    actor: Actor = Field(frozen=True)
    time_created: AwareDatetime = Field(strict=False, frozen=True)
    time_completed: AwareDatetime | None = Field(default=None, strict=False)
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    meta: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)

    @property
    def actor_uid(self) -> UUID:
        """Identity of the execution actor."""
        return self.actor.uid

    @property
    def runtime_info(self) -> RuntimeInfo:
        """Recorded invocation environment."""
        return RuntimeInfo.model_validate(self.meta["runtime_info"])

    @classmethod
    def create(
        cls,
        *,
        uid: UUID | str | None = None,
        runtime_info: RuntimeInfo | None = None,
        context: dict[str, Any] | None = None,
        actor: Actor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Self:
        """Create an execution record without selecting a persistence destination.

        Parameters
        ----------
        uid : UUID, str or None, optional
            Execution UUID or its string representation. Only None generates
            a UUID; an invalid supplied value raises a validation error.
        runtime_info : RuntimeInfo or None, optional
            Invocation environment. None captures the current environment.
        context : dict or None, optional
            Resolved workflow configuration and input provenance. None uses an
            empty dictionary.
        actor : Actor or None, optional
            Reusable execution actor. None uses this process's actor for the
            captured environment.
        meta : dict or None, optional
            Execution metadata, including orchestration correlation values.
            None uses an empty dictionary. The captured runtime is stored under
            ``runtime_info``, replacing a supplied value for that key.

        Returns
        -------
        WorkflowRun
            Running execution record; no files are created.

        Raises
        ------
        pydantic.ValidationError
            Supplied record values or declared defaults fail validation.
        """
        runtime = RuntimeInfo() if runtime_info is None else runtime_info
        now = datetime.now(UTC)
        if actor is None:
            actor = _environment_actor(runtime)
        run = cls.model_validate(
            {
                "uid": uuid4() if uid is None else uid,
                "actor": actor,
                "time_created": now,
                "context": {} if context is None else context,
                "meta": {} if meta is None else meta,
            }
        )
        run.meta["runtime_info"] = runtime.model_dump(
            mode="json", exclude={"config_sources", "validation_context"}
        )
        return run

    def fail(self) -> None:
        """Mark a reported step or required output failure."""
        self.status = "failed"

    def __enter__(self) -> Self:
        if self.time_completed is not None:
            msg = "An execution cannot be entered again after completion"
            raise ValueError(msg)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
            self.status = "cancelled"
        elif exc is not None:
            self.status = "failed"
        elif self.status == "running":
            self.status = "completed"
        self.time_completed = datetime.now(UTC)
        return False
