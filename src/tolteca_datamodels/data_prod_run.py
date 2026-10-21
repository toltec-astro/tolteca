"""Run identifiers, configurable generators and product tag selections.

Configure an optional UID and a readable name or counter policy::

    run = RunRequest(name="nightly")
    run = RunRequest(name=CounterRunNameGenerator(value=42, key="nightly"))

Counter values belong to one output namespace. A reservation consumes
its UID, name and counter value even when processing does not complete.
"""

from __future__ import annotations

from string import Formatter
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    AfterValidator,
    AwareDatetime,
    ConfigDict,
    Field,
    NonNegativeInt,
    TypeAdapter,
)
from tollan.config import FrozenBaseModel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
    "CounterRunNameFormat",
    "CounterRunNameGenerator",
    "CounterRunNameState",
    "DataProdRun",
    "PathComponent",
    "PublicationRequest",
    "RunName",
    "RunRequest",
    "RunUid",
    "prepare_run_request",
]


def _validate_component(value: str) -> str:
    if not value or value in {".", ".."} or any(c in value for c in "/\\\x00"):
        msg = f"Expected one path component: {value!r}"
        raise ValueError(msg)
    return value


type PathComponent = Annotated[str, AfterValidator(_validate_component)]
"""One filename component, excluding dot references, separators and NUL."""


def _validate_run_name(value: str) -> str:
    if value in {"index.yaml", ".staging", ".publish.lock"} or (
        value.startswith(".tag-") and value.endswith(".lock")
    ):
        msg = f"Run name is reserved by the product store: {value!r}"
        raise ValueError(msg)
    return value


type RunName = Annotated[PathComponent, AfterValidator(_validate_run_name)]
"""Namespace-unique run name usable as a product subdirectory name."""
_run_name_adapter = TypeAdapter(RunName)


def _validate_counter_format(value: str) -> str:
    fields = {name for _, name, _, _ in Formatter().parse(value) if name is not None}
    if "value" not in fields or fields - {"value", "key"}:
        msg = "Counter format requires {value} and permits only {value} and {key}"
        raise ValueError(msg)
    try:
        _run_name_adapter.validate_python(value.format(value=0, key="run"))
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
    key: PathComponent | None = None

    def _allocate(
        self,
        state: dict[str, CounterRunNameState],
        runs: Mapping[UUID, DataProdRun],
        is_available: Callable[[str], bool],
    ) -> tuple[str, Self]:
        counter = state.get("counter", CounterRunNameState())
        used = {
            run.generator.value
            for run in runs.values()
            if isinstance(run.generator, CounterRunNameGenerator)
        }
        value = counter.next if self.value is None else self.value
        key = self.key if self.key is not None else str(uuid4())
        while True:
            run_name = _run_name_adapter.validate_python(
                self.format.format(value=value, key=key)
            )
            if value not in used and is_available(run_name):
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


class RunRequest(FrozenBaseModel):
    """Identity and naming policy for a new invocation.

    Parameters
    ----------
    uid : UUID or None, optional
        Caller-assigned portable identity. None generates a UUID once per invocation.
    name : str or CounterRunNameGenerator or None, optional
        Namespace-unique directory name or counter policy. None uses the run UID
        as text. Names remain reserved after failed or unfinished invocations.
    """

    model_config = ConfigDict(extra="forbid")
    uid: RunUid | None = None
    name: RunName | CounterRunNameGenerator | None = None


def prepare_run_request(run: RunRequest, uid: UUID) -> RunRequest:
    """Resolve an invocation's UID and omitted naming values.

    Parameters
    ----------
    run : RunRequest
        Caller identity and naming policy.
    uid : UUID
        Identity generated by the driver. Used only when the request omits uid.

    Returns
    -------
    RunRequest
        Request with a fixed UID and name or counter policy. An omitted counter
        key uses the UID. Counter allocation occurs under the namespace lock.
    """
    uid = run.uid or uid
    name = run.name
    if name is None:
        name = str(uid)
    elif isinstance(name, CounterRunNameGenerator) and name.key is None:
        name = name.model_copy(update={"key": str(uid)})
    return RunRequest(uid=uid, name=name)


class PublicationRequest(FrozenBaseModel):
    """One requested version-group publication within an invocation."""

    model_config = ConfigDict(extra="forbid")
    uid: str
    data_prod_type: str
    primary_uid: str
    tag: PathComponent | None = None
    status: Literal["pending", "completed", "failed"] = "pending"
    message: str | None = None


class DataProdRun(FrozenBaseModel):
    """Invocation identity, effective configuration and publication outcomes.

    UIDs identify invocations across stores and databases. Names are unique
    within ``namespace_uid``. Failed and unfinished runs retain
    their reservation. ``context`` records resolved workflow values and inputs;
    ``publications`` records requested outputs, including those never committed.
    """

    model_config = ConfigDict(extra="forbid")
    uid: RunUid = Field(description="Stable portable invocation identity.")
    name: RunName = Field(description="Reserved name within the namespace.")
    namespace_uid: UUID = Field(strict=False)
    generator: CounterRunNameGenerator | None = Field(
        default=None,
        description="Resolved allocation; None without counter allocation.",
    )
    time_created: AwareDatetime = Field(strict=False, description="Reservation time.")
    time_completed: AwareDatetime | None = Field(default=None, strict=False)
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    actor_uid: UUID = Field(strict=False)
    meta: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    publications: dict[str, PublicationRequest] = Field(default_factory=dict)
