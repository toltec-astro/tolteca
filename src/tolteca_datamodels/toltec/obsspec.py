"""Typed observation selectors shared by catalogs and database queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Self

import pandas as pd

from .types import ToltecInfo, ToltecMasterType

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


__all__ = ["IdRange", "IdSelector", "ObsSpec"]

_ID_FIELDS = ("obsnum", "subobsnum", "scannum", "roach")


@dataclass(frozen=True)
class IdRange:
    """A half-open range of identifier values, not catalog row positions.

    ``stop=None`` leaves the upper bound open. Steps are positive and anchored
    at ``start``; identifiers and bounds are nonnegative.
    """

    start: int = 0
    stop: int | None = None
    step: int = 1

    def __post_init__(self) -> None:
        for name in ("start", "stop", "step"):
            value = getattr(self, name)
            if name == "stop" and value is None:
                continue
            if type(value) is not int or value < (1 if name == "step" else 0):
                msg = f"Invalid identifier range {name}: {value!r}"
                raise ValueError(msg)

    def __str__(self) -> str:
        stop = "" if self.stop is None else str(self.stop)
        step = "" if self.step == 1 else f":{self.step}"
        return f"[{self.start}:{stop}{step}]"


type IdSelector = int | tuple[int, ...] | IdRange | None
"""Exact identifier, set, range, or unconstrained field (None)."""


def _parse_id(value: str) -> IdSelector:
    value = value.strip()
    if value in {"*", "{}", "[]", "[:]"}:
        return None
    if value.startswith("{") and value.endswith("}"):
        members = value[1:-1].split(",")
        if all(re.fullmatch(r"\s*\d+\s*", member) for member in members):
            return tuple(sorted({int(member) for member in members}))
    elif value.startswith("[") and value.endswith("]"):
        parts = [part.strip() for part in value[1:-1].split(":")]
        if len(parts) == 1 and parts[0].isdigit():
            return int(parts[0])
        if len(parts) in {2, 3} and all(not part or part.isdigit() for part in parts):
            return IdRange(
                start=int(parts[0]) if parts[0] else 0,
                stop=int(parts[1]) if parts[1] else None,
                step=int(parts[2]) if len(parts) == 3 and parts[2] else 1,  # noqa: PLR2004 -- start:stop:step grammar
            )
    elif value.isdigit():
        return int(value)
    msg = f"Invalid identifier selector: {value!r}"
    raise ValueError(msg)


def _format_id(value: IdSelector) -> str:
    if value is None:
        return "*"
    if isinstance(value, tuple):
        return "{" + ",".join(map(str, value)) + "}"
    return str(value)


@dataclass(frozen=True)
class ObsSpec:
    """Select observations and interfaces without resolving data locations.

    ``-`` assigns fields from the left: obsnum, subobsnum, scannum, roach.
    ``/`` right-aligns the remaining fields: ``1000/2/7`` selects scan 2,
    ROACH 7, and any subobservation. An optional master precedes the first
    ``-``. The final field also accepts an interface name such as ``toltec7``.

    Examples
    --------
    >>> ObsSpec.parse("tcs-1000-{0,2}/toltec7").subobsnum
    (0, 2)
    >>> str(ObsSpec.parse("1000/2/7"))
    '1000-*-2-7'
    """

    master: ToltecMasterType | None = None
    obsnum: IdSelector = None
    subobsnum: IdSelector = None
    scannum: IdSelector = None
    roach: IdSelector = None
    interface: str | None = None

    def __post_init__(self) -> None:  # noqa: C901 -- validate each independent selector
        if self.master is not None:
            object.__setattr__(self, "master", ToltecMasterType(self.master))
        for name in _ID_FIELDS:
            value = getattr(self, name)
            if value is None or isinstance(value, IdRange):
                continue
            members = value if isinstance(value, tuple) else (value,)
            if not members or any(
                type(item) is not int or item < 0 for item in members
            ):
                msg = f"Invalid {name} selector: {value!r}"
                raise ValueError(msg)
            if name == "roach" and any(
                item not in ToltecInfo.roaches for item in members
            ):
                msg = f"Unknown ROACH in selector: {value!r}"
                raise ValueError(msg)
            if isinstance(value, tuple):
                object.__setattr__(self, name, tuple(sorted(set(value))))
        if self.interface is not None:
            if self.roach is not None:
                msg = "Specify either interface or roach, not both"
                raise ValueError(msg)
            if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", self.interface):
                msg = f"Invalid interface: {self.interface!r}"
                raise ValueError(msg)
            if self.interface in ToltecInfo.interface_roach:
                object.__setattr__(
                    self, "roach", ToltecInfo.interface_roach[self.interface]
                )
                object.__setattr__(self, "interface", None)
            elif re.fullmatch(r"toltec\d+", self.interface):
                msg = f"Unknown ROACH interface: {self.interface}"
                raise ValueError(msg)

    @classmethod
    def parse(cls, value: str | int | Self) -> Self:
        """Parse a complete selector, raising ValueError for invalid syntax.

        Paths, filesystem globs, and latest-observation policy are handled by
        callers, not by this grammar. Empty input is not a wildcard.
        """
        if isinstance(value, cls):
            return value
        if type(value) is int:
            return cls(obsnum=value)
        if not isinstance(value, str):
            msg = f"Expected an ObsSpec string or integer, got {value!r}"
            raise TypeError(msg)
        text = value.strip()
        master = None
        prefix, sep, rest = text.partition("-")
        if sep and prefix in ToltecMasterType:
            master = ToltecMasterType(prefix)
            text = rest
        segments = text.split("/")
        left = segments[0].split("-")
        right = segments[1:]
        if (
            not text
            or any(not part for part in (*left, *right))
            or any("-" in part for part in right)
            or len(left) + len(right) > len(_ID_FIELDS)
        ):
            msg = f"Invalid observation selector: {value!r}"
            raise ValueError(msg)
        assignments = dict(zip(_ID_FIELDS, left, strict=False))
        if right:
            assignments.update(zip(_ID_FIELDS[-len(right) :], right, strict=True))
        result: dict = {"master": master}
        for name, token in assignments.items():
            if name == "roach" and re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", token):
                result["interface"] = token
            else:
                result[name] = _parse_id(token)
        return cls(**result)

    def __str__(self) -> str:
        values = [_format_id(getattr(self, name)) for name in _ID_FIELDS]
        if self.interface is not None:
            values[-1] = self.interface
        while len(values) > 1 and values[-1] == "*":
            values.pop()
        prefix = "" if self.master is None else f"{self.master}-"
        return prefix + "-".join(values)

    def constraints(self) -> Iterator[tuple[str, str | IdSelector]]:
        """Yield constrained metadata fields and their typed selectors."""
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                yield field.name, value

    def matches(self, record: Mapping[str, object]) -> bool:  # noqa: PLR0911 -- reject unmatched constraints early
        """Test a metadata record; missing constrained fields do not match."""
        for name, selection in self.constraints():
            value = record.get(name)
            if value is None:
                return False
            if isinstance(selection, IdRange):
                if not isinstance(value, int) or value < selection.start:
                    return False
                if selection.stop is not None and value >= selection.stop:
                    return False
                if (value - selection.start) % selection.step:
                    return False
            elif isinstance(selection, tuple):
                if value not in selection:
                    return False
            elif value != selection:
                return False
        return True

    def select(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """Select catalog rows without opening sources or changing their metadata."""
        mask = pd.Series(data=True, index=catalog.index, dtype=bool)
        for name, selection in self.constraints():
            if name not in catalog:
                msg = f"Catalog lacks selector field {name!r}"
                raise ValueError(msg)
            values = catalog[name]
            if isinstance(selection, IdRange):
                matched = (values >= selection.start) & (
                    (values - selection.start) % selection.step == 0
                )
                if selection.stop is not None:
                    matched &= values < selection.stop
            elif isinstance(selection, tuple):
                matched = values.isin(selection)
            else:
                matched = values == selection
            mask &= matched.fillna(value=False)
        return catalog.loc[mask]
