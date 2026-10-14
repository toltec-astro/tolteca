"""Observation selector field constraints and layout for LMT data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

from pydantic import PlainSerializer, PlainValidator

from ..core.obs_spec import FieldConstraint, ObsSpecBase, Selector

__all__ = ["ObsSpec", "ObsSpecField"]


@dataclass(frozen=True)
class ObsSpec(ObsSpecBase):
    """Select LMT observations and interfaces.

    All fields default to None, leaving them unconstrained.

    Parameters
    ----------
    master : Selector, optional
        Acquisition master label or tuple of labels.
    obsnum, subobsnum, scannum : Selector, optional
        Nonnegative integer values, sets, bounds or positional selections.
    interface : Selector, optional
        Labels, numeric components, mixed sets, bounds or positional selections.
        Literal parts around a numeric selection specify its position in a
        label, for example ``foo(1,5)[-1]bar``. Bare numeric constraints on
        labels require a context with numeric interpretations when the selector
        is applied.

    Examples
    --------
    >>> ObsSpec.parse("tcs-1-1-1").obsnum
    1
    >>> ObsSpec.parse("*-1").subobsnum
    1
    """

    master: Selector = None
    obsnum: Selector = None
    subobsnum: Selector = None
    scannum: Selector = None
    interface: Selector = None

    _field_constraints: ClassVar = {
        "master": FieldConstraint(kind="label"),
        **{
            name: FieldConstraint(kind="integer")
            for name in ("obsnum", "subobsnum", "scannum")
        },
        "interface": FieldConstraint(kind="mixed"),
    }
    _primary_field: ClassVar = "obsnum"
    # ASCII digits and selector punctuation; the shared parser validates syntax.
    _primary_pattern: ClassVar = re.compile(r"[0-9*{}()\[\],:+-]+")


def _validate_obs_spec(value: Any) -> ObsSpec:
    try:
        return ObsSpec.parse(value)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


ObsSpecField = Annotated[
    ObsSpec,
    PlainValidator(_validate_obs_spec, json_schema_input_type=str | int),
    PlainSerializer(str, return_type=str),
]
"""Observation selector accepting text, an observation number, or an ObsSpec.

Validated values are ObsSpec instances; dumps contain normalized selector text.
"""
