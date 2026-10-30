"""Shared data-model types."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["DataModelInfo", "DataModelsT", "FrozenDataModel"]


class DataModelInfo:
    """Identity and descriptive information about a data model.

    Attributes
    ----------
    name : str
        Name used in record memberships and data-model selection.
    """

    name: ClassVar[str]


type DataModelsT = Annotated[frozenset[str], Field(strict=False)]
"""Immutable membership set, accepted from collections of data-model names."""


class FrozenDataModel(BaseModel):
    """Read-only data model with strict validation of supplied and default values."""

    model_config = ConfigDict(strict=True, frozen=True, validate_default=True)
