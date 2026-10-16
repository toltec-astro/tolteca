"""Shared data-model types."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, ConfigDict

__all__ = ["DataModelInfo", "DataModelsT", "FrozenDataModel"]


class DataModelInfo:
    """Identity and descriptive information about a data model.

    Attributes
    ----------
    name : str
        Name used in record memberships and data-model selection.
    """

    name: ClassVar[str]


def _unique_data_models(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


type DataModelsT = Annotated[list[str], AfterValidator(_unique_data_models)]
"""Applicable data-model names, without duplicates or precedence by order."""


class FrozenDataModel(BaseModel):
    """Read-only data model with strict validation of supplied and default values."""

    model_config = ConfigDict(strict=True, frozen=True, validate_default=True)
