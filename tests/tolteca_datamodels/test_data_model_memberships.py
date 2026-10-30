"""Immutable, unordered data-model memberships."""

import pytest
from pydantic import ValidationError

from tolteca_datamodels.core.types import DataModelsT, FrozenDataModel


@pytest.mark.parametrize(
    "names",
    [
        ["lmt", "external", "lmt"],
        ("lmt", "external"),
        {"lmt", "external"},
        frozenset({"lmt", "external"}),
    ],
)
def test_data_model_memberships_are_immutable_sets_without_registration(names):
    class Reading(FrozenDataModel):
        data_models: DataModelsT

    reading = Reading(data_models=names)
    assert isinstance(reading.data_models, frozenset)
    assert reading.data_models == {"lmt", "external"}
    assert Reading.model_validate(reading.model_dump(round_trip=True)) == reading
    assert Reading.model_validate_json(reading.model_dump_json()) == reading


@pytest.mark.parametrize("value", ["toltec", None, [1]])
def test_data_model_memberships_require_a_collection_of_strings(value):
    class Reading(FrozenDataModel):
        data_models: DataModelsT

    with pytest.raises(ValidationError):
        Reading(data_models=value)
