"""Shared data-model validation and explicit boundary conversions."""

from functools import partial

import pytest
from pydantic import ConfigDict, Field, ValidationError

from tolteca_datamodels.core.types import DataModelsT, FrozenDataModel
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt_base.metadata import LmtObsQuartetMixin
from tolteca_datamodels.lmt_base.types import (
    SeqNumT,  # noqa: TC001 -- Pydantic resolves the local model's annotation at runtime.
)
from tolteca_datamodels.toltec.filepath import ToltecFilepathInfo


def test_data_model_is_strict_frozen_and_validates_defaults():
    class Reading(FrozenDataModel):
        value: int

    with pytest.raises(ValidationError, match="valid integer"):
        Reading.model_validate({"value": "1"})
    reading = Reading(value=1)
    with pytest.raises(ValidationError, match="frozen"):
        setattr(reading, "value", 2)  # noqa: B010 -- Exercise frozen-model validation.

    class InvalidDefault(FrozenDataModel):
        value: int = Field(default="1")  # ty: ignore[invalid-assignment] -- Deliberately invalid default.

    with pytest.raises(ValidationError, match="valid integer"):
        InvalidDefault()


def test_data_model_allows_explicit_protocol_override():
    class EditableReading(FrozenDataModel):
        model_config = ConfigDict(frozen=False)
        value: int

    reading = EditableReading(value=1)
    setattr(reading, "value", 2)  # noqa: B010 -- Exercise the explicit mutable override.
    assert reading.value == 2
    with pytest.raises(ValidationError, match="valid integer"):
        EditableReading.model_validate({"value": "1"})


def test_data_model_memberships_are_unique_and_round_trip_without_registration():
    class Reading(FrozenDataModel):
        data_models: DataModelsT

    reading = Reading(data_models=["lmt", "external", "lmt"])
    assert reading.data_models == ["lmt", "external"]
    assert Reading.model_validate(reading.model_dump(round_trip=True)) == reading
    assert Reading.model_validate_json(reading.model_dump_json()) == reading


@pytest.mark.parametrize("value", ["toltec", None, [1], ("toltec",)])
def test_data_model_memberships_require_a_list_of_labels(value):
    class Reading(FrozenDataModel):
        data_models: DataModelsT

    with pytest.raises(ValidationError):
        Reading(data_models=value)


def test_filename_ingestion_converts_captures_but_model_stays_strict():
    info = FilepathInfo.from_source("toltec0_145609_000_0002_2025_10_31_09_27_57.nc")
    assert isinstance(info, ToltecFilepathInfo)
    assert info.obsnum == 145609
    assert info.scannum == 2
    data = info.model_dump(exclude_computed_fields=True)
    assert type(info).model_validate(data) == info
    data["obsnum"] = "145609"
    with pytest.raises(ValidationError, match="valid integer"):
        type(info).model_validate(data)


@pytest.mark.parametrize("use_model_validate", [False, True], ids=["init", "validate"])
@pytest.mark.parametrize("default", [-1, "1"], ids=["constraint", "strict_type"])
@pytest.mark.parametrize("use_factory", [False, True], ids=["value", "factory"])
def test_metadata_rejects_invalid_defaults(default, use_factory, use_model_validate):
    class InvalidDefault(LmtObsQuartetMixin):
        obsnum: SeqNumT = (
            Field(default_factory=lambda: default)
            if use_factory
            else Field(default=default)
        )

    data = {"master": "tcs", "subobsnum": 0, "scannum": 0}
    construct = (
        partial(InvalidDefault.model_validate, data)
        if use_model_validate
        else partial(InvalidDefault, **data)
    )
    with pytest.raises(ValidationError) as exc:
        construct()
    assert [(error["loc"], error["type"]) for error in exc.value.errors()] == [
        (("obsnum",), "int_type" if isinstance(default, str) else "greater_than_equal")
    ]
