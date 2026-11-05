"""Shared data-model validation and explicit boundary conversions."""

import pytest
from pydantic import ConfigDict, Field, ValidationError

from tolteca_datamodels.core.types import FrozenDataModel
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


def test_filepath_information_requires_declared_data_model_compatibility():
    with pytest.raises(ValidationError, match="data_models"):
        FilepathInfo.model_validate({})
    assert FilepathInfo(ext="nc", data_models=frozenset()).data_models == frozenset()
    with pytest.raises(ValidationError, match="data_models"):
        FilepathInfo.model_validate({"ext": "nc", "data_models": None})


def test_data_model_memberships_accept_lists_at_validation_boundary():
    info = FilepathInfo.model_validate(
        {"ext": "nc", "data_models": ["toltec", "lmt", "toltec"]}
    )
    assert isinstance(info.data_models, frozenset)
    assert info.data_models == frozenset({"lmt", "toltec"})


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


@pytest.mark.parametrize("default", [-1, "1"], ids=["constraint", "strict_type"])
@pytest.mark.parametrize("use_factory", [False, True], ids=["value", "factory"])
def test_metadata_rejects_invalid_defaults(default, use_factory):
    class InvalidDefault(LmtObsQuartetMixin):
        obsnum: SeqNumT = (
            Field(default_factory=lambda: default)
            if use_factory
            else Field(default=default)
        )

    data = {"master": "tcs", "subobsnum": 0, "scannum": 0}
    with pytest.raises(ValidationError, match="obsnum"):
        InvalidDefault.model_validate(data)
