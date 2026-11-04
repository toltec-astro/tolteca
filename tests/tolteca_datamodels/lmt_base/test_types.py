"""Shared observation value types and model-specific validation policies."""

from typing import get_args

import pytest
from pydantic import ValidationError

from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tolteca_datamodels.lmt.types import LmtInfo, LmtRawObsMasterCode
from tolteca_datamodels.lmt_base.metadata import (
    LmtObsQuartetMixin,
    LmtRawObsMetadata,
)
from tolteca_datamodels.lmt_base.types import LabelT
from tolteca_datamodels.toltec.hwpr import ToltecHwprMetadata
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsMetadata,
)
from tolteca_datamodels.toltec.toltec_db import ToltecDBRawObsMasterCode
from tolteca_datamodels.toltec.types import ToltecInfo


@pytest.mark.parametrize("name", ["obsnum", "subobsnum", "scannum"])
@pytest.mark.parametrize("value", [-1, None, "1"])
def test_metadata_rejects_invalid_observation_numbers(name, value):
    # Deliberately invalid input tests range, requiredness, and strict typing.
    data = {"master": "tcs", "obsnum": 145609, "subobsnum": 0, "scannum": 2}
    with pytest.raises(ValidationError, match=name):
        LmtObsQuartetMixin.model_validate(data | {name: value})
    with pytest.raises(ValidationError, match=name):
        LmtObsQuartetMixin(
            **(data | {name: value})  # ty: ignore[invalid-argument-type]
        )


def test_zero_identifiers_and_json_roundtrip():
    meta = LmtObsQuartetMixin(master="tcs", obsnum=0, subobsnum=0, scannum=0)
    assert (meta.obsnum, meta.subobsnum, meta.scannum) == (0, 0, 0)
    assert type(meta).model_validate_json(meta.model_dump_json(round_trip=True)) == meta


@pytest.mark.parametrize(
    ("model", "label_type", "code_type"),
    [
        (LmtTelMetadata, LmtInfo.MasterT, LmtRawObsMasterCode),
        (ToltecRawObsMetadata, ToltecInfo.MasterT, ToltecDBRawObsMasterCode),
        (ToltecHwprMetadata, ToltecInfo.MasterT, ToltecDBRawObsMasterCode),
    ],
)
def test_concrete_master_is_a_label_and_code_is_an_integer(
    model, label_type, code_type
):
    assert model.model_fields["master"].annotation is label_type
    assert model.model_fields["master"].is_required()
    assert model.model_fields["master_code"].annotation is code_type
    assert model.model_fields["master_code"].is_required()
    assert "master" not in model.model_computed_fields


@pytest.mark.parametrize(
    "model", [LmtRawObsMetadata, LmtTelMetadata, ToltecRawObsMetadata]
)
@pytest.mark.parametrize("name", ["uid_obs", "uid_raw_obs", "uid_raw_obs_file"])
def test_raw_observation_uids_are_nullable_labels(model, name):
    assert set(get_args(model.model_computed_fields[name].return_type)) == {
        LabelT,
        type(None),
    }
