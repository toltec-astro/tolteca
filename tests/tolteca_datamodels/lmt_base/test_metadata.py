"""Acquisition time and duration contracts on shared LMT metadata."""

import astropy.units as u
import pytest
from astropy.time import Time
from pydantic import ValidationError

from tolteca_datamodels.lmt_base.metadata import LmtRawObsMetadata


@pytest.fixture
def acquisition_data():
    return {
        "master": "tcs",
        "obsnum": 145609,
        "subobsnum": 0,
        "scannum": 2,
        "data_models": ["lmt", "toltec"],
        "instru": "lmt",
        "interface_kind": "tel",
        "interface": "tel",
        "t0": Time("2025-10-31T09:27:56"),
        "t1": Time("2025-10-31T09:30:56"),
        "t_int": 2 * u.min,
    }


def test_acquisition_times_and_duration_survive_json_roundtrip(acquisition_data):
    meta = LmtRawObsMetadata.model_validate(acquisition_data)
    restored = LmtRawObsMetadata.model_validate_json(
        meta.model_dump_json(round_trip=True)
    )
    for record in (meta, restored):
        assert record.t0 == acquisition_data["t0"]
        assert record.t1 == acquisition_data["t1"]
        assert record.t_int == 120 * u.s
        assert record.t_elapsed == (record.t1 - record.t0).to(u.s)
        assert record.t_elapsed != record.t_int


@pytest.mark.parametrize("name", ["t0", "t1", "t_int"])
def test_acquisition_time_fields_are_required(acquisition_data, name):
    LmtRawObsMetadata.model_validate(acquisition_data)
    del acquisition_data[name]
    with pytest.raises(ValidationError) as caught:
        LmtRawObsMetadata.model_validate(acquisition_data)
    assert [(e["loc"], e["type"]) for e in caught.value.errors()] == [
        ((name,), "missing")
    ]


def test_integration_duration_requires_time_units(acquisition_data):
    LmtRawObsMetadata.model_validate(acquisition_data)
    with pytest.raises(ValidationError) as caught:
        LmtRawObsMetadata.model_validate(acquisition_data | {"t_int": 2 * u.m})
    assert {e["loc"] for e in caught.value.errors()} == {("t_int",)}


def test_acquisition_schema_does_not_include_original_filepath():
    schema = LmtRawObsMetadata.model_json_schema(mode="serialization")
    assert not {"filepath_orig", "filepath_info_orig"} & schema["properties"].keys()
