"""Concrete observation records and shared record ownership."""

import importlib

import pytest
from pydantic import ValidationError

from tolteca_datamodels.data_prod import DataProd
from tolteca_datamodels.data_prod.metadata import RawObsMeta


def test_shared_records_have_one_owner():
    core = importlib.import_module("tolteca_datamodels.core.data_prod")
    concrete = importlib.import_module("tolteca_datamodels.data_prod")
    for name in set(core.__all__) & set(concrete.__all__):
        assert getattr(core, name) is getattr(concrete, name)


@pytest.mark.parametrize("missing", ["master", "obsnum", "subobsnum", "scannum"])
def test_portable_raw_identity_requires_explicit_quartet(missing):
    metadata = {
        "name": "raw",
        "data_prod_type": "dp_raw_obs",
        "master": "tcs",
        "obsnum": 1,
        "subobsnum": 0,
        "scannum": 0,
    }
    del metadata[missing]
    with pytest.raises(ValidationError) as exc:
        DataProd.model_validate({"uid": "tcs-1-0-0", "meta": metadata})
    [error] = exc.value.errors()
    assert (error["loc"], error["type"]) == (("meta", "dp_raw_obs", missing), "missing")
    with pytest.raises(ValidationError) as exc:
        RawObsMeta(**metadata)  # ty: ignore[invalid-argument-type]
    [error] = exc.value.errors()
    assert (error["loc"], error["type"]) == ((missing,), "missing")


def test_record_rejects_unknown_metadata_type_without_attribute_error():
    with pytest.raises(ValidationError, match="Unable to extract tag"):
        DataProd.model_validate({"uid": "invalid", "meta": 3})
