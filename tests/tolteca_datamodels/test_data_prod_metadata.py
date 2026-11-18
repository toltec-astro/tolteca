"""Product metadata validation and its portable/database representations."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tolteca_datamodels.data_prod import DataProd, DataProdIndex
from tolteca_datamodels.data_prod.metadata import (
    AnyDataProdItemMeta,
    AnyDataProdMeta,
    AnyInterfaceMeta,
    ArtifactMeta,
    IndexItemMeta,
    RawObsMeta,
    RoachInterfaceMeta,
    ScienceMeta,
    TelInterfaceMeta,
    _retort,
)


@pytest.fixture
def raw_metadata():
    return RawObsMeta(
        name="tcs-145609-0-1",
        data_prod_type="dp_raw_obs",
        master="tcs",
        obsnum=145609,
        subobsnum=0,
        scannum=1,
        obs_datetime=datetime(2025, 10, 31, tzinfo=UTC),
        integration_time=12.5,
        m2_offset_mm=(1.0, 2.0, 3.0),
    )


def test_typed_metadata_construction_validates_field_values():
    with pytest.raises(ValidationError) as exc:
        ScienceMeta(name=123, data_prod_type="dp_science")  # ty: ignore[invalid-argument-type]
    [error] = exc.value.errors()
    assert (error["loc"], error["type"]) == (("name",), "string_type")


def test_metadata_enrichment_validates_assignment(raw_metadata: RawObsMeta) -> None:
    raw_metadata.source_name = "M31"
    assert raw_metadata.source_name == "M31"
    with pytest.raises(ValidationError):
        raw_metadata.integration_time = "12.5"  # ty: ignore[invalid-assignment]
    assert raw_metadata.integration_time == 12.5


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("integration_time", "12.5", "float_type"),
        ("obsnum", True, "int_type"),
        ("obs_goal_typo", "pointing", "extra_forbidden"),
    ],
)
def test_portable_and_adaptix_inputs_use_metadata_validation(
    raw_metadata, field, value, error_type
):
    values = raw_metadata.model_dump(round_trip=True)
    values[field] = value
    with pytest.raises(ValidationError) as exc:
        DataProd.model_validate({"uid": raw_metadata.name, "meta": values})
    [error] = exc.value.errors()
    assert (error["loc"], error["type"]) == (("meta", "dp_raw_obs", field), error_type)

    values = raw_metadata.model_dump(mode="json", round_trip=True)
    values[field] = value
    with pytest.raises(ValidationError) as exc:
        _retort.load(values, AnyDataProdMeta)
    [error] = exc.value.errors()
    assert (error["loc"], error["type"]) == (("raw_obs", field), error_type)


def test_product_metadata_preserves_values_through_adaptix_and_yaml(
    raw_metadata, tmp_path
):
    database_values = _retort.dump(raw_metadata, AnyDataProdMeta)
    assert _retort.load(database_values, AnyDataProdMeta) == raw_metadata
    index = DataProdIndex(product=DataProd(uid=raw_metadata.name, meta=raw_metadata))
    restored = DataProdIndex.read(index.write(tmp_path / "index.yaml"))
    assert restored.product.meta == raw_metadata
    assert isinstance(restored.product.meta, RawObsMeta)
    assert restored.product.meta.m2_offset_mm == (1.0, 2.0, 3.0)


def test_partial_interface_evidence_does_not_require_product_identity():
    roach = RoachInterfaceMeta(interface="toltec7", cal_obsnum=145608)
    telescope = TelInterfaceMeta(tau=0.08)
    for meta in (roach, telescope):
        assert (
            _retort.load(_retort.dump(meta, AnyInterfaceMeta), AnyInterfaceMeta) == meta
        )
    assert roach.cal_master is None
    assert roach.cal_subobsnum is None
    assert roach.cal_scannum is None
    assert telescope.obs_goal is None


_ARTIFACT = {
    "type": "artifact",
    "format": "ecsv",
    "name": "tones",
    "producer": "tolteca_kids.tone_prep:TonePrep",
    "run_uid": "4b2d6e1a-6f0c-4a52-9f5e-0d6c5b7c1e2a",
    "interface": "toltec0",
    "scope": "blocks/0",
    "context": {"grouping": "pairs"},
}
_NAMESPACE_UID = "9a3f1c52-2b7d-4e8a-a0c4-61d2f5e8b730"


@pytest.mark.parametrize(
    "values",
    [
        _ARTIFACT,
        {"type": "data_prod_index", "namespace_uid": _NAMESPACE_UID},
        {
            "type": "data_prod_index",
            "namespace_uid": _NAMESPACE_UID,
            "run_uid": _ARTIFACT["run_uid"],
        },
    ],
)
def test_item_metadata_keeps_its_written_shape_through_adaptix(values):
    meta = _retort.load(values, AnyDataProdItemMeta)
    assert isinstance(meta, ArtifactMeta | IndexItemMeta)
    assert _retort.dump(meta, AnyDataProdItemMeta) == values


@pytest.mark.parametrize(
    ("change", "error_type"),
    [
        ({"name": None}, "missing"),
        ({"producer": None}, "missing"),
        ({"run_uid": None}, "missing"),
        ({"grouping": "pairs"}, "extra_forbidden"),
    ],
)
def test_artifact_metadata_requires_selection_keys(change, error_type):
    values = {k: v for k, v in {**_ARTIFACT, **change}.items() if v is not None}
    with pytest.raises(ValidationError) as exc:
        _retort.load(values, AnyDataProdItemMeta)
    [error] = exc.value.errors()
    assert error["type"] == error_type
