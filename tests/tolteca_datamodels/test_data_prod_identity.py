"""Product UID normalization and global versus namespace-scoped identity."""

from uuid import UUID

import pytest

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdRef,
    make_obs_product_uid,
    make_raw_obs_product_uid,
    normalize_primary_uid,
)
from tolteca_datamodels.data_prod.metadata import (
    CalGroupMeta,
    NamedGroupMeta,
    ObsMeta,
    RawObsMeta,
    ScienceMeta,
    VersionGroupMeta,
)


@pytest.mark.parametrize(
    ("make_uid", "numbers", "master", "expected"),
    [
        (make_obs_product_uid, (145609,), "TCS", "tcs-145609"),
        (make_raw_obs_product_uid, (145609, 0, 1), "ICS", "ics-145609-0-1"),
        (make_obs_product_uid, (0,), "Control2", "control2-0"),
        (make_raw_obs_product_uid, (0, 0, 0), "Control2", "control2-0-0-0"),
    ],
)
def test_uid_constructors_produce_canonical_identifiers(
    make_uid, numbers, master, expected
):
    uid = make_uid(master, *numbers)
    assert uid == expected
    assert normalize_primary_uid(uid) == uid


@pytest.mark.parametrize("master", [None, "", "123", "2tcs", "é", "tcs-a", "tcs\n"])
def test_uid_constructors_share_the_normalizers_master_rule(master):
    with pytest.raises(ValueError, match="requires a master"):
        make_obs_product_uid(master, 1)
    with pytest.raises(ValueError, match="requires a master"):
        make_raw_obs_product_uid(master, 1, 0, 1)
    for numbers in ((1,), (1, 0, 1)):
        if isinstance(master, str):
            with pytest.raises(ValueError, match="Invalid primary UID"):
                normalize_primary_uid(master + "-" + "-".join(map(str, numbers)))


@pytest.mark.parametrize(
    ("make_uid", "numbers"),
    [
        (make_obs_product_uid, (True,)),
        (make_obs_product_uid, ("1",)),
        (make_obs_product_uid, (-1,)),
        (make_raw_obs_product_uid, (1, -1, 1)),
        (make_raw_obs_product_uid, (1, 0, 1.5)),
    ],
)
def test_uid_constructors_require_nonnegative_integer_components(make_uid, numbers):
    with pytest.raises(ValueError, match="nonnegative integer"):
        make_uid("tcs", *numbers)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("TCS-00145609-000-0001", "tcs-145609-0-1"),
        ("ICS-00001", "ics-1"),
        (
            "B7F65F98-4F32-4E0A-A66A-CD278670F7CC",
            "b7f65f98-4f32-4e0a-a66a-cd278670f7cc",
        ),
    ],
)
def test_normalization_preserves_identity(value, expected):
    assert normalize_primary_uid(value) == expected
    assert normalize_primary_uid(expected) == expected


@pytest.mark.parametrize(
    "value", ["145609", "1-0-1", "tcs-1-2", "tcs-*", "tcs-1-0-1_proc"]
)
def test_primary_uid_rejects_non_entity_identifiers(value):
    with pytest.raises(ValueError, match="explicit master"):
        normalize_primary_uid(value)


@pytest.mark.parametrize(
    ("uid", "identity", "message"),
    [
        ("tcs-1", {}, "Field required"),
        ("tcs-1", {"master": "tcs", "obsnum": 2}, "must be"),
        ("tcs-1-0-1", {"master": "tcs", "obsnum": 1}, "must be"),
        ("tcs-1", {"master": "tcs", "obsnum": True}, "valid integer"),
    ],
)
def test_observation_record_requires_matching_identity(uid, identity, message):
    with pytest.raises(ValueError, match=message):
        DataProd.model_validate(
            {"uid": uid, "meta": {"name": uid, "data_prod_type": "dp_obs", **identity}}
        )


def test_raw_record_requires_matching_identity():
    meta = RawObsMeta(
        name="raw",
        data_prod_type="dp_raw_obs",
        master="tcs",
        obsnum=145609,
        subobsnum=0,
        scannum=1,
    )
    with pytest.raises(ValueError, match="Raw-observation UID must be"):
        DataProd(uid="tcs-145610-0-1", meta=meta)


@pytest.mark.parametrize(
    ("uid", "meta"),
    [
        (
            "TCS-0001",
            ObsMeta(name="obs", data_prod_type="dp_obs", master="TCS", obsnum=1),
        ),
        (
            "TCS-0001-000-0001",
            RawObsMeta(
                name="raw",
                data_prod_type="dp_raw_obs",
                master="TCS",
                obsnum=1,
                subobsnum=0,
                scannum=1,
            ),
        ),
    ],
)
def test_product_uid_normalization_preserves_recorded_master(uid, meta):
    product = DataProd(uid=uid, meta=meta)
    assert product.uid == normalize_primary_uid(uid)
    assert isinstance(product.meta, (ObsMeta, RawObsMeta))
    assert product.meta.master == "TCS"


@pytest.mark.parametrize(
    ("uid", "meta"),
    [
        ("tcs-1", ObsMeta(name="obs", data_prod_type="dp_obs", master="tcs", obsnum=1)),
        (
            "tcs-1-0-1",
            RawObsMeta(
                name="raw",
                data_prod_type="dp_raw_obs",
                master="tcs",
                obsnum=1,
                subobsnum=0,
                scannum=1,
            ),
        ),
        (
            "b7f65f98-4f32-4e0a-a66a-cd278670f7cc",
            NamedGroupMeta(name="collection", data_prod_type="dp_named_group"),
        ),
        (
            "calibration",
            CalGroupMeta(name="calibration", data_prod_type="dp_cal_group"),
        ),
        (
            "science-versions",
            VersionGroupMeta(
                name="versions",
                data_prod_type="dp_version_group",
                content_data_prod_type="dp_science",
            ),
        ),
    ],
)
def test_logical_products_have_global_identity(uid, meta):
    product = DataProd(uid=uid, meta=meta)
    assert product.reference == DataProdRef(uid=uid)
    with pytest.raises(ValueError, match="global logical product"):
        DataProd(
            uid=uid,
            meta=meta,
            namespace_uid=UUID("00000000-0000-0000-0000-000000000001"),
        )


def test_derived_product_can_be_unplaced_or_namespace_scoped():
    meta = ScienceMeta(name="science", data_prod_type="dp_science")
    draft = DataProd(uid="science/result", meta=meta)
    namespace_uid = UUID("00000000-0000-0000-0000-000000000001")
    placed = DataProd(uid=draft.uid, meta=meta, namespace_uid=namespace_uid)
    assert draft.reference == DataProdRef(uid=draft.uid)
    assert placed.reference == DataProdRef(uid=draft.uid, namespace_uid=namespace_uid)
