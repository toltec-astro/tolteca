"""Primary resolution through portable typed product associations."""

import pytest

from tolteca_datamodels.data_prod import (
    DataProdIndex,
    make_obs_product_uid,
    make_raw_obs_product_uid,
    normalize_primary_uid,
)
from tolteca_datamodels.data_prod_store import DataProdStore


@pytest.fixture
def products():
    raw = DataProdIndex(
        uid="TCS-00145609-000-0001",
        meta={
            "name": "raw",
            "data_prod_type": "dp_raw_obs",
            "master": "tcs",
            "obsnum": 145609,
            "subobsnum": 0,
            "scannum": 1,
        },
        assocs=[
            {
                "data_prod_assoc_type": "dpa_raw_obs_cal_obs",
                "uid": "tcs-145605-0-1",
            }
        ],
    )
    proc = DataProdIndex(
        uid=f"{raw.uid}_proc",
        meta={"name": "processed", "data_prod_type": "dp_raw_obs_proc"},
        assocs=[
            {
                "uid": raw.uid,
                "data_prod_assoc_type": "dpa_raw_obs_proc_raw_obs",
                "is_primary": True,
            }
        ],
    )
    group = DataProdIndex(
        uid="dp_cal_group_tcs-145609-0-1_g2",
        meta={"name": "group", "data_prod_type": "dp_cal_group"},
        assocs=[
            {
                "uid": "unloaded-member",
                "data_prod_assoc_type": "dpa_cal_group_raw_obs_proc",
            },
            {
                "uid": proc.uid,
                "data_prod_assoc_type": "dpa_cal_group_raw_obs_proc",
                "is_primary": True,
            },
        ],
    )
    return {item.reference: item for item in (raw, proc, group)}


def test_primary_chain_survives_yaml_and_selects_store_placement(products, tmp_path):
    raw, proc, group = products.values()
    calls = []

    def lookup(uid):
        calls.append(uid)
        return products[uid]

    primary = group.get_primary(lookup)
    assert primary is raw
    assert calls == [proc.reference, raw.reference]
    assert raw.get_primary() is raw
    assert group.uid == "dp_cal_group_tcs-145609-0-1_g2"
    store = DataProdStore(tmp_path, namespace="ql")
    path = group.write(
        store.get_product_path(primary.uid, "dp_cal_group", uid=group.uid)
    )
    assert path == tmp_path / "ql/145609-0-1/dp_cal_group_145609-0-1_g2/index.yaml"
    restored = DataProdIndex.read(path)
    assert restored == group
    assert restored.get_primary(lookup) is raw
    assert not {"primary", "primary_uid", "uid_primary"} & restored.model_dump().keys()


def test_observation_is_terminal_even_with_associations(products):
    raw, _proc, group = products.values()
    obs = DataProdIndex(
        uid="TCS-00145609",
        meta={
            "name": "observation",
            "data_prod_type": "dp_obs",
            "master": "tcs",
            "obsnum": 145609,
        },
        assocs=[{"uid": raw.uid, "data_prod_assoc_type": "dpa_obs_raw_obs"}],
    )
    assert obs.uid == "tcs-145609"
    assert obs.get_primary() is obs
    group.assocs[1].uid = obs.uid
    group.assocs[1].data_prod_assoc_type = "dpa_cal_group_obs"
    products[obs.reference] = obs
    assert group.get_primary(products.__getitem__) is obs


@pytest.mark.parametrize(
    ("uid", "identity", "message"),
    [
        ("tcs-1", {}, "requires master and obsnum"),
        ("tcs-1", {"master": "tcs", "obsnum": 2}, "must be"),
        ("tcs-1-0-1", {"master": "tcs", "obsnum": 1}, "must be"),
        ("tcs-1", {"master": "tcs", "obsnum": True}, "nonnegative"),
    ],
)
def test_observation_index_requires_matching_identity(uid, identity, message):
    with pytest.raises(ValueError, match=message):
        DataProdIndex(
            uid=uid,
            meta={"name": uid, "data_prod_type": "dp_obs", **identity},
        )


@pytest.mark.parametrize("count", [0, 2])
def test_nonterminal_needs_exactly_one_primary(products, count):
    _raw, proc, group = products.values()
    group.assocs = [proc.assocs[0].model_copy() for _ in range(count)]
    with pytest.raises(ValueError, match="exactly one primary"):
        group.get_primary(products.__getitem__)


def test_missing_lookup_and_target_do_not_guess_from_uids(products):
    raw, proc, group = products.values()
    with pytest.raises(ValueError, match="lookup"):
        proc.get_primary()
    with pytest.raises(KeyError, match=raw.uid):
        proc.get_primary({}.__getitem__)
    with pytest.raises(ValueError, match="different product"):
        proc.get_primary(lambda uid: group)


def test_cycle_detected_when_lookup_returns_fresh_objects(products):
    _raw, proc, group = products.values()
    proc.assocs[0].uid = group.uid
    with pytest.raises(ValueError, match="cycle"):
        group.get_primary(lambda uid: products[uid].model_copy(deep=True))


@pytest.mark.parametrize(
    ("value", "expected"),
    [("TCS-00145609-000-0001", "tcs-145609-0-1"), ("ICS-00001", "ics-1")],
)
def test_normalized_primary_uid_placement(value, expected, tmp_path):
    assert normalize_primary_uid(value) == expected
    store = DataProdStore(tmp_path)
    assert store.get_product_path(value, "dp_raw_obs_proc") == (
        store.get_product_path(expected, "dp_raw_obs_proc")
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "value", ["145609", "1-0-1", "tcs-1-2", "tcs-*", "tcs-1-0-1_proc"]
)
def test_primary_uid_rejects_non_entity_identifiers(value):
    with pytest.raises(ValueError, match="explicit master"):
        normalize_primary_uid(value)


@pytest.mark.parametrize(
    ("make_uid", "numbers"),
    [(make_obs_product_uid, (1,)), (make_raw_obs_product_uid, (1, 0, 1))],
)
def test_product_uids_still_require_master(make_uid, numbers):
    with pytest.raises(ValueError, match="requires a master"):
        make_uid(None, *numbers)
    assert make_uid("TCS", *numbers) == "tcs-" + "-".join(map(str, numbers))
