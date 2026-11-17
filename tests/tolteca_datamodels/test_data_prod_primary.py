"""Primary traversal uses marked associations and complete product references."""

from uuid import uuid4

import pytest

from tolteca_datamodels.data_prod import DataProd, DataProdAssoc, DataProdIndex
from tolteca_datamodels.data_prod.metadata import (
    ObsMeta,
    RawObsMeta,
    RawObsProcMeta,
    ScienceMeta,
    VersionGroupMeta,
)


@pytest.fixture
def products():
    raw = DataProd(
        uid="tcs-145609-0-1",
        meta=RawObsMeta(
            name="raw",
            data_prod_type="dp_raw_obs",
            master="tcs",
            obsnum=145609,
            subobsnum=0,
            scannum=1,
        ),
        assocs=[
            DataProdAssoc(
                uid="tcs-145605-0-1",
                data_prod_assoc_type="dpa_raw_obs_cal_obs",
            )
        ],
    )
    proc = DataProd(
        uid=f"{raw.uid}_proc",
        namespace_uid=uuid4(),
        meta=RawObsProcMeta(
            name="processed",
            data_prod_type="dp_raw_obs_proc",
            master="tcs",
            obsnum=145609,
            subobsnum=0,
            scannum=1,
        ),
        assocs=[
            DataProdAssoc(
                uid=raw.uid,
                data_prod_assoc_type="dpa_raw_obs_proc_raw_obs",
                is_primary=True,
            )
        ],
    )
    # Portable producers can supply their own relationship labels.
    result = DataProd(
        uid="analysis",
        namespace_uid=uuid4(),
        meta=ScienceMeta(name="analysis", data_prod_type="dp_science"),
        assocs=[
            DataProdAssoc(uid="unloaded-input", data_prod_assoc_type="derived_from"),
            DataProdAssoc(
                uid=proc.uid,
                namespace_uid=proc.namespace_uid,
                data_prod_assoc_type="derived_from",
                is_primary=True,
            ),
        ],
    )
    return {product.reference: product for product in (raw, proc, result)}


def test_primary_chain_follows_only_marked_references(products):
    raw, proc, result = products.values()
    calls = []

    def lookup(reference):
        calls.append(reference)
        return products[reference]

    assert result.get_primary(lookup) is raw
    assert calls == [proc.reference, raw.reference]


def test_primary_references_survive_yaml(products, tmp_path):
    raw, _proc, result = products.values()
    path = DataProdIndex(product=result).write(tmp_path / "result.yaml")

    restored = DataProdIndex.read(path).product

    assert restored.reference == result.reference
    assert restored.assocs == result.assocs
    assert restored.get_primary(products.__getitem__) is raw


def test_acquisition_is_terminal_with_calibration_reference(products):
    raw, _proc, _result = products.values()
    assert raw.get_primary() is raw


def test_observation_is_terminal_with_membership(products):
    raw, _proc, _result = products.values()
    obs = DataProd(
        uid="tcs-145609",
        meta=ObsMeta(
            name="observation", data_prod_type="dp_obs", master="tcs", obsnum=145609
        ),
        assocs=[DataProdAssoc(uid=raw.uid, data_prod_assoc_type="dpa_obs_raw_obs")],
    )
    assert obs.get_primary() is obs


@pytest.mark.parametrize("count", [0, 2])
def test_nonterminal_needs_exactly_one_primary(products, count):
    _raw, _proc, result = products.values()
    result.assocs = [result.assocs[1].model_copy() for _ in range(count)]
    with pytest.raises(ValueError, match="exactly one primary"):
        result.get_primary(products.__getitem__)


def test_nonterminal_requires_lookup(products):
    _raw, proc, _result = products.values()
    with pytest.raises(ValueError, match="lookup"):
        proc.get_primary()


def test_missing_primary_target_is_reported(products):
    raw, proc, _result = products.values()
    with pytest.raises(KeyError) as exc:
        proc.get_primary({}.__getitem__)
    assert exc.value.args == (raw.reference,)


@pytest.mark.parametrize("field", ["uid", "namespace_uid"])
def test_primary_lookup_requires_complete_identity(products, field):
    _raw, proc, result = products.values()
    wrong = proc.model_copy(
        update={field: "another-product" if field == "uid" else uuid4()}
    )
    with pytest.raises(ValueError, match="different product"):
        result.get_primary(lambda reference: wrong)


def test_equal_uids_in_different_namespaces_are_not_a_cycle(products):
    raw, proc, result = products.values()
    result.uid = proc.uid
    lookup = {product.reference: product for product in products.values()}

    assert result.get_primary(lookup.__getitem__) is raw


def test_cycle_detected_when_lookup_returns_fresh_objects(products):
    _raw, proc, result = products.values()
    proc.assocs[0].uid = result.uid
    proc.assocs[0].namespace_uid = result.namespace_uid
    with pytest.raises(ValueError, match="cycle"):
        result.get_primary(lambda reference: products[reference].model_copy(deep=True))


def test_empty_version_group_resolves_its_primary(products):
    raw, _proc, _result = products.values()
    group = DataProdIndex(
        product=DataProd(
            uid=f"{raw.uid}_proc",
            meta=VersionGroupMeta(
                name="processed versions",
                data_prod_type="dp_version_group",
                content_data_prod_type="dp_raw_obs_proc",
            ),
            assocs=[
                DataProdAssoc(
                    uid=raw.uid,
                    data_prod_assoc_type="dpa_version_group_raw_obs",
                    is_primary=True,
                )
            ],
        )
    )
    assert group.product.get_primary(products.__getitem__) is raw
