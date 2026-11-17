"""Shared placement, identity and atomic index publication."""

from copy import replace

import pytest
import yaml
from pydantic import ValidationError

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdAssoc,
    DataProdIndex,
    DataProdStore,
)
from tolteca_datamodels.data_prod.metadata import ScienceMeta


def test_resolve_compact_header_requires_version_group(tmp_path):
    index = DataProdIndex(
        product=DataProd(
            uid="analysis",
            meta=ScienceMeta(name="Analysis", data_prod_type="dp_science"),
        )
    )
    data = index.model_dump(mode="json", round_trip=True)
    data["document"] = "version_group_header"
    data["version_count"] = 0
    path = tmp_path / "index.yaml"
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValidationError, match="Expected a version group index"):
        DataProdStore.resolve_index(path)


def test_layout_for_logical_acquisition_and_grouped_products(tmp_path):
    store = DataProdStore(tmp_path, namespace="ql")
    assert store.get_index_path("tcs-145609", "dp_obs") == (
        tmp_path / "ql/145609/dp_obs.yaml"
    )
    assert store.get_index_path("tcs-145609-0-1", "dp_raw_obs") == (
        tmp_path / "ql/145609-0-1/dp_raw_obs.yaml"
    )
    assert store.get_index_path("tcs-145605-0-0", "dp_cal_group") == (
        tmp_path / "ql/145605-0-0/dp_cal_group.yaml"
    )
    assert store.get_index_path("ics-1", "dp_drivefit") == (
        tmp_path / "ql/ics-1/dp_drivefit/index.yaml"
    )
    assert store.get_index_path(
        "tcs-145609-0-1", "dp_raw_obs_proc", run_name="0000-check-a"
    ) == (tmp_path / "ql/145609-0-1/dp_raw_obs_proc/0000-check-a/index.yaml")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [("TCS-00145609-000-0001", "tcs-145609-0-1"), ("ICS-00001", "ics-1")],
)
def test_normalized_primary_uid_placement(value, expected, tmp_path):
    store = DataProdStore(tmp_path)
    assert store.get_product_path(value, "dp_raw_obs_proc") == (
        store.get_product_path(expected, "dp_raw_obs_proc")
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["dp_obs", "dp_raw_obs", "dp_cal_group"])
def test_intrinsic_product_cannot_have_a_run_index(tmp_path, kind):
    store = DataProdStore(tmp_path)
    with pytest.raises(ValueError, match="Logical descriptors do not have run indexes"):
        store.get_index_path("tcs-145609-0-1", kind, run_name="0000-check-a")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["dp_obs", "dp_raw_obs", "dp_cal_group"])
def test_index_publication_keeps_canonical_ids_and_separates_locations(tmp_path, kind):
    identity = {"master": "tcs", "obsnum": 145609}
    primary_uid = "tcs-145609"
    if kind != "dp_obs":
        identity.update(subobsnum=0, scannum=1)
        primary_uid += "-0-1"
    uid = f"dp_cal_group_{primary_uid}" if kind == "dp_cal_group" else primary_uid
    index = DataProdIndex(
        product=DataProd(
            uid=uid,
            meta={
                "name": "observation descriptor",
                "data_prod_type": kind,
                **identity,
            },
        )
    )
    store = DataProdStore(tmp_path)
    path = store.write_index(index, primary_uid=primary_uid)
    assert (
        path == tmp_path / "scratch" / primary_uid.removeprefix("tcs-") / f"{kind}.yaml"
    )
    assert not path.with_suffix("").exists()
    saved = DataProdIndex.read(path)
    assert saved.product.uid == index.product.uid
    assert saved.product.meta == index.product.meta
    assert saved.get_source_uri(saved.product.data_items[0].source) == path.as_uri()
    assert not index.product.data_items
    with pytest.raises(FileExistsError):
        store.write_index(index, primary_uid=primary_uid)
    updated = index.model_copy(
        update={
            "product": index.product.model_copy(
                update={"meta": replace(index.product.meta, name="updated")}
            )
        }
    )
    assert (
        store.write_index(
            updated,
            primary_uid=primary_uid,
            overwrite=True,
        )
        == path
    )
    assert DataProdIndex.read(path).product.meta.name == "updated"
    with pytest.raises(ValueError, match="Extra inputs"):
        DataProdAssoc(
            uid=index.product.uid,
            data_prod_assoc_type="dpa_raw_obs_proc_raw_obs",
            index="raw.yaml",
        )
