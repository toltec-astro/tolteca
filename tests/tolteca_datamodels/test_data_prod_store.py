"""Shared placement, identity and atomic index publication."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tolteca_datamodels.data_prod import (
    DataProdIndex,
    DataProdIndexAssoc,
    DataProdIndexSource,
)
from tolteca_datamodels.data_prod_store import DataProdStore


def test_source_timestamp_loads_from_dump_without_coercing_other_values():
    source = DataProdIndexSource(
        source_uri="index.yaml",
        location="site",
        size=1,
        last_verified_at=datetime(2025, 10, 31, tzinfo=UTC),
    )
    data = source.model_dump(mode="json")
    assert isinstance(data["last_verified_at"], str)
    assert DataProdIndexSource.model_validate(data) == source
    with pytest.raises(ValidationError, match="size"):
        DataProdIndexSource.model_validate(data | {"size": "1"})


def test_layout_for_logical_acquisition_and_grouped_products(tmp_path):
    store = DataProdStore(tmp_path, namespace="ql")
    assert store.get_index_path("tcs-145609", "dp_obs", directory=False) == (
        tmp_path / "ql/145609/dp_obs.yaml"
    )
    assert store.get_index_path("tcs-145609-0-1", "dp_raw_obs", directory=False) == (
        tmp_path / "ql/145609-0-1/dp_raw_obs.yaml"
    )
    assert store.get_index_path("ics-1", "dp_drivefit") == (
        tmp_path / "ql/ics-1/dp_drivefit/index.yaml"
    )
    assert store.get_index_path(
        "tcs-145609-0-1", "dp_raw_obs_proc", run_name="0000-check-a"
    ) == (tmp_path / "ql/145609-0-1/dp_raw_obs_proc/0000-check-a/index.yaml")
    assert list(tmp_path.iterdir()) == []


def test_index_publication_keeps_canonical_ids_and_separates_locations(tmp_path):
    index = DataProdIndex(
        uid="tcs-145609-0-1",
        meta={
            "name": "raw",
            "data_prod_type": "dp_raw_obs",
            "master": "tcs",
            "obsnum": 145609,
            "subobsnum": 0,
            "scannum": 1,
        },
    )
    store = DataProdStore(tmp_path)
    path = store.write_index(
        index, primary_uid=index.get_primary().uid, directory=False
    )
    assert DataProdIndex.read(path) == index
    with pytest.raises(FileExistsError):
        store.write_index(index, primary_uid=index.get_primary().uid, directory=False)
    updated = index.model_copy(update={"meta": {**index.meta, "name": "updated"}})
    assert (
        store.write_index(
            updated,
            primary_uid=updated.get_primary().uid,
            directory=False,
            overwrite=True,
        )
        == path
    )
    assert DataProdIndex.read(path).meta["name"] == "updated"
    with pytest.raises(ValueError, match="Extra inputs"):
        DataProdIndexAssoc(
            uid=index.uid,
            data_prod_assoc_type="dpa_raw_obs_proc_raw_obs",
            index="raw.yaml",
        )
