"""Batch transport preserves records without resolving their dependencies."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from tolteca_datamodels.core.data_prod import DataProdStoreBase
from tolteca_datamodels.data_prod import (
    Actor,
    DataProd,
    DataProdBatch,
    DataProdIndex,
    DataProdInventory,
    DataProdInventoryIndex,
    DataProdItem,
    DataProdNamespaceIndex,
    DataProdRef,
    DataProdReview,
    DataProdSource,
    Location,
    WorkflowRun,
)
from tolteca_datamodels.data_prod.metadata import RawObsMeta


@pytest.fixture
def raw_index():
    return DataProdIndex(
        product=DataProd(
            uid="tcs-145609-0-1",
            meta=RawObsMeta(
                name="raw",
                data_prod_type="dp_raw_obs",
                master="tcs",
                obsnum=145609,
                subobsnum=0,
                scannum=1,
            ),
        )
    )


@pytest.fixture
def actor():
    return Actor(uid=uuid4(), name="reviewer", created_at=datetime.now(UTC))


def test_mixed_batch_preserves_order_records_and_submission_metadata(
    raw_index, tmp_path
):
    store = DataProdStoreBase(tmp_path, namespace="science")
    with WorkflowRun.create(context={"gain": 2}) as run:
        pass
    store.write_run(run)
    namespace = store.read_namespace_index(run_uids=[run.uid])
    inventory = DataProdInventoryIndex(
        inventory=DataProdInventory(uid=uuid4(), name="reported"),
        entries=[raw_index, namespace],
    )
    batch = DataProdBatch(
        meta={"purpose": "review"},
        items=[raw_index, namespace, inventory, raw_index.model_copy(deep=True)],
    )
    restored = DataProdBatch.model_validate_json(batch.model_dump_json())
    assert [type(item) for item in restored.items] == [
        DataProdIndex,
        DataProdNamespaceIndex,
        DataProdInventoryIndex,
        DataProdIndex,
    ]
    assert restored.items == batch.items
    contribution = restored.items[2]
    assert isinstance(contribution, DataProdInventoryIndex)
    assert contribution.inventory == inventory.inventory
    assert [type(entry) for entry in contribution.entries] == [
        DataProdIndex,
        DataProdNamespaceIndex,
    ]
    assert restored.meta == {"purpose": "review"}


def test_empty_batch_is_valid():
    batch = DataProdBatch()
    assert batch.items == []
    assert batch.reviews == []
    assert batch.actors == {}
    assert batch.meta == {}
    assert DataProdBatch.model_validate_json(batch.model_dump_json()) == batch


def test_review_only_batch_preserves_assessments_and_actors(actor):
    review = DataProdReview(
        product=DataProdRef(uid="tcs-145609-0-1"),
        actor_uid=actor.uid,
        namespace="quality",
        role="curator",
        previous_uid=uuid4(),
        flags={"needs_followup": False},
        note="Checked the recorded acquisition.",
    )
    batch = DataProdBatch(actors={actor.uid: actor}, reviews=[review])
    restored = DataProdBatch.model_validate_json(batch.model_dump_json())
    assert restored.items == []
    assert restored.reviews == [review]
    assert restored.actors == {actor.uid: actor}


def test_batch_rejects_malformed_item():
    with pytest.raises(ValidationError):
        DataProdBatch.model_validate({"items": [{"unrecognized": "record"}]})


def test_batch_rejects_actor_key_mismatch(actor):
    with pytest.raises(ValidationError, match="differs from actor UID"):
        DataProdBatch(actors={uuid4(): actor})


def test_wrapping_loaded_index_preserves_source_resolution(raw_index, tmp_path):
    source = DataProdSource.model_validate({"location": "local", "path": "data.nc"})
    raw_index.locations = {"local": Location(root=".")}
    raw_index.product.data_items = [DataProdItem(source=source)]
    loaded = DataProdIndex.read(raw_index.write(tmp_path / "index.yaml"))
    [wrapped] = DataProdBatch(items=[loaded]).items
    assert isinstance(wrapped, DataProdIndex)
    assert wrapped.get_source_uri(source) == (tmp_path / "data.nc").as_uri()
