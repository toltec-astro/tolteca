"""Portable record identity, extension boundaries and document references."""

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import PurePosixPath
from uuid import UUID

import pytest
from pydantic import ValidationError, create_model

from tolteca_datamodels.core.data_prod import (
    Actor,
    CounterRunNameGenerator,
    DataProdAssoc,
    DataProdAudit,
    DataProdBase,
    DataProdIndexBase,
    DataProdItem,
    DataProdNamespace,
    DataProdNamespaceIndex,
    DataProdRef,
    DataProdSource,
    DataProdStoreBase,
    DataProdTag,
    Location,
    StoreRun,
    WorkflowRun,
)


class ProductRecord(DataProdBase[dict[str, str]]):
    """Product with extension-defined metadata."""


class ProductIndex(DataProdIndexBase[ProductRecord]):
    """Portable document for the extension product."""


@pytest.mark.parametrize(
    ("record_cls", "values"),
    [
        (DataProdRef, {}),
        (DataProdAssoc, {"data_prod_assoc_type": "derived_from"}),
        (ProductRecord, {"meta": {}}),
    ],
    ids=["reference", "association", "product"],
)
def test_product_identity_requires_nonempty_uid(record_cls, values):
    with pytest.raises(ValidationError) as exc_info:
        record_cls.model_validate({"uid": "", **values})

    assert [(error["loc"], error["type"]) for error in exc_info.value.errors()] == [
        (("uid",), "string_too_short")
    ]


@pytest.mark.parametrize(
    ("uid", "error_type"), [("", "string_too_short"), (b"measurement:1", "string_type")]
)
def test_tag_validates_supplied_and_default_selection(uid, error_type):
    tag_with_default = create_model(
        "DefaultSelectionTag",
        __base__=DataProdTag,
        current=(DataProdRef | None, {"uid": uid}),
    )

    for model, values in (
        (DataProdTag, {"current": {"uid": uid}}),
        (tag_with_default, {}),
    ):
        with pytest.raises(ValidationError) as exc_info:
            model.model_validate(values)
        assert [(error["loc"], error["type"]) for error in exc_info.value.errors()] == [
            (("current", "uid"), error_type)
        ]


def test_tag_selection_remains_editable():
    tag = DataProdTag()
    assert tag.current is None
    assert tag.audit is None

    reference = DataProdRef(uid="measurement:1")
    tag.current = reference
    assert tag.current == reference
    tag.current = None
    assert tag.current is None


def test_core_imports_do_not_load_concrete_products_or_instruments():
    script = """
import importlib
import pkgutil
import sys

from tolteca_datamodels.core import data_prod

for module in pkgutil.walk_packages(data_prod.__path__, data_prod.__name__ + "."):
    importlib.import_module(module.name)

assert not any(
    name == prefix or name.startswith(prefix + ".")
    for prefix in (
        "tolteca_datamodels.data_prod", "tolteca_datamodels.lmt_base",
        "tolteca_datamodels.toltec", "tolteca_datamodels.lmt", "tolteca_db",
    )
    for name in sys.modules
)
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
    )


def test_extension_association_needs_no_database_registration():
    association = DataProdAssoc(
        uid="measurement:reference",
        data_prod_assoc_type="measurement_reference",
        context={"quantity": "temperature"},
    )
    index = ProductIndex(
        product=ProductRecord(uid="measurement:1", meta={}, assocs=[association])
    )

    assert index.product.assocs == [association]
    assert index.products == []


def test_index_rejects_undefined_item_location():
    item = DataProdItem(
        source=DataProdSource(path=PurePosixPath("values.txt"), location="missing")
    )

    with pytest.raises(ValidationError, match="Undefined storage location 'missing'"):
        ProductIndex(
            product=ProductRecord(uid="measurement:1", meta={}, data_items=[item])
        )


@pytest.mark.parametrize(
    ("location", "path", "duplicate"),
    [
        ("first", "values.txt", True),
        ("second", "values.txt", False),
        ("first", "other.txt", False),
    ],
    ids=["same-reference", "different-location", "different-path"],
)
def test_item_source_identity_uses_location_and_path(location, path, duplicate):
    items = [
        DataProdItem(source=DataProdSource(path=PurePosixPath(path), location=location))
        for location, path in [("first", "values.txt"), (location, path)]
    ]
    product = ProductRecord(uid="measurement:1", meta={}, data_items=items)
    locations = {"first": Location(root="first"), "second": Location(root="second")}

    if duplicate:
        with pytest.raises(ValidationError, match="Source is listed more than once"):
            ProductIndex(product=product, locations=locations)
    else:
        index = ProductIndex(product=product, locations=locations)
        assert index.product.data_items == items


def test_different_products_can_reference_the_same_source():
    item = DataProdItem(
        source=DataProdSource(path=PurePosixPath("values.txt"), location="payload")
    )
    locations = {"payload": Location(root=".")}
    child = ProductIndex(
        product=ProductRecord(uid="child", meta={}, data_items=[item]),
        locations=locations,
    )
    parent = ProductIndex(
        product=ProductRecord(uid="parent", meta={}, data_items=[item]),
        locations=locations,
        products=[child],
    )

    assert (
        parent.product.data_items[0].source
        == parent.products[0].product.data_items[0].source
    )


@pytest.fixture
def namespace():
    timestamp = datetime(2026, 11, 9, tzinfo=UTC)
    return DataProdNamespace(
        uid=UUID(int=1),
        path=PurePosixPath("measurements"),
        audit=DataProdAudit(created_at=timestamp, updated_at=timestamp),
    )


@pytest.fixture
def namespace_index(namespace):
    timestamp = datetime(2026, 11, 9, tzinfo=UTC)
    actor = Actor(uid=UUID(int=2), name="test", created_at=timestamp)
    run = WorkflowRun(uid=UUID(int=3), actor=actor, time_created=timestamp)
    return DataProdNamespaceIndex(
        namespace=namespace,
        actors={actor.uid: actor},
        runs={run.uid: run},
        store_runs={run.uid: StoreRun(run_uid=run.uid, name="measurement")},
    )


@pytest.mark.parametrize("field", ["actors", "runs", "store_runs"])
def test_namespace_mapping_keys_must_match_record_identity(namespace_index, field):
    values = namespace_index.model_dump(round_trip=True)
    records = values[field]
    records[UUID(int=99)] = records.pop(next(iter(records)))

    with pytest.raises(ValidationError, match="mapping key") as exc_info:
        DataProdNamespaceIndex.model_validate(values)

    assert [error["loc"] for error in exc_info.value.errors()] == [(field,)]


def test_product_actor_keys_must_match_record_identity(namespace_index):
    index = ProductIndex(
        product=ProductRecord(uid="measurement:1", meta={}),
        actors=namespace_index.actors,
    )
    assert index.actors == namespace_index.actors
    values = index.model_dump(round_trip=True)
    actor = values["actors"].pop(next(iter(values["actors"])))
    values["actors"][UUID(int=99)] = actor

    with pytest.raises(ValidationError, match="Actor mapping key") as exc_info:
        ProductIndex.model_validate(values)

    assert [error["loc"] for error in exc_info.value.errors()] == [("actors",)]


@pytest.mark.parametrize("other_execution", [False, True], ids=["empty", "different"])
def test_namespace_store_run_requires_its_execution(namespace_index, other_execution):
    values = namespace_index.model_dump(round_trip=True)
    run = values["runs"].pop(next(iter(values["runs"])))
    if other_execution:
        run["uid"] = UUID(int=99)
        values["runs"][run["uid"]] = run

    with pytest.raises(ValidationError, match="must reference an included execution"):
        DataProdNamespaceIndex.model_validate(values)


@pytest.mark.parametrize("duplicate", ["name", "counter"])
def test_namespace_rejects_conflicting_reservations(namespace_index, duplicate):
    values = namespace_index.model_dump(round_trip=True)
    run_uid, run = next(iter(values["runs"].items()))
    other_uid = UUID(int=4)
    values["runs"][other_uid] = {**run, "uid": other_uid}
    first = StoreRun(
        run_uid=run_uid,
        name="0003-first",
        generator=CounterRunNameGenerator(value=3, key="first"),
    )
    other = StoreRun(
        run_uid=other_uid,
        name=first.name if duplicate == "name" else "0003-other",
        generator=None
        if duplicate == "name"
        else CounterRunNameGenerator(value=3, key="other"),
    )
    values["store_runs"] = {
        run_uid: first.model_dump(round_trip=True),
        other_uid: other.model_dump(round_trip=True),
    }

    with pytest.raises(ValidationError, match="reserved by multiple executions"):
        DataProdNamespaceIndex.model_validate(values)


def test_namespace_reservations_use_declared_counter_values(namespace_index, tmp_path):
    original = next(iter(namespace_index.runs.values()))
    runs = [
        original,
        *(original.model_copy(update={"uid": UUID(int=i)}) for i in (4, 5)),
    ]
    records = [
        StoreRun(
            run_uid=runs[0].uid,
            name="0003-first",
            generator=CounterRunNameGenerator(value=3, key="first"),
        ),
        StoreRun(run_uid=runs[1].uid, name="0003-literal"),
        StoreRun(run_uid=runs[2].uid, name="another-literal"),
    ]
    index = DataProdNamespaceIndex(
        namespace=namespace_index.namespace.model_copy(update={"counter_next": 4}),
        actors=namespace_index.actors,
        runs={run.uid: run for run in runs},
        store_runs={record.run_uid: record for record in records},
    )
    store = DataProdStoreBase(tmp_path, namespace=index.namespace.path)
    store.write_namespace_index(index)

    assert store.read_namespace_index() == index


def test_namespace_records_survive_store_export(namespace_index, tmp_path):
    run = next(iter(namespace_index.runs.values()))
    unreserved = run.model_copy(update={"uid": UUID(int=4)})
    namespace_index.runs[unreserved.uid] = unreserved
    restored = DataProdNamespaceIndex.model_validate_json(
        namespace_index.model_dump_json(round_trip=True)
    )
    assert restored == namespace_index

    store = DataProdStoreBase(tmp_path, namespace=restored.namespace.path)
    store.write_namespace_index(restored)

    assert store.read_namespace_index() == namespace_index


@pytest.mark.parametrize(
    "namespace_uid",
    [UUID(int=1), UUID(int=2), None],
    ids=["matching", "different", "global-product"],
)
def test_included_namespace_must_match_product(namespace, namespace_uid):
    product = ProductRecord(uid="measurement:1", namespace_uid=namespace_uid, meta={})

    if namespace_uid == namespace.uid:
        index = ProductIndex(product=product, namespace=namespace)
        assert index.namespace is not None
        assert index.product.namespace_uid == index.namespace.uid
    else:
        with pytest.raises(ValidationError, match="Product namespace differs"):
            ProductIndex(product=product, namespace=namespace)


def test_namespace_record_is_optional_for_namespaced_product():
    index = ProductIndex(
        product=ProductRecord(uid="measurement:1", namespace_uid=UUID(int=1), meta={})
    )

    assert index.product.namespace_uid == UUID(int=1)
    assert index.namespace is None


@pytest.mark.parametrize(
    "run_uid",
    [UUID(int=1), UUID(int=2), None],
    ids=["matching", "different", "missing-execution"],
)
def test_store_run_must_reference_included_execution(run_uid):
    timestamp = datetime(2026, 11, 9, tzinfo=UTC)
    run = (
        WorkflowRun(
            uid=run_uid,
            actor=Actor(uid=UUID(int=3), name="test", created_at=timestamp),
            time_created=timestamp,
        )
        if run_uid is not None
        else None
    )
    store_run = StoreRun(run_uid=UUID(int=1), name="measurement")
    product = ProductRecord(uid="measurement:1", meta={})

    if run_uid == store_run.run_uid:
        index = ProductIndex(product=product, run=run, store_run=store_run)
        assert index.run is not None
        assert index.store_run is not None
        assert index.store_run.run_uid == index.run.uid
    else:
        with pytest.raises(
            ValidationError, match="Store run must reference the included execution"
        ):
            ProductIndex(product=product, run=run, store_run=store_run)
