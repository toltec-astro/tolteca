"""Persistent run allocation, concurrent publication and tag selection."""

import multiprocessing
import shutil
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import TypeAdapter, ValidationError
from tollan.config import RuntimeInfo
from tollan.pipeline import Artifact, ArtifactFile, ArtifactSpec

from tolteca_datamodels.data_prod import (
    CounterRunNameGenerator,
    DataProdIndex,
    DataProdStore,
    DataProdTagBusyError,
    RunName,
    resolve_index,
    write_artifacts,
)
from tolteca_datamodels.data_prod import WorkflowRun as ExecutionRun

UID = "tcs-145609-0-1"
TYPE = "dp_raw_obs_proc"


def _index(run=None):
    return DataProdIndex.model_validate(
        {
            "locations": {}
            if run is None
            else {"self": {"root": ".", "location_type": "local"}},
            "product": {
                "uid": f"{UID}_proc",
                "meta": {"name": f"{UID}_proc", "data_prod_type": TYPE},
                "assocs": [
                    {
                        "uid": UID,
                        "data_prod_assoc_type": "dpa_raw_obs_proc_raw_obs",
                        "is_primary": True,
                    }
                ],
                "data_items": []
                if run is None
                else [
                    {
                        "meta": {
                            "type": "artifact",
                            "format": fmt,
                            "run_uid": str(run.uid),
                        },
                        "source": {
                            "path": name,
                            "location": "self",
                            "availability_state": "available",
                        },
                    }
                    for name, fmt in (
                        ("payload.txt", "text"),
                        ("index.yaml", "data_prod_index"),
                    )
                ],
            },
        }
    )


@contextmanager
def _stage(store, *, run=None, tag="main"):
    with (
        ExitStack() as _run_records,
        ExecutionRun.create(uid=(run or {}).get("uid")) as context,
        store.materialize(
            _index(context).model_copy(update={"run": context}),
            tag=tag,
            run_name=(run or {}).get("name"),
        ) as output,
    ):
        _run_records.callback(store.write_run, context)
        yield output


def _write(root, key, *, tag="main", value=None):
    store = DataProdStore(root)
    with _stage(
        store, run={"name": CounterRunNameGenerator(value=value, key=key)}, tag=tag
    ) as pending:
        (pending.base_path / "payload.txt").write_text(key)
    return pending.base_path


def _parallel_writer(root, key, barrier, output):
    with _stage(
        DataProdStore(root), tag=key, run={"name": CounterRunNameGenerator(key=key)}
    ) as pending:
        (pending.base_path / "payload.txt").write_text(key)
        barrier.wait(timeout=30)
    output.put(str(pending.base_path))


def _hold_tag(root, ready):
    with _stage(DataProdStore(root), run={"name": CounterRunNameGenerator(key="held")}):
        ready.set()
        multiprocessing.Event().wait(60)


def test_counter_allocations_tags_and_portable_run(tmp_path):
    first = _write(tmp_path, "batch")
    second = _write(tmp_path, "experiment", tag="test", value=7)
    lower = _write(tmp_path, "unused", value=3)
    last = _write(tmp_path, "batch", tag=None)
    assert [p.name for p in (first, second, lower, last)] == [
        "0000-batch",
        "0007-experiment",
        "0003-unused",
        "0008-batch",
    ]
    product = first.parent
    inventory = DataProdIndex.read(product)
    assert inventory.product.meta.content_data_prod_type == TYPE
    assert DataProdStore(tmp_path).read_namespace().namespace.counter_next == 9
    assert (
        inventory.product.tags["main"].current.uid
        == DataProdIndex.read(lower).product.uid
    )
    assert resolve_index(product) == lower / "index.yaml"
    assert resolve_index(product, tag="test") == second / "index.yaml"
    with pytest.raises(FileExistsError):
        _write(tmp_path, "different-key", value=7)
    assert DataProdIndex.read(product) == inventory
    moved = tmp_path / "detached"
    shutil.copytree(first, moved)
    shutil.rmtree(first)
    index = DataProdIndex.read(moved)
    assert index.store_run.name == first.name
    descriptor = next(
        item
        for item in index.product.data_items
        if item.meta["format"] == "data_prod_index"
    )
    assert descriptor.meta["run_uid"] == str(index.run.uid)
    assert index.store_run.generator.value == 0
    assert index.product.audit.created_at >= index.run.time_created
    assert next(iter(index.locations.values())).root == "."
    assert index.resolve_location("self").url == moved.as_uri()
    assert (moved / "payload.txt").read_text() == "batch"


def test_explicit_and_uuid_ids_are_opaque_and_do_not_consume_counter(tmp_path):
    store = DataProdStore(tmp_path)
    with _stage(store, run={"name": "0000-batch"}) as pending:
        (pending.base_path / "payload.txt").write_text("literal")
    literal = pending.base_path
    with _stage(store) as pending:
        assert UUID(pending.store_run.name).version == 4
        (pending.base_path / "payload.txt").write_text("uuid")
    assert store.read_namespace().namespace.counter_next == 0
    # The automatic candidate collides with a literal ID; the allocator skips it.
    assert _write(tmp_path, "batch").name == "0001-batch"
    # No number was parsed from the literal ID, so counter value 0 is still unused.
    assert _write(tmp_path, "free", value=0).name == "0000-free"
    with pytest.raises(FileExistsError), _stage(store, run={"name": literal.name}):
        pytest.fail("A literal ID was reused")


def test_failed_run_retains_reservation_and_does_not_advance_tag(tmp_path):
    store = DataProdStore(tmp_path)
    with (  # noqa: PT012 -- Exercise work and context-manager failure together.
        pytest.raises(RuntimeError, match="writer failed"),
        _stage(
            store, run={"name": CounterRunNameGenerator(value=42, key="failed")}
        ) as pending,
    ):
        root = pending.base_path.parent.parent
        before = DataProdIndex.read(root)
        assert store.read_namespace().namespace.counter_next == 43
        assert before.product.tags["main"].current is None
        assert before.products == []
        with pytest.raises(KeyError):
            resolve_index(root)
        msg = "writer failed"
        raise RuntimeError(msg)
    assert DataProdIndex.read(root) == before
    with pytest.raises(FileExistsError):
        _write(tmp_path, "retry", value=42)
    assert _write(tmp_path, "next").name == "0043-next"
    assert (
        DataProdIndex.read(root).product.tags["main"].current.uid
        == DataProdIndex.read(root / "0043-next").product.uid
    )
    assert store.read_run(pending.run.uid).status == "failed"


def test_parallel_tags_allocate_distinct_values_without_lost_updates(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    barrier, output = ctx.Barrier(2), ctx.Queue()
    processes = [
        ctx.Process(target=_parallel_writer, args=(tmp_path, key, barrier, output))
        for key in ("one", "two")
    ]
    for process in processes:
        process.start()
    try:
        paths = [Path(output.get(timeout=45)) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert {
            DataProdIndex.read(path).store_run.generator.value for path in paths
        } == {
            0,
            1,
        }
        inventory = DataProdIndex.read(paths[0].parent)
        assert DataProdStore(tmp_path).read_namespace().namespace.counter_next == 2
        assert set(inventory.product.tags) == {"one", "two"}
        assert len(inventory.product.data_items) == 1
        assert len(inventory.products) == 2
        for path in paths:
            assert (
                DataProdStore(tmp_path)
                .read_run(DataProdIndex.read(path).run.uid)
                .time_completed
                is not None
            )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        output.close()


def test_tag_contention_and_reservation_survive_process_death(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(target=_hold_tag, args=(tmp_path, ready))
    process.start()
    try:
        assert ready.wait(30)
        with pytest.raises(DataProdTagBusyError, match="main.*busy"):
            _write(tmp_path, "blocked")
        other = _write(tmp_path, "allowed", tag="test")
        assert other.name == "0002-allowed"
    finally:
        process.terminate()
        process.join(timeout=10)
    assert _write(tmp_path, "recovered").name == "0003-recovered"
    with pytest.raises(FileExistsError):
        _write(tmp_path, "reuse", value=0)


def test_failed_install_preserves_current_and_leaves_run_reserved(
    tmp_path, monkeypatch
):
    first = _write(tmp_path, "first")
    store = DataProdStore(tmp_path)
    root = first.parent / "index.yaml"
    replace = Path.replace
    with (  # noqa: PT012 -- Exercise work and context-manager failure together.
        pytest.raises(OSError, match="interrupted"),
        monkeypatch.context() as patch,
        _stage(store, run={"name": CounterRunNameGenerator(key="orphan")}) as pending,
    ):
        (pending.base_path / "payload.txt").write_text("orphan")
        reserved = root.read_bytes()

        def fail(source, destination):
            if destination == root:
                msg = "interrupted index replacement"
                raise OSError(msg)
            return replace(source, destination)

        patch.setattr(Path, "replace", fail)
    assert root.read_bytes() == reserved
    assert resolve_index(root) == first / "index.yaml"
    assert (first.parent / "0001-orphan/index.yaml").is_file()
    assert store.read_run(pending.run.uid).status == "failed"
    with pytest.raises(KeyError):
        store.set_tag(DataProdIndex.read(first.parent / "0001-orphan"))
    assert _write(tmp_path, "next").name == "0002-next"


@pytest.mark.parametrize(
    "spec",
    [
        {"generator": "counter", "value": -1},
        {"generator": "counter", "key": "../bad"},
        {"generator": "counter", "key": ""},
        {"generator": "counter", "num": 3},
        {"generator": "counter", "format": "{key}"},
        {"generator": "counter", "format": "{value.foo}"},
        {"generator": "counter", "format": "{value}/unsafe"},
        {"generator": "missing"},
        "../outside",
        "index.yaml",
        ".tag-main.lock",
    ],
)
def test_run_request_rejects_invalid_values(spec):
    with pytest.raises(ValidationError):
        TypeAdapter(RunName | CounterRunNameGenerator).validate_python(spec)


def test_publication_preserves_run_names_and_qualified_tags(tmp_path):
    path = _write(tmp_path, "night 1", tag="origin:main")

    assert path.name == "0000-night 1"
    assert resolve_index(path.parent, tag="origin:main") == (path / "index.yaml")


def test_promotion_changes_selection_and_preserves_immutable_run(tmp_path):
    original = _write(tmp_path, "original")
    candidate = _write(tmp_path, "candidate", tag=None)
    contents = (candidate / "index.yaml").read_bytes()
    store = DataProdStore(tmp_path)
    with pytest.raises(DataProdTagBusyError), _stage(store):
        store.set_tag(DataProdIndex.read(candidate))
    assert resolve_index(original.parent) == original / "index.yaml"
    assert store.set_tag(DataProdIndex.read(candidate)) == candidate / "index.yaml"
    assert resolve_index(original.parent) == candidate / "index.yaml"
    assert (
        DataProdIndex.read(original.parent).product.tags["main"].current.uid
        == DataProdIndex.read(candidate).product.uid
    )
    assert (candidate / "index.yaml").read_bytes() == contents
    store.set_tag(DataProdIndex.read(original))
    assert (
        DataProdIndex.read(original.parent).product.tags["main"].current.uid
        == DataProdIndex.read(original).product.uid
    )


def test_caller_uid_is_independent_of_name_and_cannot_be_reserved_again(tmp_path):
    uid = UUID("03f1f28e-c179-471f-a684-0767207f2975")
    store = DataProdStore(tmp_path)
    with _stage(store, run={"uid": str(uid), "name": "nightly"}) as pending:
        (pending.base_path / "payload.txt").write_text("data")
    path = pending.base_path
    product = DataProdIndex.read(path)
    assert path.name == "nightly"
    assert product.run.uid == uid
    assert product.product.uid == f"{UID}_proc/{uid}"
    membership = next(
        a
        for a in DataProdIndex.read(path.parent).product.assocs
        if a.uid == product.product.uid
    )
    assert set(membership.runs) == {uid}
    assert store.read_namespace_index().store_runs[uid].name == "nightly"
    with (
        pytest.raises(ValueError, match="Conflicting execution"),
        _stage(store, run={"uid": uid, "name": "another-name"}),
    ):
        pytest.fail("One UID must not reserve a second invocation")
    with (
        pytest.raises(FileExistsError, match="Run name"),
        _stage(store, run={"name": "nightly"}),
    ):
        pytest.fail("A different UID must not reuse the name")
    assert (
        store.read_namespace().namespace.uid == DataProdIndex.read(path).namespace.uid
    )


def test_materialize_binds_a_copy_and_reuses_the_description(tmp_path):
    from tolteca_datamodels.data_prod import Location

    stores = [
        DataProdStore(tmp_path / name, namespace=name) for name in ("ql", "science")
    ]
    with ExecutionRun.create() as run:
        description = _index(run).model_copy(update={"run": run})
        description.locations["remote"] = Location(root="https://example.org/archive/")
        original = description.model_dump_json()
        outputs = []
        for store in stores:
            with store.materialize(description, run_name="test") as index:
                assert isinstance(index, DataProdIndex)
                assert index is not description
                assert index.base_path is not None
                assert index.run is not None
                assert index.namespace is not None
                assert index.base_path.is_dir()
                assert index.base_path.parent.name == ".staging"
                assert index.run.uid == run.uid
                assert index.namespace.uid == store.read_namespace().namespace.uid
                (index.base_path / "payload.txt").write_text("shared result")
                assert description.model_dump_json() == original
            assert index.base_path is not None
            assert index.base_path.name == "test"
            assert (index.base_path / "index.yaml").is_file()
            assert (
                DataProdIndex.read(index.base_path.parent)
                .products[0]
                .locations["remote"]
                .root
                == "https://example.org/archive/"
            )
            outputs.append(index)
        assert description.model_dump_json() == original
        assert description.base_path is None
        assert outputs[0].product.namespace_uid != outputs[1].product.namespace_uid
        assert outputs[0].product.uid == outputs[1].product.uid


def test_materialize_requires_identity_and_rejects_changing_it(tmp_path):
    store = DataProdStore(tmp_path)
    with pytest.raises(ValueError, match="index.run"), store.materialize(_index()):
        pytest.fail("Missing execution accepted")
    with ExecutionRun.create() as run:
        description = _index(run).model_copy(update={"run": run})
        with (
            pytest.raises(ValueError, match="identity changed"),
            store.materialize(description) as index,
        ):
            index.product.uid = "different-product"
        assert index.base_path is None
        assert not list(store.namespace_path.glob("*/dp_raw_obs_proc/*/index.yaml"))


def test_materialized_index_retains_namespace_creator_attribution(tmp_path):
    store = DataProdStore(tmp_path)
    namespace = store.ensure_namespace().namespace
    actor = ExecutionRun.create().actor.model_copy(update={"uid": uuid4()})
    with ExecutionRun.create(actor=actor) as execution:
        assert execution.actor_uid != namespace.audit.created_by
        with store.materialize(
            _index().model_copy(update={"run": execution}), run_name="test"
        ) as index:
            pass
    assert index.base_path is not None
    restored = DataProdIndex.read(index.base_path / "index.yaml")
    assert restored.namespace is not None
    assert restored.namespace.audit.created_by == namespace.audit.created_by
    assert namespace.audit.created_by in restored.actors
    assert restored.namespace.audit.updated_by in restored.actors
    assert execution.actor_uid in restored.actors


@pytest.mark.parametrize("recorded_promoter", [False, True])
def test_promotion_reuses_environment_actor_and_preserves_producer(
    tmp_path, recorded_promoter
):
    store = DataProdStore(tmp_path)
    promoter = ExecutionRun.create().actor
    if recorded_promoter:
        promoter = promoter.model_copy(
            update={"uid": uuid4(), "created_at": datetime(2024, 1, 1, tzinfo=UTC)}
        )
        store.write_run(ExecutionRun.create(actor=promoter))
    with (
        ExecutionRun.create(
            runtime_info=RuntimeInfo(username="product-producer")
        ) as producer,
        store.materialize(
            _index().model_copy(update={"run": producer}), run_name="candidate"
        ) as index,
    ):
        pass
    assert index.base_path is not None
    original = (index.base_path / "index.yaml").read_bytes()
    before = store.read_namespace()
    group_before = DataProdIndex.read(index.base_path.parent)

    store.set_tag(index)
    store.set_tag(index)

    group = DataProdIndex.read(index.base_path.parent)
    tag = group.product.tags["main"]
    assert tag.audit.created_by == producer.actor_uid
    assert tag.audit.created_at == group_before.product.tags["main"].audit.created_at
    assert tag.audit.updated_by == promoter.uid
    assert group.product.audit.updated_by == promoter.uid
    assert group.actors[promoter.uid] == promoter
    assert group.actors[producer.actor_uid] == producer.actor
    after = store.read_namespace()
    assert after.actors == {**before.actors, promoter.uid: promoter}
    assert after.namespace.audit == before.namespace.audit
    assert (index.base_path / "index.yaml").read_bytes() == original


def test_materialize_resolves_a_primary_chain_from_supporting_indexes(tmp_path):
    from tolteca_datamodels.data_prod import DataProd, DataProdAssoc
    from tolteca_datamodels.data_prod.metadata import ScienceMeta

    store = DataProdStore(tmp_path)
    source = DataProdIndex.read(_write(tmp_path, "processed"))
    with ExecutionRun.create() as run:
        description = DataProdIndex(
            run=run,
            product=DataProd(
                uid="selected-results",
                meta=ScienceMeta(name="Selected results", data_prod_type="dp_science"),
                assocs=[
                    DataProdAssoc(
                        uid=source.product.uid,
                        namespace_uid=source.product.namespace_uid,
                        data_prod_assoc_type="dpa_reduction_data_prod",
                        is_primary=True,
                    )
                ],
            ),
            products=[source],
        )
        with store.materialize(description) as index:
            assert index.base_path is not None
            assert index.base_path.parent.parent.parent.name == "145609-0-1"
        saved = DataProdIndex.read(index.base_path)
        assert saved.product.assocs[0].uid == source.product.uid
        assert saved.products == []
        assert description.products == [source]


def test_new_publication_and_tag_lookup_do_not_read_prior_versions(tmp_path):
    """The root stays bounded and the complete graph is expanded only on demand."""
    first = _write(tmp_path, "first")
    old = first / "index.yaml"
    saved = old.read_bytes()
    old.write_text("deliberately unreadable old payload index")
    second = _write(tmp_path, "second")
    assert resolve_index(second.parent) == second / "index.yaml"
    group = DataProdIndex.read(
        second.parent, run_uids=[DataProdIndex.read(second).run.uid]
    )
    assert len(group.products) == 1
    old.write_bytes(saved)
    previous = DataProdIndex.read(
        first.parent, run_uids=[DataProdIndex.read(first).run.uid]
    )
    assert len(previous.products) == 1 and previous.product.tags == {}
    assert group.product.tags["main"].current == group.products[0].product.reference
    assert len(DataProdIndex.read(second.parent).products) == 2
    from tollan.utils.yaml import yaml_load

    header = yaml_load(second.parent / "index.yaml")
    assert isinstance(header, dict)
    assert header["version_count"] == 2
    assert not header.get("products")


@pytest.mark.parametrize("committed", [False, True])
def test_finished_run_settles_interrupted_publication(tmp_path, monkeypatch, committed):
    """A vanished collector can leave a pending publication on either side of commit."""
    from tolteca_datamodels.core.data_prod.store import _StoreRunWriter

    store = DataProdStore(tmp_path)
    run = ExecutionRun.create()
    proposal = _index(run)
    proposal.run = run
    directory = store.get_product_path(UID, TYPE, uid=proposal.product.uid)
    with monkeypatch.context() as lost_writer:
        lost_writer.setattr(_StoreRunWriter, "_record", lambda *_: None)
        if committed:
            with store.materialize(proposal) as out:
                assert out.base_path is not None
                (out.base_path / "payload.txt").write_text("installed")
        else:
            msg = "interrupted"
            with (
                pytest.raises(RuntimeError, match="interrupted"),
                store.materialize(proposal),
            ):
                raise RuntimeError(msg)
    raw = store.read_run(run.uid)
    assert raw.status == "running"
    members = DataProdIndex.read(directory, run_uids=[run.uid]).products
    assert len(members) == (1 if committed else 0)

    def status():
        stored = store.read_namespace_index(run_uids=[run.uid]).store_runs[run.uid]
        return stored.publications[proposal.product.uid].status

    store.write_run(raw)
    assert status() == "pending"
    with raw:
        if not committed:
            raw.fail()
    store.write_run(raw)
    stored = store.read_namespace_index(run_uids=[run.uid]).store_runs[run.uid]
    assert stored.publications[proposal.product.uid].status == (
        "completed" if committed else "failed"
    )


class _Text(ArtifactSpec[str]):
    def write(self, data, *, path):
        if data == "fail":
            msg = "writer failed"
            raise RuntimeError(msg)
        (path / "notes").mkdir()
        target = path / "notes" / "report.txt"
        target.write_text(data)
        return [ArtifactFile(path=target, format="text")]


def test_written_artifacts_become_items_of_the_realization(tmp_path):
    store = DataProdStore(tmp_path)
    with (
        ExecutionRun.create() as run,
        store.materialize(_index().model_copy(update={"run": run})) as index,
    ):
        (item,) = write_artifacts(
            index, [Artifact(data="ok", spec=_Text())], meta={"producer": "test"}
        )
    saved = DataProdIndex.read(index.base_path)
    (item,) = (i for i in saved.product.data_items if i.source == item.source)
    assert item.meta == {
        "type": "artifact",
        "format": "text",
        "name": "notes/report.txt",
        "run_uid": str(run.uid),
        "producer": "test",
    }
    path = saved.resolve_location(item.source.location).joinpath(item.source.path)
    assert path.path.read_text() == "ok"
    # The store's own index item shares the producer's location.
    assert set(saved.locations) == {"self"}


def test_failed_artifact_writer_installs_nothing(tmp_path):
    first = _write(tmp_path, "first")
    store = DataProdStore(tmp_path)
    with (
        pytest.raises(RuntimeError, match="writer failed"),
        ExecutionRun.create() as run,
        store.materialize(_index().model_copy(update={"run": run})) as index,
    ):
        write_artifacts(index, [Artifact(data="fail", spec=_Text())])
    assert resolve_index(first.parent) == first / "index.yaml"
    assert len(DataProdIndex.read(first.parent).products) == 1
