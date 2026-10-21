"""Persistent run allocation, concurrent publication and tag selection."""

import multiprocessing
import shutil
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_datamodels.data_prod_run import CounterRunNameGenerator, RunRequest
from tolteca_datamodels.data_prod_store import DataProdStore, DataProdTagBusyError

UID = "tcs-145609-0-1"
TYPE = "dp_raw_obs_proc"


def _index(run=None):
    return DataProdIndex.model_validate(
        {
            "uid": f"{UID}_proc",
            "meta": {"name": f"{UID}_proc", "data_prod_type": TYPE},
            "data_items": []
            if run is None
            else [
                {
                    "source_uri": name,
                    "location": "self",
                    "meta": {
                        "type": "artifact",
                        "format": fmt,
                        "run_uid": str(run.uid),
                    },
                    "availability_state": "available",
                }
                for name, fmt in (
                    ("payload.txt", "text"),
                    ("index.yaml", "data_prod_index"),
                )
            ],
            "locations": {}
            if run is None
            else {"self": {"root_uri": ".", "location_type": "local"}},
        }
    )


def _stage(store, *, run=None, tag="main"):
    return store.publication(_index(), primary_uid=UID, run=run, tag=tag)


def _write(root, key, *, tag="main", value=None):
    store = DataProdStore(root)
    with _stage(
        store, run={"name": CounterRunNameGenerator(value=value, key=key)}, tag=tag
    ) as pending:
        (pending.path / "payload.txt").write_text(key)
        return pending.commit(_index(pending.run))


def _parallel_writer(root, key, barrier, output):
    with _stage(
        DataProdStore(root), tag=key, run={"name": CounterRunNameGenerator(key=key)}
    ) as pending:
        (pending.path / "payload.txt").write_text(key)
        barrier.wait(timeout=30)
        output.put(str(pending.commit(_index(pending.run))))


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
    assert inventory.meta["content_data_prod_type"] == TYPE
    assert DataProdStore(tmp_path).read_namespace().namespace.counter_next == 9
    assert inventory.tags["main"].current.uid == DataProdIndex.read(lower).uid
    assert DataProdStore.resolve_index(product) == lower / "index.yaml"
    assert DataProdStore.resolve_index(product, tag="test") == second / "index.yaml"
    with pytest.raises(FileExistsError):
        _write(tmp_path, "different-key", value=7)
    assert DataProdIndex.read(product) == inventory
    moved = tmp_path / "detached"
    shutil.copytree(first, moved)
    shutil.rmtree(first)
    index = DataProdIndex.read(moved)
    assert index.run.name == first.name
    assert index.run.generator.value == 0
    assert index.audit.created_at >= index.run.time_created
    assert next(iter(index.locations.values())).root_uri == moved.as_uri()
    assert (moved / "payload.txt").read_text() == "batch"


def test_explicit_and_uuid_ids_are_opaque_and_do_not_consume_counter(tmp_path):
    store = DataProdStore(tmp_path)
    with _stage(store, run={"name": "0000-batch"}) as pending:
        (pending.path / "payload.txt").write_text("literal")
        literal = pending.commit(_index(pending.run))
    with _stage(store) as pending:
        assert UUID(pending.run.name).version == 4
        (pending.path / "payload.txt").write_text("uuid")
        pending.commit(_index(pending.run))
    assert store.read_namespace().namespace.counter_next == 0
    # The automatic candidate collides with a literal ID; the allocator skips it.
    assert _write(tmp_path, "batch").name == "0001-batch"
    # No number was parsed from the literal ID, so counter value 0 is still unused.
    assert _write(tmp_path, "free", value=0).name == "0000-free"
    with pytest.raises(FileExistsError), _stage(store, run={"name": literal.name}):
        pytest.fail("A literal ID was reused")


def test_failed_run_retains_reservation_and_does_not_advance_tag(tmp_path):
    store = DataProdStore(tmp_path)
    with _stage(
        store, run={"name": CounterRunNameGenerator(value=42, key="failed")}
    ) as pending:
        root = pending.product_path
        before = DataProdIndex.read(root)
        assert store.read_namespace().namespace.counter_next == 43
        assert before.tags["main"].current is None
        assert before.products == []
        with pytest.raises(KeyError):
            store.resolve_index(root)
    assert DataProdIndex.read(root) == before
    with pytest.raises(FileExistsError):
        _write(tmp_path, "retry", value=42)
    assert _write(tmp_path, "next").name == "0043-next"
    assert (
        DataProdIndex.read(root).tags["main"].current.uid
        == DataProdIndex.read(root / "0043-next").uid
    )
    assert store.read_namespace().runs[pending.run.uid].status == "failed"


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
        assert {DataProdIndex.read(path).run.generator.value for path in paths} == {
            0,
            1,
        }
        inventory = DataProdIndex.read(paths[0].parent)
        assert DataProdStore(tmp_path).read_namespace().namespace.counter_next == 2
        assert set(inventory.tags) == {"one", "two"}
        assert len(inventory.data_items) == 1
        assert len(inventory.products) == 2
        for path in paths:
            assert (
                DataProdStore(tmp_path)
                .read_namespace()
                .runs[DataProdIndex.read(path).run.uid]
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
        assert other.name == "0001-allowed"
    finally:
        process.terminate()
        process.join(timeout=10)
    assert _write(tmp_path, "recovered").name == "0002-recovered"
    with pytest.raises(FileExistsError):
        _write(tmp_path, "reuse", value=0)


def test_failed_commit_preserves_current_and_leaves_run_reserved(tmp_path, monkeypatch):
    first = _write(tmp_path, "first")
    store = DataProdStore(tmp_path)
    root = first.parent / "index.yaml"
    replace = Path.replace
    with _stage(store, run={"name": CounterRunNameGenerator(key="orphan")}) as pending:
        (pending.path / "payload.txt").write_text("orphan")
        reserved = root.read_bytes()

        def fail(source, destination):
            if destination == root:
                msg = "interrupted index replacement"
                raise OSError(msg)
            return replace(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", fail)
            with pytest.raises(OSError, match="interrupted"):
                pending.commit(_index(pending.run))
    assert root.read_bytes() == reserved
    assert store.resolve_index(root) == first / "index.yaml"
    assert (first.parent / "0001-orphan/index.yaml").is_file()
    assert store.read_namespace().runs[pending.run.uid].status == "failed"
    with pytest.raises(KeyError):
        store.set_tag(UID, TYPE, "0001-orphan")
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
        TypeAdapter(RunRequest).validate_python({"name": spec})


def test_promotion_changes_selection_and_preserves_immutable_run(tmp_path):
    original = _write(tmp_path, "original")
    candidate = _write(tmp_path, "candidate", tag=None)
    contents = (candidate / "index.yaml").read_bytes()
    store = DataProdStore(tmp_path)
    with _stage(store), pytest.raises(DataProdTagBusyError):
        store.set_tag(UID, TYPE, candidate.name)
    assert store.resolve_index(original.parent) == original / "index.yaml"
    assert store.set_tag(UID, TYPE, candidate.name) == candidate / "index.yaml"
    assert store.resolve_index(original.parent) == candidate / "index.yaml"
    assert (
        DataProdIndex.read(original.parent).tags["main"].current.uid
        == DataProdIndex.read(candidate).uid
    )
    assert (candidate / "index.yaml").read_bytes() == contents
    store.set_tag(UID, TYPE, original.name)
    assert (
        DataProdIndex.read(original.parent).tags["main"].current.uid
        == DataProdIndex.read(original).uid
    )


def test_caller_uid_is_independent_of_name_and_cannot_be_reserved_again(tmp_path):
    uid = UUID("03f1f28e-c179-471f-a684-0767207f2975")
    store = DataProdStore(tmp_path)
    with _stage(store, run={"uid": str(uid), "name": "nightly"}) as pending:
        (pending.path / "payload.txt").write_text("data")
        path = pending.commit(_index(pending.run))
    product = DataProdIndex.read(path)
    assert path.name == "nightly"
    assert product.run.uid == uid
    assert product.uid == f"{UID}_proc/{uid}"
    membership = next(
        a for a in DataProdIndex.read(path.parent).assocs if a.uid == product.uid
    )
    assert set(membership.runs) == {uid}
    assert store.read_namespace().runs[uid].name == "nightly"
    before = (store.namespace_path / "namespace.yaml").read_bytes()
    with (
        pytest.raises(FileExistsError, match="Run UID"),
        _stage(store, run={"uid": uid, "name": "another-name"}),
    ):
        pytest.fail("One UID must not reserve a second invocation")
    with (
        pytest.raises(FileExistsError, match="Run name"),
        _stage(store, run={"name": "nightly"}),
    ):
        pytest.fail("A different UID must not reuse the name")
    assert (store.namespace_path / "namespace.yaml").read_bytes() == before
