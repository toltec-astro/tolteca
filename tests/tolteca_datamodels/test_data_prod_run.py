"""Execution lifecycle and its namespace/store persistence boundary."""

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

import pytest
from pydantic import Field, ValidationError, create_model
from tollan.config import RuntimeInfo

from tolteca_datamodels.core.data_prod import (
    Actor,
    CounterRunNameGenerator,
    DataProdStoreBase,
    WorkflowRun,
)


@pytest.mark.parametrize(
    ("values", "field", "error_type"),
    [
        ({"uid": ""}, "uid", "uuid_parsing"),
        ({"context": []}, "context", "dict_type"),
        ({"meta": []}, "meta", "dict_type"),
        ({"context": {b"gain": 2}}, "context", "string_type"),
        ({"meta": {b"purpose": "review"}}, "meta", "string_type"),
    ],
    ids=["empty-uid", "context-list", "meta-list", "context-key", "meta-key"],
)
def test_create_rejects_invalid_record_values(values, field, error_type):
    with pytest.raises(ValidationError) as exc_info:
        WorkflowRun.create(**values)

    assert [(error["loc"][0], error["type"]) for error in exc_info.value.errors()] == [
        (field, error_type)
    ]


@pytest.mark.parametrize("factory_default", [False, True], ids=["value", "factory"])
def test_create_validates_subclass_defaults(factory_default):
    default = (
        Field(default_factory=lambda: "invalid")
        if factory_default
        else Field(default="invalid")
    )
    run_cls = create_model(
        "InvalidStatusRun",
        __base__=WorkflowRun,
        status=(WorkflowRun.model_fields["status"].annotation, default),
    )

    with pytest.raises(ValidationError) as exc_info:
        run_cls.create()

    assert [(error["loc"], error["type"]) for error in exc_info.value.errors()] == [
        (("status",), "literal_error")
    ]


@pytest.mark.parametrize("uid", [None, UUID(int=1), str(UUID(int=1))])
def test_create_retains_supplied_identity_and_captures_runtime(uid):
    actor = Actor(
        uid=UUID(int=2), name="review", created_at=datetime(2026, 11, 9, tzinfo=UTC)
    )
    runtime = RuntimeInfo(username="review", hostname="host")
    metadata = {"purpose": "review", "runtime_info": {"old": "value"}}
    context = {"gain": 2}

    run = WorkflowRun.create(
        uid=uid, actor=actor, runtime_info=runtime, context=context, meta=metadata
    )

    assert isinstance(run.uid, UUID)
    if uid is not None:
        assert run.uid == UUID(str(uid))
    assert run.actor == actor
    assert run.context == context
    assert run.meta["purpose"] == "review"
    assert run.runtime_info == runtime
    assert metadata == {"purpose": "review", "runtime_info": {"old": "value"}}


def test_create_uses_empty_context_and_metadata_for_none():
    run = WorkflowRun.create(context=None, meta=None)

    assert run.context == {}
    assert set(run.meta) == {"runtime_info"}
    assert run.status == "running"
    assert run.time_completed is None


def test_run_keeps_uuid_and_timestamp_input_formats():
    run = WorkflowRun.create()
    values = run.model_dump(round_trip=True)
    values["uid"] = str(run.uid)
    values["time_created"] = run.time_created.isoformat()

    assert WorkflowRun.model_validate(values) == run
    assert WorkflowRun.model_validate_json(run.model_dump_json(round_trip=True)) == run


def test_execution_is_independent_of_store_and_namespace(tmp_path):
    run = WorkflowRun.create(context={"gain": 2})
    assert list(tmp_path.iterdir()) == []
    stores = [
        DataProdStoreBase(tmp_path, namespace=n) for n in ("quicklook", "science")
    ]
    with run:
        for store in stores:
            store.write_run(run)
        assert all(store.read_run(run.uid).status == "running" for store in stores)
    for store in stores:
        store.write_run(run)
        assert store.read_run(run.uid) == run
        exported = store.read_namespace_index()
        assert exported.runs == {run.uid: run}
        assert exported.store_runs == {}
        assert "namespace_uid" not in run.model_dump()
        assert "name" not in run.model_dump()
    assert (
        stores[0].read_namespace().namespace.uid
        != stores[1].read_namespace().namespace.uid
    )
    assert run.status == "completed"


def test_store_reserves_portable_run_without_runtime_capture(tmp_path):
    actor = Actor(
        uid=uuid4(), name="imported", created_at=datetime(2026, 11, 9, tzinfo=UTC)
    )
    run = WorkflowRun(uid=uuid4(), actor=actor, time_created=actor.created_at)
    store = DataProdStoreBase(tmp_path, namespace="science")

    store.write_run(run, run_name="external")
    store.write_run(run, run_name="external")

    saved = store.read_namespace_index()
    assert saved.runs == {run.uid: run}
    assert saved.actors == {actor.uid: actor}
    assert saved.store_runs[run.uid].name == "external"
    assert saved.runs[run.uid].meta == {}


@pytest.mark.parametrize(
    ("name", "name_conflicting"),
    [
        ("nightly", "another"),
        ("nightly", CounterRunNameGenerator()),
        (
            CounterRunNameGenerator(value=7, key="nightly"),
            CounterRunNameGenerator(key="another"),
        ),
        (
            CounterRunNameGenerator(value=7, key="nightly"),
            CounterRunNameGenerator(value=8, key="nightly"),
        ),
        (
            CounterRunNameGenerator(value=7, key="nightly"),
            CounterRunNameGenerator(format="{value:06d}-{key}", key="nightly"),
        ),
    ],
    ids=[
        "fixed-name",
        "fixed-to-counter",
        "counter-key",
        "counter-value",
        "counter-format",
    ],
)
def test_saved_execution_rejects_conflicting_naming_policy(
    tmp_path, name, name_conflicting
):
    store = DataProdStoreBase(tmp_path, namespace="science")
    run = WorkflowRun.create()
    store.write_run(run, run_name=name)
    saved = store.read_namespace_index(run_uids=[run.uid])

    with pytest.raises(
        ValueError, match="Conflicting (output name|store-run naming policy)"
    ):
        store.write_run(run, run_name=name_conflicting)

    assert store.read_namespace_index(run_uids=[run.uid]) == saved
    store.write_run(run, run_name=name)
    assert store.read_namespace_index(run_uids=[run.uid]) == saved


@pytest.mark.parametrize(
    "interrupted_after",
    ["counter_advanced", "counter_claimed", "name_claimed", "reservation_saved"],
)
def test_counter_reservation_interruption_retains_claims(
    tmp_path, monkeypatch, interrupted_after
):
    from tolteca_datamodels.core.data_prod import store as store_module

    store = DataProdStoreBase(tmp_path, namespace="science")
    run = WorkflowRun.create()
    records = store.namespace_path / ".runs"
    write_yaml = store_module._write_yaml
    sync_directory = store_module._sync_directory

    def write_then_interrupt(record, path, **kwargs):
        result = write_yaml(record, path, **kwargs)
        if (
            interrupted_after == "counter_advanced"
            and path.name == "namespace.yaml"
            and record.namespace.counter_next == 8
        ) or (
            interrupted_after == "reservation_saved" and path.name == "store_run.yaml"
        ):
            msg = "reservation interrupted"
            raise OSError(msg)
        return result

    def sync_then_interrupt(path):
        sync_directory(path)
        if (interrupted_after, path.name) in {
            ("counter_claimed", "by_counter"),
            ("name_claimed", "by_name"),
        }:
            msg = "reservation interrupted"
            raise OSError(msg)

    with monkeypatch.context() as patch:
        patch.setattr(store_module, "_write_yaml", write_then_interrupt)
        patch.setattr(store_module, "_sync_directory", sync_then_interrupt)
        with pytest.raises(OSError, match="reservation interrupted"):
            store.write_run(run, run_name=CounterRunNameGenerator(value=7, key="first"))

    reopened = DataProdStoreBase(tmp_path, namespace="science")
    assert reopened.read_namespace().namespace.counter_next == 8
    counter_claimed = (records / "by_counter/7").exists()
    name_claimed = (records / "by_name/0007-first").exists()
    if counter_claimed or name_claimed:
        with pytest.raises(FileExistsError):
            reopened.write_run(
                WorkflowRun.create(),
                run_name=CounterRunNameGenerator(value=7, key="another"),
            )

    reopened.write_run(run, run_name=CounterRunNameGenerator(key="first"))
    saved = reopened.read_namespace_index(run_uids=[run.uid])
    completed = interrupted_after == "reservation_saved"
    assert saved.store_runs[run.uid].name == (
        "0007-first" if completed else "0008-first"
    )
    assert saved.namespace.counter_next == (8 if completed else 9)
    assert (records / "by_counter/7").exists() == counter_claimed
    assert (records / "by_name/0007-first").exists() == name_claimed


def test_interrupted_fixed_name_requires_manual_claim_removal(tmp_path, monkeypatch):
    from tolteca_datamodels.core.data_prod import store as store_module

    store = DataProdStoreBase(tmp_path, namespace="science")
    run = WorkflowRun.create()
    write_yaml = store_module._write_yaml

    def interrupt_reservation(record, path, **kwargs):
        if path.name == "store_run.yaml":
            msg = "reservation interrupted"
            raise OSError(msg)
        return write_yaml(record, path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(store_module, "_write_yaml", interrupt_reservation)
        with pytest.raises(OSError, match="reservation interrupted"):
            store.write_run(run, run_name="fixed")

    reopened = DataProdStoreBase(tmp_path, namespace="science")
    assert reopened.read_namespace_index(run_uids=[run.uid]).store_runs == {}
    claim = reopened.namespace_path / ".runs/by_name/fixed"
    assert claim.is_dir()
    with pytest.raises(FileExistsError, match="already reserved"):
        reopened.write_run(run, run_name="fixed")

    claim.rmdir()
    reopened.write_run(run, run_name="fixed")
    saved = reopened.read_namespace_index(run_uids=[run.uid])
    assert saved.store_runs[run.uid].name == "fixed"
    assert reopened.read_namespace().namespace.counter_next == 0


@pytest.mark.parametrize("same_execution", [True, False])
def test_conflicting_run_or_actor_preserves_saved_records(tmp_path, same_execution):
    store = DataProdStoreBase(tmp_path, namespace="science")
    original = WorkflowRun.create()
    store.write_run(original)
    saved = store.read_namespace_index()
    values = original.model_dump(round_trip=True)
    values["actor"]["name"] = "conflicting-name"
    if not same_execution:
        values["uid"] = uuid4()
    conflicting = WorkflowRun.model_validate(values)

    with pytest.raises(ValueError, match="Conflicting (execution|actor)"):
        store.write_run(conflicting)

    assert store.read_namespace_index() == saved


def test_store_preserves_completed_execution_on_repeated_writes(tmp_path):
    store = DataProdStoreBase(tmp_path, namespace="science")
    run = WorkflowRun.create()
    running = run.model_copy(deep=True)
    store.write_run(run)
    with run:
        pass
    store.write_run(run)
    store.write_run(running)
    assert store.read_run(run.uid) == run

    conflicting = run.model_copy(update={"status": "failed"})
    with pytest.raises(ValueError, match="Conflicting completed execution"):
        store.write_run(conflicting)
    assert store.read_run(run.uid) == run


def test_namespace_path_normalization_and_persistence(tmp_path):
    store = DataProdStoreBase(tmp_path, namespace="./archive//night 1/")
    assert store.namespace == PurePosixPath("archive/night 1")
    assert store.namespace_path == tmp_path / "archive/night 1"
    assert list(tmp_path.iterdir()) == []

    descriptor = store.ensure_namespace()
    assert descriptor.namespace.path == store.namespace
    assert descriptor.namespace.model_dump(mode="json")["path"] == "archive/night 1"
    assert store.read_namespace() == descriptor
    reopened = DataProdStoreBase(tmp_path, namespace=store.namespace)
    assert reopened.read_namespace().namespace.uid == descriptor.namespace.uid


@pytest.mark.parametrize("namespace_first", [True, False])
def test_namespace_and_runs_share_default_environment_actor(tmp_path, namespace_first):
    runtime = RuntimeInfo()
    store = DataProdStoreBase(tmp_path, namespace="science")
    if namespace_first:
        store.ensure_namespace(runtime_info=runtime)
    run = WorkflowRun.create(runtime_info=runtime)
    descriptor = store.ensure_namespace(runtime_info=runtime)
    store.write_run(run)

    assert descriptor.namespace.audit.created_by == run.actor_uid
    assert store.read_namespace().actors == {run.actor_uid: run.actor}
    other = DataProdStoreBase(tmp_path, namespace="quicklook").ensure_namespace(
        runtime_info=runtime
    )
    assert other.namespace.uid != descriptor.namespace.uid
    assert other.actors == descriptor.actors

    next_run = WorkflowRun.create(
        runtime_info=runtime.model_copy(
            update={"cmd": "another invocation", "exec_path": Path("/another/entry")}
        ),
        context={"gain": 2},
    )
    assert next_run.uid != run.uid
    assert next_run.actor == run.actor
    assert next_run.runtime_info.cmd == "another invocation"


@pytest.mark.parametrize(
    "changed",
    [
        {"hostname": "other-host"},
        {"username": "other-user"},
        {"platform": "other-platform"},
        {"python_prefix": Path("/other/environment")},
        {"timezone": "UTC"},
    ],
    ids=["host", "user", "platform", "environment", "timezone"],
)
def test_changed_environment_selects_another_actor(tmp_path, changed):
    runtime = RuntimeInfo(
        hostname="host",
        username="user",
        platform="platform",
        python_prefix=Path("/environment"),
        timezone="America/New_York",
    )
    store = DataProdStoreBase(tmp_path, namespace="science")
    original = store.ensure_namespace(runtime_info=runtime)
    original_uid = original.namespace.audit.created_by
    new_runtime = runtime.model_copy(update=changed)
    run = WorkflowRun.create(runtime_info=new_runtime)
    current = store.ensure_namespace(runtime_info=new_runtime)

    assert run.actor_uid != original_uid
    assert current.actors[run.actor_uid] == run.actor
    assert current.namespace.audit == original.namespace.audit
    assert set(current.actors) == {original_uid, run.actor_uid}


def test_imported_actor_is_reused_without_replacing_run_actors(tmp_path):
    runtime = RuntimeInfo()
    default_run = WorkflowRun.create(runtime_info=runtime)
    recorded_actor = default_run.actor.model_copy(
        update={"uid": uuid4(), "created_at": datetime(2024, 1, 1, tzinfo=UTC)}
    )
    recorded_run = WorkflowRun.create(runtime_info=runtime, actor=recorded_actor)
    origin = DataProdStoreBase(tmp_path / "origin", namespace="science")
    origin.write_run(recorded_run)
    exported = origin.read_namespace_index()

    store = DataProdStoreBase(tmp_path / "copy", namespace="science")
    store.write_namespace_index(exported)
    descriptor = store.ensure_namespace(runtime_info=runtime)
    assert descriptor.actors == {recorded_actor.uid: recorded_actor}
    assert descriptor.namespace.audit == exported.namespace.audit
    assert store.read_run(recorded_run.uid) == recorded_run

    # A supplied execution keeps its own actor even when the facts match.
    store.write_run(default_run)
    retained = store.ensure_namespace(runtime_info=runtime)
    assert retained.actors == {
        recorded_actor.uid: recorded_actor,
        default_run.actor_uid: default_run.actor,
    }
    assert store.read_run(default_run.uid).actor == default_run.actor
    assert store.read_run(recorded_run.uid).actor == recorded_actor


@pytest.mark.parametrize("history_size", [0, 200])
def test_store_run_writes_do_not_load_history(tmp_path, monkeypatch, history_size):
    from tolteca_datamodels.core.data_prod import store as store_module

    store = DataProdStoreBase(tmp_path, namespace="lab")
    for _ in range(history_size):
        with WorkflowRun.create() as prior:
            pass
        store.write_run(prior, run_name=CounterRunNameGenerator())
    reads = []
    load = store_module._read_yaml

    def observed(path):
        reads.append(path)
        return load(path)

    monkeypatch.setattr(store_module, "_read_yaml", observed)
    with WorkflowRun.create(context={"gain": 2}) as run:
        store.write_run(run)
    store.write_run(run)

    def reject_history_scan(*args, **kwargs):
        pytest.fail("Saving a run or reservation must not scan run history")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "glob", reject_history_scan)
        patch.setattr(Path, "iterdir", reject_history_scan)
        store.write_run(run, run_name=CounterRunNameGenerator())
        store.write_run(run, run_name=CounterRunNameGenerator())
    run_reads = [p for p in reads if p.name in {"run.yaml", "store_run.yaml"}]
    assert {p.parent.name for p in run_reads} == {str(run.uid)}
    assert store.read_namespace_index(run_uids=[run.uid]).runs == {run.uid: run}
    assert len(store.read_namespace_index().runs) == history_size + 1


def test_run_failure_and_cancellation_do_not_depend_on_persistence(tmp_path):
    msg = "computation failed"
    with pytest.raises(RuntimeError), WorkflowRun.create() as failed:
        raise RuntimeError(msg)
    assert failed.status == "failed"
    with WorkflowRun.create() as reported:
        reported.fail()
    assert reported.status == "failed"
    with pytest.raises(KeyboardInterrupt), WorkflowRun.create() as cancelled:
        raise KeyboardInterrupt
    assert cancelled.status == "cancelled"
    assert all(r.time_completed is not None for r in (failed, reported, cancelled))
    assert list(tmp_path.iterdir()) == []
    store = DataProdStoreBase(tmp_path, namespace="lab")
    for run in (failed, reported, cancelled):
        store.write_run(run)
        assert store.read_run(run.uid) == run
