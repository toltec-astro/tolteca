"""Version indexes validate available members and select exact references."""

import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from uuid import uuid4

import pytest
import yaml
from pydantic import ValidationError

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdAssoc,
    DataProdIndex,
    DataProdItem,
    DataProdRef,
    DataProdSource,
    DataProdStore,
    DataProdTag,
    Location,
    WorkflowRun,
    resolve_index,
)
from tolteca_datamodels.data_prod.metadata import (
    RawObsProcMeta,
    ScienceMeta,
    VersionGroupMeta,
)


@pytest.fixture
def version_group():
    uid = "tcs-145609-0-1_proc"
    namespace_uid = uuid4()
    with WorkflowRun.create() as older_run, WorkflowRun.create() as selected_run:
        pass
    members = [
        DataProdIndex(
            product=DataProd(
                uid=f"{uid}/{run.uid}",
                namespace_uid=namespace_uid,
                meta=RawObsProcMeta(
                    name=uid,
                    data_prod_type="dp_raw_obs_proc",
                    version_group=DataProdRef(uid=uid),
                ),
                assocs=[
                    DataProdAssoc(
                        uid="tcs-145609-0-1",
                        data_prod_assoc_type="dpa_raw_obs_proc_raw_obs",
                        is_primary=True,
                    )
                ],
            ),
            run=run,
        )
        for run in (older_run, selected_run)
    ]
    older, selected = (member.product.reference for member in members)
    elsewhere = DataProdRef(uid=selected.uid, namespace_uid=uuid4())
    return DataProdIndex(
        product=DataProd(
            uid=uid,
            meta=VersionGroupMeta(
                name=uid,
                data_prod_type="dp_version_group",
                content_data_prod_type="dp_raw_obs_proc",
            ),
            assocs=[
                DataProdAssoc(
                    uid="tcs-145609-0-1",
                    data_prod_assoc_type="dpa_version_group_raw_obs",
                    is_primary=True,
                ),
                *[
                    DataProdAssoc(
                        uid=reference.uid,
                        namespace_uid=reference.namespace_uid,
                        data_prod_assoc_type="dpa_version_group_data_prod",
                    )
                    for reference in (older, selected, elsewhere)
                ],
                DataProdAssoc(
                    uid="tcs-145609",
                    data_prod_assoc_type="dpa_version_group_obs",
                ),
            ],
            tags={
                "main": DataProdTag(current=selected),
                "older": DataProdTag(current=older),
                "elsewhere": DataProdTag(current=elsewhere),
                "empty": DataProdTag(),
            },
        ),
        products=members,
    )


def test_version_selection_uses_exact_member_references(version_group, tmp_path):
    path = version_group.write(tmp_path / "index.yaml")
    selected = version_group.products[1]
    result = DataProdIndex.read(path, run_uids=[selected.run.uid])

    assert [member.product.reference for member in result.products] == [
        selected.product.reference
    ]
    assert result.product.assocs == [
        version_group.product.assocs[0],  # primary acquisition
        version_group.product.assocs[2],  # selected membership, correct namespace
        version_group.product.assocs[4],  # another non-membership relationship
    ]
    assert result.product.tags == {"main": version_group.product.tags["main"]}


def test_version_group_is_not_materialized(version_group, tmp_path):
    store = DataProdStore(tmp_path)
    with WorkflowRun.create() as run:
        pass
    index = version_group.model_copy(update={"run": run})
    with (
        pytest.raises(ValueError, match="forms from its materialized members"),
        store.materialize(index),
    ):
        pass
    # Rejected before the store reserves a run name.
    assert not (store.namespace_path / ".runs").exists()


def test_empty_version_selection_keeps_non_membership_relationships(
    version_group, tmp_path
):
    path = version_group.write(tmp_path / "index.yaml")
    result = DataProdIndex.read(path, run_uids=[])

    assert result.products == []
    assert result.product.assocs == [
        version_group.product.assocs[0],
        version_group.product.assocs[4],
    ]
    assert result.product.tags == {}


UID = "tcs-145609-0-1_proc"


def _draft(run, *, locations=None, data_items=()):
    """Describe one realization of the processed-acquisition version group."""
    return DataProdIndex(
        product=DataProd(
            uid=UID,
            meta=RawObsProcMeta(name=UID, data_prod_type="dp_raw_obs_proc"),
            assocs=[
                DataProdAssoc(
                    uid="tcs-145609-0-1",
                    data_prod_assoc_type="dpa_raw_obs_proc_raw_obs",
                    is_primary=True,
                )
            ],
            data_items=list(data_items),
        ),
        locations=locations or {},
        run=run,
    )


def _publish(store, name, **draft):
    """Publish one realization through the store."""
    with (
        WorkflowRun.create() as run,
        store.materialize(_draft(run, **draft), run_name=name) as index,
    ):
        pass
    return index


def _publish_interrupted(store, directory, name, monkeypatch):
    """Publish a realization whose version-group header is never replaced."""
    root = directory / "index.yaml"
    replace = Path.replace

    def fail(source, destination):
        if destination == root:
            msg = "interrupted index replacement"
            raise OSError(msg)
        return replace(source, destination)

    with (
        WorkflowRun.create() as run,
        pytest.raises(OSError, match="interrupted"),
        monkeypatch.context() as patch,
        store.materialize(_draft(run), run_name=name),
    ):
        patch.setattr(Path, "replace", fail)
    return run.uid


@pytest.fixture
def published(tmp_path):
    """A version group with two realizations committed by the store."""
    store = DataProdStore(tmp_path)
    members = [_publish(store, name) for name in ("version-1", "version-2")]
    return store, members[0].base_path.parent, members


def test_compact_header_requires_version_group(version_group, tmp_path):
    data = version_group.products[0].model_dump(mode="json", round_trip=True)
    data["document"] = "version_group_header"
    data["version_count"] = 0
    path = tmp_path / "index.yaml"
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValidationError, match="Expected a version group index"):
        DataProdIndex.read(path)


@pytest.mark.parametrize("compact", [False, True], ids=["expanded", "compact"])
def test_available_member_requires_declared_type(
    compact, version_group, published, tmp_path
):
    if compact:
        _, path, members = published
        member_path = members[1].base_path / "index.yaml"
        data = yaml.safe_load(member_path.read_text())
        metadata = data["product"]["meta"]
        requested = members[1].run.uid
    else:
        path = member_path = version_group.write(tmp_path / "expanded" / "index.yaml")
        data = yaml.safe_load(path.read_text())
        metadata = data["products"][1]["product"]["meta"]
        requested = version_group.products[1].run.uid
    replacement = ScienceMeta(
        name=metadata["name"],
        data_prod_type="dp_science",
        version_group=DataProdRef(uid=UID),
    )
    metadata.clear()
    metadata.update(replacement.model_dump(mode="json", round_trip=True))
    member_path.write_text(yaml.safe_dump(data))

    with pytest.raises(
        ValidationError, match="must have product type 'dp_raw_obs_proc'"
    ):
        DataProdIndex.read(path, run_uids=[requested])


def test_supporting_product_is_not_a_version_member(version_group, tmp_path):
    # The same UID in another namespace need not belong to this group's type.
    supporting = DataProdIndex(
        product=DataProd(
            uid=version_group.products[0].product.uid,
            namespace_uid=uuid4(),
            meta=ScienceMeta(name="supporting", data_prod_type="dp_science"),
        )
    )
    version_group.products.append(supporting)
    path = version_group.write(tmp_path / "index.yaml")

    result = DataProdIndex.read(path)

    assert result.products[-1].product == supporting.product
    assert result.product.assocs == version_group.product.assocs


def test_committed_member_requires_membership_edge(published):
    _, directory, _ = published
    record_path = directory / ".members" / "1.yaml"
    data = yaml.safe_load(record_path.read_text())
    data["edge"]["data_prod_assoc_type"] = "dpa_version_group_raw_obs"
    record_path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValidationError, match="version-group membership association"):
        DataProdIndex.read(directory)


@pytest.mark.parametrize("selection", ["run", "tag"])
def test_committed_lookup_must_identify_the_requested_member(selection, published):
    _, directory, members = published
    selected = members[1]
    # The selected member's lookup record now holds another member's record.
    lookup = (
        directory
        / ".members"
        / "by_uid"
        / f"{quote(selected.product.uid, safe='')}.yaml"
    )
    lookup.write_text((directory / ".members" / "1.yaml").read_text())
    read = {
        "run": lambda: DataProdIndex.read(directory, run_uids=[selected.run.uid]),
        "tag": lambda: resolve_index(directory),
    }[selection]

    with pytest.raises(ValueError, match="differs from the requested member"):
        read()


@pytest.mark.parametrize("state", ["absent", "interrupted", "superseded"])
def test_absent_or_uncommitted_requests_are_omitted(state, published, monkeypatch):
    store, directory, members = published
    requested = uuid4()
    if state != "absent":
        requested = _publish_interrupted(store, directory, "orphan", monkeypatch)
    if state == "superseded":
        # A later publication takes the position the interrupted one recorded.
        _publish(store, "version-3")

    result = DataProdIndex.read(directory, run_uids=[requested, members[1].run.uid])

    assert [member.product.reference for member in result.products] == [
        members[1].product.reference
    ]


def test_version_reads_preserve_source_resolution(tmp_path):
    store = DataProdStore(tmp_path)
    _publish(store, "version-1")
    member = _publish(
        store,
        "version-2",
        locations={
            "local": Location(root="payloads"),
            "archive": Location(root="https://archive.example/products"),
        },
        data_items=[
            DataProdItem(
                source=DataProdSource(
                    location=location, path=PurePosixPath("model.ecsv")
                )
            )
            for location in ("local", "archive")
        ],
    )
    directory = member.base_path.parent
    full = DataProdIndex.read(directory)
    selected = DataProdIndex.read(directory, run_uids=[member.run.uid])
    exact = DataProdIndex.read(member.base_path)
    expanded_path = full.write(directory / "expanded.yaml")
    expanded = DataProdIndex.read(expanded_path, run_uids=[member.run.uid])
    expected = [
        (member.base_path / "payloads/model.ecsv").as_uri(),
        "https://archive.example/products/model.ecsv",
    ]

    def model_uris(index):
        return [
            index.get_source_uri(item.source)
            for item in index.product.data_items
            if item.source.path.name == "model.ecsv"
        ]

    for index in (full.products[1], selected.products[0], exact, expanded.products[0]):
        assert index.product.reference == member.product.reference
        assert model_uris(index) == expected

    detached_path = tmp_path / "detached"
    shutil.copytree(member.base_path, detached_path)
    detached = DataProdIndex.read(detached_path)

    assert detached.product.reference == member.product.reference
    assert model_uris(detached) == [
        (detached_path / "payloads/model.ecsv").as_uri(),
        expected[1],
    ]
