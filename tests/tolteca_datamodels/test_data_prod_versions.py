"""Version indexes validate available members and select exact references."""

import shutil
from pathlib import PurePosixPath
from urllib.parse import quote
from uuid import uuid4

import pytest
import yaml
from pydantic import ValidationError

from tolteca_datamodels.core.data_prod import StoreRun
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


def _write_compact_group(group, directory):
    """Prepare serialized input for the version reader."""
    data = group.model_dump(mode="json", round_trip=True, exclude={"products"})
    data["product"]["assocs"] = [
        assoc
        for assoc in data["product"]["assocs"]
        if assoc["data_prod_assoc_type"] != "dpa_version_group_data_prod"
    ]
    data["document"] = "version_group_header"
    data["version_count"] = len(group.products)
    for position, member in enumerate(group.products, 1):
        name = f"version-{position}"
        member.store_run = StoreRun(run_uid=member.run.uid, name=name)
        member.write(directory / name / "index.yaml")
        edge = DataProdAssoc(
            uid=member.product.uid,
            namespace_uid=member.product.namespace_uid,
            data_prod_assoc_type="dpa_version_group_data_prod",
        )
        _write_member_record(
            directory,
            member.product.uid,
            {"position": position, "name": name, "edge": edge.model_dump(mode="json")},
        )
    path = directory / "index.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _lookup_path(directory, member_uid):
    return directory / ".members" / "by_uid" / f"{quote(member_uid, safe='')}.yaml"


def _write_member_record(directory, member_uid, data):
    root = directory / ".members"
    (root / "by_uid").mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data)
    _lookup_path(directory, member_uid).write_text(text)
    (root / f"{data['position']}.yaml").write_text(text)


def test_compact_header_requires_version_group(version_group, tmp_path):
    data = version_group.products[0].model_dump(mode="json", round_trip=True)
    data["document"] = "version_group_header"
    data["version_count"] = 0
    path = tmp_path / "index.yaml"
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValidationError, match="Expected a version group index"):
        DataProdIndex.read(path)


@pytest.mark.parametrize("compact", [False, True], ids=["expanded", "compact"])
def test_available_member_requires_declared_type(compact, version_group, tmp_path):
    if compact:
        path = _write_compact_group(version_group, tmp_path)
        member_path = tmp_path / "version-2" / "index.yaml"
        data = yaml.safe_load(member_path.read_text())
        metadata = data["product"]["meta"]
    else:
        path = member_path = version_group.write(tmp_path / "index.yaml")
        data = yaml.safe_load(path.read_text())
        metadata = data["products"][1]["product"]["meta"]
    replacement = ScienceMeta(
        name=metadata["name"],
        data_prod_type="dp_science",
        version_group=DataProdRef(uid=version_group.product.uid),
    )
    metadata.clear()
    metadata.update(replacement.model_dump(mode="json", round_trip=True))
    member_path.write_text(yaml.safe_dump(data))

    with pytest.raises(
        ValidationError, match="must have product type 'dp_raw_obs_proc'"
    ):
        DataProdIndex.read(path, run_uids=[version_group.products[1].run.uid])


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


def test_compact_member_requires_membership_edge(version_group, tmp_path):
    path = _write_compact_group(version_group, tmp_path)
    record_path = tmp_path / ".members" / "1.yaml"
    data = yaml.safe_load(record_path.read_text())
    data["edge"]["data_prod_assoc_type"] = "dpa_version_group_raw_obs"
    record_path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValidationError, match="version-group membership association"):
        DataProdIndex.read(path)


def test_committed_lookup_must_identify_requested_run(version_group, tmp_path):
    path = _write_compact_group(version_group, tmp_path)
    requested = version_group.products[0].run.uid
    lookup = _lookup_path(tmp_path, version_group.products[0].product.uid)
    lookup.write_text((tmp_path / ".members" / "2.yaml").read_text())

    with pytest.raises(ValueError, match="differs from the requested member"):
        DataProdIndex.read(path, run_uids=[requested])


@pytest.mark.parametrize("state", ["absent", "beyond_count", "uncommitted_lookup"])
def test_absent_or_uncommitted_requests_are_omitted(state, version_group, tmp_path):
    path = _write_compact_group(version_group, tmp_path)
    requested = uuid4()
    selected = version_group.products[0]
    data = yaml.safe_load((tmp_path / ".members" / "1.yaml").read_text())
    if state == "beyond_count":
        data["position"] = 3
        _write_member_record(tmp_path, f"{version_group.product.uid}/{requested}", data)
    elif state == "uncommitted_lookup":
        data["name"] = "abandoned"
        lookup = _lookup_path(tmp_path, f"{version_group.product.uid}/{requested}")
        lookup.write_text(yaml.safe_dump(data))

    result = DataProdIndex.read(path, run_uids=[requested, selected.run.uid])

    assert [member.product.reference for member in result.products] == [
        selected.product.reference
    ]


def test_version_reads_preserve_source_resolution(version_group, tmp_path):
    member = version_group.products[1]
    member.locations = {
        "local": Location(root="payloads"),
        "archive": Location(root="https://archive.example/products"),
    }
    member.product.data_items = [
        DataProdItem(
            source=DataProdSource(location=location, path=PurePosixPath("model.ecsv"))
        )
        for location in member.locations
    ]
    path = _write_compact_group(version_group, tmp_path)
    full = DataProdIndex.read(path)
    selected = DataProdIndex.read(path, run_uids=[member.run.uid])
    exact = DataProdIndex.read(tmp_path / "version-2")
    expanded_path = full.write(tmp_path / "expanded.yaml")
    expanded = DataProdIndex.read(expanded_path, run_uids=[member.run.uid])
    expected = [
        (tmp_path / "version-2/payloads/model.ecsv").as_uri(),
        "https://archive.example/products/model.ecsv",
    ]
    for index in (full.products[1], selected.products[0], exact, expanded.products[0]):
        assert index.product.reference == member.product.reference
        assert [
            index.get_source_uri(item.source) for item in index.product.data_items
        ] == expected

    detached_path = tmp_path / "detached"
    shutil.copytree(tmp_path / "version-2", detached_path)
    detached = DataProdIndex.read(detached_path)

    assert detached.product.reference == member.product.reference
    assert [
        detached.get_source_uri(item.source) for item in detached.product.data_items
    ] == [
        (detached_path / "payloads/model.ecsv").as_uri(),
        expected[1],
    ]
