"""Collection identity, independent result versions and exact contributor scope."""

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdAssoc,
    DataProdIndex,
    DataProdStore,
    WorkflowRun,
)
from tolteca_datamodels.data_prod.metadata import NamedGroupMeta, ObsMeta, ScienceMeta


def _observation(number):
    return DataProdIndex(
        product=DataProd(
            uid=f"tcs-{number}",
            meta=ObsMeta(
                name=str(number), data_prod_type="dp_obs", master="tcs", obsnum=number
            ),
        )
    )


def _group():
    return DataProdIndex(
        product=DataProd(
            uid="b7f65f98-4f32-4e0a-a66a-cd278670f7cc",
            meta=NamedGroupMeta(
                name="Orion collection",
                data_prod_type="dp_named_group",
                context={"project_id": "example", "source": "Orion"},
            ),
            assocs=[
                DataProdAssoc(
                    uid=f"tcs-{n}", data_prod_assoc_type="dpa_named_group_data_prod"
                )
                for n in (149101, 149102)
            ],
        )
    )


def test_named_group_is_terminal_even_when_member_has_primary_marker(tmp_path):
    group = _group()
    group.product.assocs[0].is_primary = True
    assert (
        group.product.get_primary(lambda ref: pytest.fail("membership was traversed"))
        is group.product
    )
    with (
        WorkflowRun.create() as run,
        pytest.raises(ValueError, match="Logical descriptors"),
        DataProdStore(tmp_path).materialize(group.model_copy(update={"run": run})),
    ):
        pytest.fail("A named collection was treated as a scientific realization")


@pytest.mark.parametrize(
    "uid", ["orion", "named-orion", "named-../orion", "tcs-149101"]
)
def test_named_primary_identity_cannot_escape_or_collide_with_observations(uid):
    with pytest.raises(ValueError):
        DataProd(
            uid=uid,
            meta=NamedGroupMeta(name="Invalid", data_prod_type="dp_named_group"),
        )


def test_group_identity_and_directory_are_independent_across_namespaces(tmp_path):
    group = _group()
    group.product.uid = str(uuid4())
    uid = UUID(group.product.uid)
    first = DataProdStore(tmp_path, namespace="nightly")
    second = DataProdStore(tmp_path, namespace="release")
    with pytest.raises(ValueError, match="primary_path first"):
        first.write_index(group, primary_uid=group.product.uid)
    first_path = first.write_index(
        group, primary_uid=group.product.uid, primary_path="orion"
    )
    group.product.meta.name = "Orion release"
    second_path = second.write_index(
        group, primary_uid=group.product.uid, primary_path="orion-final"
    )
    assert first_path.relative_to(first.namespace_path) == Path(
        "orion/dp_named_group.yaml"
    )
    assert second_path.relative_to(second.namespace_path) == Path(
        "orion-final/dp_named_group.yaml"
    )
    assert group.product.namespace_uid is None
    assert (
        DataProdIndex.read(first_path).product.reference
        == DataProdIndex.read(second_path).product.reference
        == group.product.reference
    )
    assert first.read_namespace_index().primary_paths == {uid: "orion"}
    # Copy the namespace descriptor; placement must survive read/write round trips.
    copied = DataProdStore(tmp_path / "copied", namespace="nightly")
    copied.write_namespace_index(first.read_namespace_index())
    assert copied.get_index_path(str(uid), "dp_named_group").relative_to(
        copied.namespace_path
    ) == Path("orion/dp_named_group.yaml")
    with pytest.raises(ValueError, match="already assigned"):
        first.write_index(
            group, primary_uid=str(uid), primary_path="renamed", overwrite=True
        )
    other = _group()
    with pytest.raises(ValueError, match="unique within"):
        first.write_index(other, primary_uid=other.product.uid, primary_path="orion")
    assert first.read_namespace_index().primary_paths == {uid: "orion"}


@pytest.mark.parametrize(
    "path",
    ["../escape", ".runs", "namespace.yaml", "149101", "149101-0-2", "ics-149101"],
)
def test_named_directory_cannot_claim_store_or_acquisition_paths(tmp_path, path):
    group = _group()
    with pytest.raises(ValueError):
        DataProdStore(tmp_path).write_index(
            group, primary_uid=group.product.uid, primary_path=path
        )


def test_collection_refresh_keeps_published_versions(tmp_path):
    store = DataProdStore(tmp_path / "dataprod_toltec", namespace="example")
    group = _group()
    store.write_index(group, primary_uid=group.product.uid, primary_path="orion")
    science = DataProdIndex(
        product=DataProd(
            uid=f"{group.product.uid}_science",
            meta=ScienceMeta(name="Orion science", data_prod_type="dp_science"),
            assocs=[
                DataProdAssoc(
                    uid=group.product.uid,
                    data_prod_assoc_type="dpa_reduction_named_group",
                    is_primary=True,
                )
            ],
        ),
        products=[group],
    )
    paths = []
    for _ in range(2):
        with (
            WorkflowRun.create() as run,
            store.materialize(science.model_copy(update={"run": run})) as out,
        ):
            pass
        assert out.base_path is not None
        paths.append(out.base_path / "index.yaml")

    refreshed = group.model_copy(deep=True)
    refreshed.product.assocs = refreshed.product.assocs[:1]
    store.write_index(refreshed, primary_uid=group.product.uid, overwrite=True)

    versions = paths[0].parent.parent
    assert versions == store.namespace_path / "orion" / "science"
    assert all(path.is_file() for path in paths)
    assert DataProdStore.resolve_index(versions) == paths[1]
    assert (
        len(
            DataProdIndex.read(
                store.get_index_path(group.product.uid, "dp_named_group")
            ).product.assocs
        )
        == 1
    )
