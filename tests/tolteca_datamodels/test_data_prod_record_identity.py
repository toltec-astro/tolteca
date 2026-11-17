"""Product placement and selection follow recorded identities, not UID text."""

from urllib.parse import quote

import pytest
import yaml
from pydantic import ValidationError

from tolteca_datamodels.core.data_prod import DataProdRef, StoreRun
from tolteca_datamodels.data_prod import DataProdIndex, DataProdStore
from tolteca_datamodels.data_prod import WorkflowRun as ExecutionRun

UID = "tcs-145609-0-1"
GROUP = f"{UID}_proc"
SCIENCE = f"dp_science_{UID}_g1"


def _index(
    uid=GROUP,
    meta=None,
    assoc_type="dpa_raw_obs_proc_raw_obs",
    target=UID,
):
    return DataProdIndex.model_validate(
        {
            "product": {
                "uid": uid,
                "meta": meta or {"name": uid, "data_prod_type": "dp_raw_obs_proc"},
                "assocs": [
                    {
                        "uid": target,
                        "data_prod_assoc_type": assoc_type,
                        "is_primary": True,
                    }
                ],
            },
        }
    )


def _publish(store, index=None, tag="main"):
    with (
        ExecutionRun.create() as run,
        store.materialize(
            (index or _index()).model_copy(update={"run": run}), tag=tag
        ) as out,
    ):
        pass
    return out.base_path / "index.yaml"


def test_realization_records_its_version_group(tmp_path):
    path = _publish(DataProdStore(tmp_path))
    realized = DataProdIndex.read(path)
    assert realized.product.meta.version_group == DataProdRef(uid=GROUP)


def test_changed_version_group_record_is_rejected(tmp_path):
    store = DataProdStore(tmp_path)
    with ExecutionRun.create() as run:
        with (
            pytest.raises(ValueError, match="identity changed"),
            store.materialize(_index().model_copy(update={"run": run})) as index,
        ):
            index.product.meta = index.product.meta.model_copy(
                update={"version_group": None}
            )
        assert index.base_path is None
    assert not list(store.namespace_path.glob("*/dp_raw_obs_proc/*/index.yaml"))


def test_exported_version_is_placed_by_its_recorded_group(tmp_path):
    science = _index(
        uid=SCIENCE,
        meta={"name": SCIENCE, "data_prod_type": "dp_science"},
        assoc_type="dpa_reduction_raw_obs",
    )
    realized = DataProdIndex.read(_publish(DataProdStore(tmp_path / "a"), science))
    # Placement uses the recorded group, not the text of the product UID.
    realized.product.uid = "renamed-realization"
    realized.store_run = StoreRun(run_uid=realized.run.uid, name="copy")
    target = DataProdStore(tmp_path / "b", namespace="mirror")
    path = target.write_index(realized, primary_uid=UID)
    group_directory = "dp_science_145609-0-1_g1"
    assert path == (
        tmp_path
        / "b"
        / "mirror"
        / "145609-0-1"
        / group_directory
        / "copy"
        / "index.yaml"
    )


def test_exported_version_without_recorded_group_is_rejected(tmp_path):
    realized = DataProdIndex.read(_publish(DataProdStore(tmp_path / "a")))
    realized.product.meta.version_group = None
    with pytest.raises(ValueError, match="must record its version group"):
        DataProdStore(tmp_path / "b").write_index(realized, primary_uid=UID)


def test_unregistered_primary_association_type_is_rejected(tmp_path):
    index = _index(assoc_type="dpa_raw_obs_proc_unknown")
    with pytest.raises(ValueError, match="Unregistered primary association type"):
        _publish(DataProdStore(tmp_path), index)


def test_primary_uid_must_match_the_declared_target_type(tmp_path):
    # dpa_quicklook_obs references an observation; an acquisition UID is invalid.
    index = _index(
        meta={
            "tag": "quicklook",
            "name": GROUP,
            "data_prod_type": "dp_quicklook",
            "workflow": "test",
            "n_inputs": 1,
        },
        assoc_type="dpa_quicklook_obs",
    )
    with pytest.raises(ValueError, match="Invalid dp_obs UID"):
        _publish(DataProdStore(tmp_path), index)


def test_tag_selects_its_member_through_the_membership_record(tmp_path):
    store = DataProdStore(tmp_path)
    first = _publish(store)
    second = _publish(store)
    group = first.parent.parent / "index.yaml"
    assert DataProdStore.resolve_index(group) == second

    # The selected member's record now holds another committed member's record.
    selected = DataProdIndex.read(second).product.uid
    members = group.parent / ".members"
    record = members / "by_uid" / f"{quote(selected, safe='')}.yaml"
    record.write_text((members / "1.yaml").read_text())
    with pytest.raises(ValueError, match="differs from the requested member"):
        DataProdStore.resolve_index(group)


def test_live_header_requires_its_document_kind(tmp_path):
    store = DataProdStore(tmp_path)
    group = _publish(store).parent.parent / "index.yaml"
    data = yaml.safe_load(group.read_text())
    assert data["document"] == "version_group_header"
    del data["document"]
    group.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        DataProdIndex.read(group)
    with pytest.raises(ValidationError):
        _publish(store)


def test_exported_group_accepts_new_versions(tmp_path):
    source = DataProdStore(tmp_path / "a")
    _publish(source)
    group = DataProdIndex.read(_publish(source).parent.parent)
    copy = DataProdStore(tmp_path / "b", namespace="mirror")
    for member in group.products:
        copy.write_index(member, primary_uid=UID)
    exported = copy.write_index(group, primary_uid=UID)
    added = _publish(copy)
    assert DataProdStore.resolve_index(exported) == added
    assert len(DataProdIndex.read(exported).products) == 3
