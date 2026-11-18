"""Product placement and selection follow recorded identities, not UID text."""

import pytest
import yaml
from pydantic import ValidationError

from tolteca_datamodels.core.data_prod import DataProdRef
from tolteca_datamodels.data_prod import (
    DataProdIndex,
    DataProdStore,
    WorkflowRun,
    resolve_index,
)

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
        WorkflowRun.create() as run,
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
    with WorkflowRun.create() as run:
        with (
            pytest.raises(ValueError, match="identity changed"),
            store.materialize(_index().model_copy(update={"run": run})) as index,
        ):
            index.product.meta = index.product.meta.model_copy(
                update={"version_group": None}
            )
        assert index.base_path is None
    assert not list(store.namespace_path.glob("*/dp_raw_obs_proc/*/index.yaml"))


@pytest.mark.parametrize("document", ["realization", "version_group"])
def test_write_index_does_not_replace_published_records(tmp_path, document):
    store = DataProdStore(tmp_path)
    path = _publish(store)
    group = path.parent.parent / "index.yaml"
    index = DataProdIndex.read(path if document == "realization" else group)
    before = path.read_bytes(), group.read_bytes()
    with pytest.raises(ValueError, match=r"materialize\(\)|materialized members"):
        store.write_index(index, overwrite=True)
    assert (path.read_bytes(), group.read_bytes()) == before


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


def test_unknown_tag_label_selects_nothing(tmp_path):
    group = _publish(DataProdStore(tmp_path)).parent.parent
    with pytest.raises(KeyError):
        resolve_index(group, tag="unknown")


def test_tag_selects_a_realization_of_its_own_namespace(tmp_path):
    realization = DataProdIndex.read(_publish(DataProdStore(tmp_path / "a")))
    other = DataProdStore(tmp_path / "b")
    _publish(other)
    with pytest.raises(ValueError, match="another namespace"):
        other.set_tag(realization)


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
    # A projection writes the members and the exported group into a new namespace.
    source = DataProdStore(tmp_path / "a")
    paths = [_publish(source), _publish(source)]
    exported = DataProdIndex.read(paths[0].parent.parent)
    copy = DataProdStore(tmp_path / "b", namespace="mirror")
    directory = copy.get_product_path(UID, "dp_raw_obs_proc")
    for path in paths:
        DataProdIndex.read(path).write(directory / path.parent.name / "index.yaml")
    exported.write(directory / "index.yaml")
    added = _publish(copy)
    assert resolve_index(directory) == added
    assert len(DataProdIndex.read(directory).products) == 3
