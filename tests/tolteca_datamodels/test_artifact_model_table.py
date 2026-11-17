"""Artifact query rows retain metadata, version choices and exact sources."""

import pytest
from tollan.accessor.pandas.model_table import ModelTableAccessor
from tollan.utils.fileloc import fileloc

from tests.data_prod_helpers import publish_model as _publish_model
from tolteca_datamodels.data_prod import (
    DataProdIndex,
    DataProdStore,
    DataProdTag,
    Location,
    artifact_table_from_indexes,
)
from tolteca_datamodels.raw_obs_context import (
    ArtifactBinding,
)


def test_metadata_query_keeps_formats_copies_and_untagged_sources(
    tmp_path, monkeypatch
):
    store = DataProdStore(tmp_path)
    first = _publish_model(store, "first")
    second = _publish_model(store, "second")
    group = DataProdIndex.read(second.base_path.parent)
    group.product.tags["review"] = DataProdTag(current=second.product.reference)
    member = next(
        i for i in group.products if i.product.reference == second.product.reference
    )
    original = next(
        i for i in member.product.data_items if i.meta.get("name") == "model"
    )
    json_item = original.model_copy(deep=True)
    json_item.meta.update(name="plot", format="json")
    json_item.source = type(json_item.source).model_validate(
        {**json_item.source.model_dump(round_trip=True), "path": "figure.json"}
    )
    html_item = json_item.model_copy(deep=True)
    html_item.meta["format"] = "html"
    html_item.source = type(html_item.source).model_validate(
        {**html_item.source.model_dump(round_trip=True), "path": "figure.html"}
    )
    member.product.data_items.extend([json_item, html_item])
    member.locations["mirror"] = Location(root="/mirror")
    mirror = json_item.model_copy(deep=True)
    mirror.source.location = "mirror"
    member.product.data_items.append(mirror)

    def forbidden(*args, **kwargs):
        pytest.fail("Projection tried to reopen an index or payload")

    monkeypatch.setattr(DataProdIndex, "read", forbidden)
    table = artifact_table_from_indexes([group, member])
    # Each distinct source survives two selecting tags, while the duplicate
    # member supplied explicitly does not add rows.
    plots = ModelTableAccessor(table).query("name == 'plot' and tag == 'main'")
    assert len(plots) == 3
    assert {"html", "json"} == set(plots.format)
    chosen = (
        ModelTableAccessor(plots)
        .query("format == 'json' and location == 'mirror'")
        .iloc[0]
    )
    binding = ArtifactBinding(
        product=chosen["product"],
        producer=chosen.producer,
        name=chosen["name"],
        source=chosen["item"].source,
    )
    assert binding.resolve(chosen["index"]).source.location == "mirror"
    assert str(fileloc(chosen.source).path) == "/mirror/figure.json"
    assert (
        len(table.loc[(table["product"] == first.product.reference) & table.tag.isna()])
        > 0
    )
    assert (
        len(
            ModelTableAccessor(table).query(
                "name == 'plot' and tag in ['main', 'review']"
            )
        )
        == 6
    )


def test_conflicting_group_reports_need_inventory_identity(tmp_path):
    first = _publish_model(DataProdStore(tmp_path), "first")
    old = DataProdIndex.read(first.base_path.parent)
    second = _publish_model(DataProdStore(tmp_path), "second")
    current = DataProdIndex.read(second.base_path.parent)
    with pytest.raises(ValueError, match="Conflicting version selections"):
        artifact_table_from_indexes([old, current])


def test_observation_selection_uses_primary_not_calibration_or_product_name(tmp_path):
    from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

    index = _publish_model(DataProdStore(tmp_path), "fit")
    from tolteca_datamodels.data_prod.metadata import PointingMeta

    index.product.meta = PointingMeta(data_prod_type="dp_pointing")
    # Reduction metadata does not need to duplicate observation identity.
    primary = next(a for a in index.product.assocs if a.is_primary)
    assert primary.uid == "tcs-145609-0-1"
    index.product.assocs.append(
        primary.model_copy(update={"is_primary": False, "uid": "tcs-149099"})
    )
    table = artifact_table_from_indexes([index])
    assert len(ObsSpec.parse("145609-0-1").select(table)) == len(table)
    assert table.obsnum.tolist() == [145609] * len(table)
    index.product.assocs = [
        a.model_copy(update={"uid": "tcs-152434"}) if a.is_primary else a
        for a in index.product.assocs
    ]
    table = artifact_table_from_indexes([index])
    assert len(ObsSpec.parse("152434").select(table)) == len(table)
    assert table.scannum.isna().all()
    assert len(ObsSpec.parse("145609").select(table)) == 0
    index.product.assocs = [
        a.model_copy(update={"uid": "822fe110-5211-4cbc-8b65-9b01e493fa61"})
        if a.is_primary
        else a
        for a in index.product.assocs
    ]
    assert artifact_table_from_indexes([index]).obsnum.isna().all()
