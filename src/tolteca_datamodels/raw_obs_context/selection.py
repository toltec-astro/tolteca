"""Shared artifact queries, diagnostics and exact source pinning."""

from uuid import UUID

from pydantic import ConfigDict, Field
from tollan.accessor.pandas.model_table import ModelTableAccessor
from tollan.config import FrozenBaseModel

from .artifact import ArtifactBinding


def _query(table, choice):
    selected = (
        table if choice is None else ModelTableAccessor(table).query(choice.select)
    )
    # Several tags or inventory reports may describe the very same source.
    return selected.drop_duplicates(["product", "location", "source_path", "source"])


def _one(table, choice, description):
    selected = _query(table, choice)
    if len(selected) != 1:
        msg = (
            f"Select exactly one {description}; "
            f"found {len(selected)} eligible artifacts"
        )
        raise ValueError(msg + _selection_details(table, selected, choice))
    return next(selected.itertuples())


def _selection_details(eligible, selected, choice):
    query = "all eligible artifacts" if choice is None else choice.select
    candidates = eligible if selected.empty else selected
    columns = [
        "product_uid",
        "namespace_uid",
        "run_name",
        "tag",
        "inventory_name",
        "producer",
        "interface",
        "scope",
        "location",
        "source_path",
    ]
    summary = candidates[columns].drop_duplicates()
    limit = 8
    details = f"\nQuery: {query}"
    if summary.empty:
        return details + "\nNo eligible artifacts were found before this query."
    details += "\n" + (
        "Eligible candidates before query:"
        if selected.empty
        else "Matching candidates:"
    )
    details += "\n" + summary.head(limit).to_string(index=False)
    if len(summary) > limit:
        details += f"\n... {len(summary) - limit} more candidate rows"
    return (
        details
        + "\nNarrow select using the version, inventory or source columns above."
    )


def _artifact(row, indexes):
    _pin(row, indexes)
    item = row.item
    binding = ArtifactBinding(
        product=row.product,
        producer=item.meta["producer"],
        name=item.meta["name"],
        interface=item.meta.get("interface"),
        scope=item.meta.get("scope"),
        source=item.source,
    )
    binding.resolve(row.index)
    return binding


def _pin(row, indexes):
    """Retain the selected product's records, including distinct reported sources."""
    incoming = row.index.model_copy(deep=True)
    incoming.products = []
    incoming.locations = {
        label: location.model_copy(
            update={"root": row.index.resolve_location(label).url}
        )
        for label, location in incoming.locations.items()
    }
    incoming._bind_locations(None)
    old = indexes.get(row.product)
    if old is None:
        indexes[row.product] = incoming
        return
    if old.product.meta != incoming.product.meta:
        msg = f"Conflicting metadata for selected product {row.product}"
        raise ValueError(msg)
    for label, location in incoming.locations.items():
        if (
            label in old.locations
            and old.resolve_location(label).url != incoming.resolve_location(label).url
        ):
            msg = (
                f"Conflicting storage location for selected product "
                f"{row.product}: {label}"
            )
            raise ValueError(msg)
        old.locations[label] = location
    existing = {(i.source.location, i.source.path): i for i in old.product.data_items}
    for item in incoming.product.data_items:
        key = (item.source.location, item.source.path)
        if key in existing and existing[key] != item:
            msg = f"Conflicting source report for selected artifact {key}"
            raise ValueError(msg)
        if key not in existing:
            old.product.data_items.append(item)
    old._bind_locations(None)


class ArtifactSelectConfig(FrozenBaseModel):
    """Select eligible calibration artifacts with a model-table expression.

    Parameters
    ----------
    select : str
        Pandas query, for example ``tag == 'main'``. The calibration role first
        restricts eligible artifacts; the expression must leave one exact source
        per target. Tags choose realizations without changing recorded lineage.
    """

    model_config = ConfigDict(extra="forbid")
    select: str = Field(min_length=1)


class InputDBConfig(FrozenBaseModel):
    """Retrieve raw records and artifact candidates from a DPDB connection.

    Parameters
    ----------
    db_bind : str
        Connection name in the shared ``db.binds`` configuration.
    namespace_uid : UUID or None, optional
        Restrict candidate products to this namespace; None includes all.
    primary_uid : str or None, optional
        Restrict candidates to this observation/acquisition's primary descendants.
        None includes all primaries. Include the recorded calibration acquisition
        when using this bound for KIDs models.
    """

    model_config = ConfigDict(extra="forbid")
    db_bind: str = Field(min_length=1)
    namespace_uid: UUID | None = Field(default=None, strict=False)
    primary_uid: str | None = None
