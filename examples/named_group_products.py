"""Publish illustrative result versions under a refreshable named collection.

Run with ``python examples/named_group_products.py OUTPUT_DIRECTORY``.
This demonstrates persistence using selection receipts; it does not compute maps.
"""

# ruff: noqa: INP001 -- runnable standalone guide example
import json
from pathlib import Path

from tolteca_datamodels.data_prod import (
    CounterRunNameGenerator,
    DataProd,
    DataProdAssoc,
    DataProdIndex,
    DataProdItem,
    DataProdSource,
    DataProdStore,
    Location,
    WorkflowRun,
)
from tolteca_datamodels.data_prod.metadata import NamedGroupMeta, ObsMeta, ScienceMeta

# Fixed identity for this repeatable example. New collections generate uuid4() once.
GROUP_UID = "b7f65f98-4f32-4e0a-a66a-cd278670f7cc"


def collection(numbers):
    """Describe an example collection with explicitly selected observations."""
    return DataProdIndex(
        product=DataProd(
            uid=GROUP_UID,
            meta=NamedGroupMeta(
                name="Orion",
                data_prod_type="dp_named_group",
                context={"project_id": "example", "source": "Orion"},
            ),
            assocs=[
                DataProdAssoc(
                    uid=f"tcs-{n}", data_prod_assoc_type="dpa_named_group_data_prod"
                )
                for n in numbers
            ],
        )
    )


def publish(store, group, run, numbers, *, uid=None):
    """Materialize exact selection receipts for one logical result."""
    uid = uid or f"{group.product.uid}_science"
    description = DataProdIndex(
        product=DataProd(
            uid=uid,
            meta=ScienceMeta(name=uid, data_prod_type="dp_science"),
            assocs=[
                DataProdAssoc(
                    uid=group.product.uid,
                    data_prod_assoc_type="dpa_reduction_named_group",
                    is_primary=True,
                ),
                *(
                    DataProdAssoc(
                        uid=f"tcs-{n}",
                        data_prod_assoc_type="dpa_reduction_obs",
                        context={"tag": "reduction_obs", "roles": ["science"]},
                    )
                    for n in numbers
                ),
            ],
        ),
        products=[group],
        run=run,
    )
    with store.materialize(
        description, run_name=CounterRunNameGenerator(key="coadd")
    ) as index:
        payload = index.base_path / "selection.json"
        payload.write_text(
            json.dumps(
                {"observations": list(numbers), "numerical_coadd": False}, indent=2
            )
            + "\n"
        )
        index.locations["self"] = Location(root=".", location_type="local")
        index.product.data_items.append(
            DataProdItem(
                meta={"type": "artifact", "name": "selection", "format": "json"},
                source=DataProdSource.model_validate(
                    {"location": "self", "path": payload.name}
                ),
            )
        )
    return index


def build_example(root):
    """Create a new store and return its exact published result indexes."""
    store = DataProdStore(root / "dataprod_toltec", namespace="commissioning")
    for number in (149101, 149102, 149103):
        obs = DataProdIndex(
            product=DataProd(
                uid=f"tcs-{number}",
                meta=ObsMeta(
                    name=str(number),
                    data_prod_type="dp_obs",
                    master="tcs",
                    obsnum=number,
                ),
            )
        )
        store.write_index(obs, primary_uid=obs.product.uid)
    group = collection((149101, 149102))
    store.write_index(group, primary_uid=group.product.uid, primary_path="orion")
    with WorkflowRun.create(context={"observations": [149101, 149102]}) as first:
        original = publish(store, group, first, (149101, 149102))
        subset = publish(store, group, first, (149101,), uid=f"{GROUP_UID}_subset")
    store.write_run(first)
    group = collection((149101, 149102, 149103))
    store.write_index(group, primary_uid=group.product.uid, overwrite=True)
    with WorkflowRun.create(
        context={"observations": [149101, 149102, 149103]}
    ) as second:
        refreshed = publish(store, group, second, (149101, 149102, 149103))
    store.write_run(second)
    # Select the earlier full result; the independent subset keeps its own main.
    store.set_tag(
        group.product.uid,
        "dp_science",
        original.store_run.name,
        uid=f"{GROUP_UID}_science",
    )
    return original, subset, refreshed


if __name__ == "__main__":
    import sys

    build_example(Path(sys.argv[1]))
