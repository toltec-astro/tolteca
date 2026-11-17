"""Inventory envelopes retain reporting identity and validate their entries."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from tolteca_datamodels.data_prod import DataProdInventory, DataProdInventoryIndex


def test_empty_inventory_contribution():
    inventory = DataProdInventory(uid=uuid4(), name="archive")

    contribution = DataProdInventoryIndex(inventory=inventory)

    assert contribution.inventory == inventory
    assert contribution.entries == []


def test_inventory_rejects_malformed_entry():
    with pytest.raises(ValidationError):
        DataProdInventoryIndex.model_validate(
            {
                "inventory": DataProdInventory(uid=uuid4()),
                "entries": [{"unrecognized": "record"}],
            }
        )
