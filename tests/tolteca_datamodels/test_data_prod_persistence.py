"""YAML persistence preserves typed metadata and its validation contract."""

from datetime import UTC, date, datetime
from uuid import UUID

import pytest
import yaml
from pydantic import ConfigDict, Json, ValidationError, computed_field
from yaml.constructor import ConstructorError

from tolteca_datamodels.core.data_prod import (
    Actor,
    DataProdAudit,
    DataProdBase,
    DataProdIndexBase,
)
from tolteca_datamodels.core.types import FrozenDataModel


class MeasurementMeta(FrozenDataModel):
    """Strict metadata with nullable values and serialized field types."""

    model_config = ConfigDict(extra="forbid")

    uid: UUID
    acquired: date
    labels: frozenset[str]
    count: int
    uncertainty: float | None
    reference: float | None = 7.0
    settings: Json[dict[str, int]]

    @computed_field
    @property
    def doubled_count(self) -> int:
        """Derived count, reconstructed from declared metadata."""
        return self.count * 2


class MeasurementRecord(DataProdBase[MeasurementMeta]):
    """Product with extension-owned metadata."""


class MeasurementIndex(DataProdIndexBase[MeasurementRecord]):
    """Portable measurement document."""


@pytest.fixture
def measurement_index():
    """A complete product whose explicit null differs from its default."""
    return MeasurementIndex(
        product=MeasurementRecord(
            uid="measurement:1",
            meta=MeasurementMeta.model_validate(
                {
                    "uid": UUID(int=1),
                    "acquired": date(2026, 11, 6),
                    "labels": frozenset({"calibration"}),
                    "count": 2,
                    "uncertainty": None,
                    "reference": None,
                    "settings": '{"order": 1}',
                }
            ),
        ),
    )


def test_yaml_round_trip_preserves_typed_metadata(tmp_path, measurement_index):
    child = measurement_index.model_copy(deep=True)
    child.product.uid = "measurement:2"
    measurement_index.products.append(child)

    restored = MeasurementIndex.read(measurement_index.write(tmp_path))

    assert restored.product == measurement_index.product
    assert restored.products[0].product == child.product
    assert isinstance(restored.products[0], MeasurementIndex)
    assert restored.product.meta.uncertainty is None
    assert restored.product.meta.reference is None
    assert restored.product.meta.doubled_count == 4


def test_yaml_read_keeps_strict_metadata_validation(tmp_path, measurement_index):
    path = measurement_index.write(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["product"]["meta"]["count"] = "2"
    path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValidationError) as exc_info:
        MeasurementIndex.read(path)

    assert [(error["loc"], error["type"]) for error in exc_info.value.errors()] == [
        (("product", "meta", "count"), "int_type"),
    ]


@pytest.mark.parametrize(
    ("creator", "modifier"),
    [
        (UUID(int=1), UUID(int=2)),
        (None, UUID(int=2)),
        (UUID(int=1), None),
        (None, None),
    ],
    ids=["both-known", "modifier-only", "creator-only", "timestamps-only"],
)
def test_yaml_preserves_independently_known_actors(
    tmp_path, measurement_index, creator, modifier
):
    audit = DataProdAudit(
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        created_by=creator,
        updated_at=datetime(2026, 11, 9, tzinfo=UTC),
        updated_by=modifier,
    )
    measurement_index.product.audit = audit
    measurement_index.actors = {
        uid: Actor(uid=uid, name=f"actor-{uid.int}", created_at=audit.created_at)
        for uid in (creator, modifier)
        if uid is not None
    }

    restored = MeasurementIndex.read(measurement_index.write(tmp_path))

    assert restored.product.audit == audit
    assert restored.actors == measurement_index.actors


def test_index_read_rejects_python_yaml_tags(tmp_path):
    path = tmp_path / "index.yaml"
    path.write_text("product: !!python/tuple [0, 1]\n")
    with pytest.raises(ConstructorError):
        MeasurementIndex.read(path)
