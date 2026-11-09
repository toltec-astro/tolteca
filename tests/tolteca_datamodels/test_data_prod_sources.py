"""Portable item locations, literal paths and document-relative resolution."""

from datetime import UTC, datetime
from pathlib import PurePosixPath

import pytest
from pydantic import BaseModel, ValidationError
from tollan.utils.fileloc import FileLoc, fileloc

from tolteca_datamodels.core.data_prod import (
    DataProdBase,
    DataProdIndexBase,
    DataProdItem,
    DataProdSource,
    Location,
)


class MeasurementMeta(BaseModel):
    """Metadata for a product outside the observation product vocabulary."""

    quantity: str
    unit: str


class MeasurementRecord(DataProdBase[MeasurementMeta]):
    """Portable measurement product."""


class MeasurementIndex(DataProdIndexBase[MeasurementRecord]):
    """Portable index for a measurement product."""


@pytest.mark.parametrize(
    ("root", "expected"),
    [
        ("/archive/scan data", "file:///archive/scan%20data"),
        ("file:///archive/scan%20data", "file:///archive/scan%20data"),
        ("archive:/data", "file://archive/data"),
        ("https://archive.example/data", "https://archive.example/data"),
    ],
)
def test_location_resolution_returns_fileloc(root, expected):
    index = MeasurementIndex(
        product=MeasurementRecord(
            uid="measurement:1",
            meta=MeasurementMeta(quantity="temperature", unit="K"),
        ),
        locations={"archive": Location(root=root)},
    )
    resolved = index.resolve_location("archive")
    assert isinstance(resolved, FileLoc)
    assert resolved.url == expected
    assert index.locations["archive"].root == root


@pytest.mark.parametrize("root", [".", "..", "arrays"])
def test_relative_root_requires_document_directory_not_config_context(root, tmp_path):
    location = Location.model_validate(
        {"root": root}, context={"root_path": tmp_path / "configuration"}
    )
    with pytest.raises(ValueError, match="requires an index directory"):
        location.resolve()
    for dirname in ("original", "relocated"):
        directory = tmp_path / dirname / "products"
        assert location.resolve(directory).url == (directory / root).resolve().as_uri()
        assert location.root == root


def test_location_resolution_validates_external_root():
    location = Location(root="unsupported://archive/data")
    with pytest.raises(ValidationError):
        location.resolve()


def test_extension_preserves_typed_metadata_and_nested_sources(tmp_path):
    source = DataProdItem(
        source=DataProdSource.model_validate(
            {"path": "values.zarr", "location": "payload"}
        )
    )
    child = MeasurementIndex(
        product=MeasurementRecord(
            uid="measurement:1",
            meta=MeasurementMeta(quantity="temperature", unit="K"),
            data_items=[source],
        ),
        locations={"payload": Location(root="arrays")},
    )
    index = MeasurementIndex(
        product=MeasurementRecord(
            uid="measurements", meta=MeasurementMeta(quantity="temperature", unit="K")
        ),
        products=[child],
    )
    restored = MeasurementIndex.read(index.write(tmp_path / "index.yaml"))
    [child] = restored.products
    assert isinstance(child, MeasurementIndex)
    assert isinstance(child.product.meta, MeasurementMeta)
    assert child.product.meta.unit == "K"
    assert (
        child.get_source_uri(child.product.data_items[0].source)
        == (tmp_path / "arrays/values.zarr").as_uri()
    )
    assert not (tmp_path / "arrays").exists()


def test_relative_index_can_be_relocated_without_rewriting_references(tmp_path):
    from shutil import copytree

    source = tmp_path / "first"
    source.mkdir()
    (source / "values.txt").write_text("measurement")
    index = MeasurementIndex(
        product=MeasurementRecord(
            uid="measurement:1",
            meta=MeasurementMeta(quantity="temperature", unit="K"),
            data_items=[
                DataProdItem(
                    source=DataProdSource.model_validate(
                        {"path": "values.txt", "location": "self"}
                    )
                )
            ],
        ),
        locations={"self": Location(root=".")},
    )
    path = index.write(source)
    opened = MeasurementIndex.read(path)
    assert opened.model_dump() == index.model_dump()
    copied = MeasurementIndex.read(opened.write(source / "copy.yaml"))
    assert copied.model_dump(round_trip=True) == opened.model_dump(round_trip=True)
    target = tmp_path / "moved"
    copytree(source, target)
    relocated = MeasurementIndex.read(target)
    assert relocated.locations["self"].root == "."
    assert (
        relocated.get_source_uri(relocated.product.data_items[0].source)
        == (target / "values.txt").as_uri()
    )
    assert (
        opened.get_source_uri(opened.product.data_items[0].source)
        == (source / "values.txt").as_uri()
    )


@pytest.mark.parametrize("path", ["../outside.nc", "/absolute.nc", "."])
def test_item_path_uses_subpath_validation(path):
    with pytest.raises(ValidationError):
        DataProdSource.model_validate({"path": path, "location": "self"})


def test_item_path_is_literal_and_survives_yaml_relocation(tmp_path):
    import shutil

    original = tmp_path / "original"
    original.mkdir()
    name = "data/scan 50%23#?.nc"
    payload = original / name
    payload.parent.mkdir()
    payload.write_bytes(b"payload")
    source = DataProdSource.model_validate({"path": name, "location": "self"})
    index = MeasurementIndex(
        product=MeasurementRecord(
            uid="measurement:1",
            meta=MeasurementMeta(quantity="temperature", unit="K"),
            data_items=[DataProdItem(source=source)],
        ),
        locations={"self": Location(root=".")},
    )
    index.write(original)
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    restored = MeasurementIndex.read(moved)
    [item] = restored.product.data_items
    assert item.source.path == PurePosixPath(name)
    assert item.source.model_dump(mode="json", round_trip=True)["path"] == name
    assert restored.get_source_uri(item.source) == (moved / name).as_uri()
    assert fileloc(restored.get_source_uri(item.source)).path.read_bytes() == b"payload"
    # Remote location roots use the same literal path convention.
    restored.locations["self"] = Location(root="https://archive.example/data")
    assert (
        restored.get_source_uri(item.source)
        == "https://archive.example/data/data/scan%2050%2523%23%3F.nc"
    )


def test_source_timestamp_loads_from_dump_without_coercing_other_values():
    source = DataProdItem(
        source=DataProdSource.model_validate(
            {
                "path": "index.yaml",
                "location": "site",
                "size": 1,
                "last_verified_at": datetime(2025, 10, 31, tzinfo=UTC),
            }
        )
    )
    data = source.source.model_dump(mode="json")
    assert isinstance(data["last_verified_at"], str)
    assert DataProdSource.model_validate(data) == source.source
    with pytest.raises(ValidationError, match="size"):
        DataProdSource.model_validate(data | {"size": "1"})
