"""Portable product records, source locations and YAML indexes.

Specialize ``DataProdBase`` with a metadata type and ``DataProdIndexBase``
with that product type. ``read`` and ``write`` preserve nested product types and
relative source references. Use ``get_source_uri`` to resolve an artifact against
the opened index directory without opening the artifact.
"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)
from pydantic_core import to_json
from tollan.config.types import PathSegment, SubPath
from tollan.utils.fileloc import FileLoc, fileloc

from ..types import FrozenDataModel
from ._io import _read_yaml, _write_yaml
from .actor import _ActorsByUid
from .namespace import DataProdAudit, DataProdNamespace
from .run import RunUid, StoreRun, WorkflowRun

__all__ = [
    "DataProdAssoc",
    "DataProdAssocRun",
    "DataProdBase",
    "DataProdIndexBase",
    "DataProdItem",
    "DataProdRef",
    "DataProdSource",
    "DataProdTag",
    "Location",
]


def _index_location_label(root_uri):
    """Bound generated location labels independently of filesystem depth."""
    return f"index:{sha256(root_uri.encode()).hexdigest()}"


class Location(BaseModel):
    """Storage root for resolving item paths."""

    # Product-index entries remain editable before publication.
    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    audit: DataProdAudit | None = None

    root: str = Field(
        description=(
            "Storage root, relative to the index directory or an external location."
        )
    )
    location_type: str | None = Field(
        default=None, description="Storage backend label, when known."
    )

    @property
    def is_relative(self) -> bool:
        """Whether the root requires an index directory for resolution."""
        return not (urlsplit(self.root).scheme or Path(self.root).is_absolute())

    def resolve(self, base_path: Path | None = None) -> FileLoc:
        """Resolve the root without changing its stored declaration.

        Parameters
        ----------
        base_path : pathlib.Path or None, optional
            Index directory. Required for a relative root.

        Returns
        -------
        tollan.utils.fileloc.FileLoc
            Validated, absolute location. No resource is opened.

        Raises
        ------
        ValueError
            A relative root has no index directory, or FileLoc rejects the root.
        """
        if self.is_relative and base_path is None:
            msg = f"Artifact location requires an index directory: {self.root!r}"
            raise ValueError(msg)
        return fileloc(self.root, local_parent_path=base_path)


class DataProdSource(BaseModel):
    """Location and storage facts for a data item."""

    # Product-index entries remain editable before publication.
    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    path: SubPath = Field(description="Item path relative to the location root.")
    location: str = Field(description="Key in the index's locations mapping.")
    role: str = Field(default="primary", description="Storage role of this source.")
    availability_state: str | None = Field(
        default=None, description="Last known availability."
    )
    size: int | None = Field(
        default=None, ge=0, description="Size in bytes, when known."
    )
    checksum: str | None = Field(default=None, description="Recorded content checksum.")
    last_verified_at: datetime | None = Field(
        # YAML indexes also carry ISO-formatted timestamps from JSON-mode dumps.
        default=None,
        strict=False,
        description="Last verification time.",
    )


class DataProdItem(BaseModel):
    """One constituent of a product, with its metadata and source reference.

    Items may describe acquisition files, calibration inputs or generated artifacts.
    Different products may reference the same source with different item metadata.
    """

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    meta: dict[str, Any] = Field(default_factory=dict)
    source: DataProdSource
    audit: DataProdAudit | None = None


class DataProdRef(FrozenDataModel):
    """Portable product identity, with a separate namespace for scoped products."""

    uid: str = Field(min_length=1, description="Referenced product's identifier.")
    namespace_uid: UUID | None = Field(default=None, strict=False)


class DataProdTag(BaseModel):
    """An editable version-group selection of a member and its namespace.

    Set ``current`` to None for an empty selection.
    """

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)
    current: DataProdRef | None = Field(
        default=None, description="Selected member, or None for an empty tag."
    )
    audit: DataProdAudit | None = None


class DataProdAssocRun(FrozenDataModel):
    """Execution facts for one invocation and one product association."""

    context: dict[str, Any] | None = None
    created_at: AwareDatetime | None = Field(default=None, strict=False)
    created_by: UUID | None = Field(default=None, strict=False)


class DataProdAssoc(BaseModel):
    """A directed relationship from this product to another product."""

    # Associations are editable while assembling a product's relationship graph.
    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    data_prod_assoc_type: str = Field(
        description="Application-defined relationship label."
    )
    uid: str = Field(
        min_length=1,
        description="Related product's identifier, independent of storage.",
    )
    is_primary: bool = Field(
        default=False,
        description="Follow this association when resolving the product's primary.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Relationship metadata."
    )
    namespace_uid: UUID | None = Field(
        default=None,
        strict=False,
        description="Target namespace; None for a global singleton.",
    )
    runs: dict[RunUid, DataProdAssocRun] = Field(
        default_factory=dict, description="Invocations establishing this association."
    )
    audit: DataProdAudit | None = None


class DataProdBase[MetaT](BaseModel):
    """Product identity, metadata, artifact references and relationships.

    Specialize ``MetaT`` with the application's metadata type. A record identifies
    a product independently of the document or database that stores it.
    """

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)
    uid: str = Field(min_length=1, description="Data-product identifier.")
    namespace_uid: UUID | None = Field(default=None, strict=False)
    meta: MetaT = Field(description="Application-defined product metadata.")
    data_items: list[DataProdItem] = Field(default_factory=list)
    assocs: list[DataProdAssoc] = Field(default_factory=list)
    tags: dict[PathSegment, DataProdTag] = Field(default_factory=dict)
    audit: DataProdAudit | None = None

    @property
    def reference(self) -> DataProdRef:
        """Portable identity used by associations and tags."""
        return DataProdRef(uid=self.uid, namespace_uid=self.namespace_uid)


class DataProdIndexBase[ProductT: DataProdBase](BaseModel):
    """Portable product document with its locations and supporting records.

    Specialize ``ProductT`` with a product record type. Nested ``products`` retain
    the same document type. Relative locations use the directory passed to
    ``read``; serialization preserves those relative references.

    Each item must reference a declared location, and its location/path pair must
    be unique within the product. An included namespace must match the product's
    namespace UUID. An included store run must reference the included execution.
    Actor mapping keys must match the contained actors' UUIDs.
    """

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)
    format_version: Literal[1] = 1
    product: ProductT
    locations: dict[str, Location] = Field(default_factory=dict)
    namespace: DataProdNamespace | None = None
    actors: _ActorsByUid = Field(default_factory=dict)
    run: WorkflowRun | None = None
    store_run: StoreRun | None = None
    provenance: dict[str, Any] | None = None
    products: list[Self] = Field(default_factory=list)
    _base_path: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        sources = set()
        for item in self.product.data_items:
            source = item.source
            if source.location not in self.locations:
                msg = f"Undefined storage location {source.location!r}"
                raise ValueError(msg)
            key = (source.location, source.path)
            if key in sources:
                msg = f"Source is listed more than once: {key!r}"
                raise ValueError(msg)
            sources.add(key)
        if (
            self.namespace is not None
            and self.product.namespace_uid != self.namespace.uid
        ):
            msg = "Product namespace differs from the included namespace"
            raise ValueError(msg)
        if self.store_run is not None and (
            self.run is None or self.store_run.run_uid != self.run.uid
        ):
            msg_0 = "Store run must reference the included execution"
            raise ValueError(msg_0)
        return self

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read a YAML index without opening its payloads.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML file or directory containing ``index.yaml``.

        Returns
        -------
        Self
            Validated document with relative locations bound to its directory.
            Metadata is restored through its declared field types. Serialized
            references are unchanged.

        Raises
        ------
        pydantic.ValidationError
            The document does not satisfy the index or metadata schema.

        Notes
        -----
        Invalid YAML and unsupported tags, including Python-specific object tags,
        raise ``yaml.YAMLError``.
        """
        path = Path(path)
        if path.is_dir():
            path /= "index.yaml"
        index = cls.model_validate_json(to_json(_read_yaml(path)))
        index._bind_locations(path.parent.resolve())
        return index

    @property
    def base_path(self) -> Path | None:
        """Directory resolving relative locations, or None before binding."""
        return self._base_path

    def _bind_locations(self, base_path: Path | None) -> None:
        self._base_path = base_path
        for child in self.products:
            child._bind_locations(base_path)

    def resolve_location(self, key: str) -> FileLoc:
        """Resolve a location against this document's directory.

        Parameters
        ----------
        key : str
            Key in ``locations``.

        Returns
        -------
        tollan.utils.fileloc.FileLoc
            Validated, absolute location. The stored root remains unchanged.

        Raises
        ------
        ValueError
            A relative location has no document directory.
        """
        return self.locations[key].resolve(self._base_path)

    def get_source_uri(self, source: DataProdSource) -> str:
        """Resolve an artifact reference without changing the document.

        Parameters
        ----------
        source : DataProdSource
            Artifact reference from this product.

        Returns
        -------
        str
            Absolute URI, with literal item path characters URL-encoded.
        """
        return self.resolve_location(source.location).joinpath(source.path).url

    def write(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write declared fields, explicit nulls and relative references to YAML.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML filename or directory for ``index.yaml``.
        overwrite : bool, optional
            Replace an existing index atomically; defaults to False.

        Returns
        -------
        pathlib.Path
            Written filename. Artifact payloads are not copied.

        Notes
        -----
        Metadata uses Pydantic's round-trip representation. Computed fields are
        reconstructed when the document is read, rather than saved as input fields.
        YAML formatting and comments are not preserved.
        """
        path = Path(path)
        if path.is_dir() or path.suffix not in (".yaml", ".yml"):
            path /= "index.yaml"
        return _write_yaml(self, path, overwrite=overwrite)
