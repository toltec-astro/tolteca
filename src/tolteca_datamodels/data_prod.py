"""Portable data-product indexes.

Read an index from a YAML file or a product directory::

    index = DataProdIndex.read("dataprod_toltec/scratch/145609-0-1/dp_raw_obs.yaml")
    sources = index.data_items

``meta`` describes the product; ``data_items`` lists its source references and
their metadata. ``assocs`` records related products by identifier. Index paths,
storage locations and namespace directory names do not change product identity.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, cast
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator
from tollan.utils.yaml import yaml_dump, yaml_load

from .data_prod_namespace import (
    DataProdActor,
    DataProdAudit,
    DataProdNamespace,
)
from .data_prod_run import (
    DataProdRun,
    PathComponent,
    RunUid,
)
from .lmt_base.funcs import make_uid_obs, make_uid_raw_obs
from .types import FrozenDataModel

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DataProdIndex",
    "DataProdIndexAssoc",
    "DataProdIndexAssocRun",
    "DataProdIndexCalRef",
    "DataProdIndexLocation",
    "DataProdIndexSource",
    "DataProdRef",
    "DataProdTag",
    "RawObsFileCalObsFileMeta",
    "make_obs_product_uid",
    "make_raw_obs_product_uid",
    "normalize_primary_uid",
]


def normalize_primary_uid(primary_uid: str) -> str:
    """Return the canonical identifier of a primary observation or acquisition.

    Parameters
    ----------
    primary_uid : str
        Explicit master followed by an observation number, or by observation,
        subobservation and scan numbers. For example, ``TCS-00145609-000-0001``.

    Returns
    -------
    str
        Lowercase master and unpadded numbers, such as ``tcs-145609-0-1``.
        The master remains part of the identifier, including for TCS.

    Raises
    ------
    ValueError
        The value does not identify one observation or acquisition with an
        explicit master. Directory labels with an omitted master are not UIDs.

    Examples
    --------
    >>> normalize_primary_uid("TCS-00145609-000-0001")
    'tcs-145609-0-1'
    >>> normalize_primary_uid("ICS-00001")
    'ics-1'
    """
    if (
        re.fullmatch(r"[A-Za-z][A-Za-z0-9]*-[0-9]+(?:-[0-9]+-[0-9]+)?", primary_uid)
        is None
    ):
        msg = (
            "Primary entity UID requires an explicit master and one or three "
            "observation numbers"
        )
        raise ValueError(msg)
    master, *numbers = primary_uid.split("-")
    return "-".join((master.lower(), *(str(int(number)) for number in numbers)))


def _resolve_primary(product, get_next):
    """Follow primary links, detecting cycles by product UID."""
    visited = set()
    while True:
        if not product.uid:
            msg = "Primary resolution requires product UIDs"
            raise ValueError(msg)
        namespace = getattr(product, "namespace", None)
        identity = (None if namespace is None else str(namespace.uid), product.uid)
        if identity in visited:
            msg = f"Primary association cycle at {product.uid!r}"
            raise ValueError(msg)
        visited.add(identity)
        target = get_next(product)
        if target is None:
            return product
        product = target


def _select_primary_assoc(product_uid, assocs):
    """Select the single marked association of a nonterminal product."""
    primary = [assoc for assoc in assocs if assoc.is_primary]
    if len(primary) != 1:
        msg = f"Product {product_uid!r} requires exactly one primary association"
        raise ValueError(msg)
    return primary[0]


def _validate_observation_identity(master, *numbers):
    """Require a master label and nonnegative integer observation numbers."""
    if (
        not isinstance(master, str)
        or not master.isalnum()
        or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in numbers)
    ):
        msg = "An observation requires a master and nonnegative observation numbers"
        raise ValueError(msg)


def make_obs_product_uid(master: str, obsnum: int) -> str:
    """Return the location-independent identifier of an observation product.

    Parameters
    ----------
    master : str
        Acquisition master, such as ``tcs`` or ``ics``.
    obsnum : int
        Observation number.

    Returns
    -------
    str
        Lowercase master followed by the unpadded observation number.

    Raises
    ------
    ValueError
        The master is not an alphanumeric label or the observation number is
        not a nonnegative integer.
    """
    _validate_observation_identity(master, obsnum)
    return cast("str", make_uid_obs(master.lower(), obsnum))


def make_raw_obs_product_uid(
    master: str, obsnum: int, subobsnum: int, scannum: int
) -> str:
    """Return the location-independent identifier of a raw-observation product.

    Parameters
    ----------
    master : str
        Acquisition master, such as ``tcs`` or ``ics``.
    obsnum, subobsnum, scannum : int
        Observation, subobservation and scan numbers.

    Returns
    -------
    str
        Acquisition quartet with a lowercase master.

    Raises
    ------
    ValueError
        The master is not an alphanumeric label or an observation number is
        not a nonnegative integer.
    """
    _validate_observation_identity(master, obsnum, subobsnum, scannum)
    return cast("str", make_uid_raw_obs(master.lower(), obsnum, subobsnum, scannum))


class DataProdIndexLocation(FrozenDataModel):
    """Storage root used by source URIs in a product index."""

    # Product-index entries remain editable before publication.
    model_config = ConfigDict(extra="forbid", frozen=False)

    audit: DataProdAudit | None = None

    root_uri: str = Field(description="Storage root for relative source URIs.")
    location_type: str | None = Field(
        default=None, description="Storage backend label, when known."
    )


class DataProdIndexSource(FrozenDataModel):
    """A source reference and its metadata in a product index."""

    # Product-index entries remain editable before publication.
    model_config = ConfigDict(extra="forbid", frozen=False)

    audit: DataProdAudit | None = None

    source_uri: str = Field(
        description="Absolute URI or path relative to the location root."
    )
    location: str = Field(description="Key in the index's locations mapping.")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Metadata describing this source."
    )
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


class DataProdRef(FrozenDataModel):
    """Portable product identity, with a separate namespace for scoped products."""

    uid: str
    namespace_uid: UUID | None = Field(default=None, strict=False)


class DataProdTag(FrozenDataModel):
    """A version-group selection identifying a member and its namespace."""

    model_config = ConfigDict(extra="forbid")
    current: DataProdRef | None = Field(
        default=None, description="Selected member, or None for an empty tag."
    )
    audit: DataProdAudit | None = None


class DataProdIndexAssocRun(FrozenDataModel):
    """Execution facts for one invocation and one product association."""

    context: dict[str, Any] | None = None
    created_at: AwareDatetime | None = Field(default=None, strict=False)
    created_by: UUID | None = Field(default=None, strict=False)


class DataProdIndexAssoc(FrozenDataModel):
    """A directed relationship from this product to another product."""

    # Associations are editable while assembling a product's relationship graph.
    model_config = ConfigDict(extra="forbid", frozen=False)

    data_prod_assoc_type: str = Field(
        description="Relationship type registered in DPDB."
    )
    uid: str = Field(
        description="Related product's identifier, independent of storage."
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
    runs: dict[RunUid, DataProdIndexAssocRun] = Field(
        default_factory=dict, description="Invocations establishing this association."
    )
    audit: DataProdAudit | None = None


class RawObsFileCalObsFileMeta(FrozenDataModel):
    """Known calibration quartet for one acquisition interface.

    Missing components remain None until acquisition metadata supplies them.
    A VNA uses its own quartet, regardless of older target-sweep header values.
    """

    model_config = ConfigDict(extra="forbid")
    cal_master: Annotated[str, Field(pattern="^[a-z][a-z0-9]*$")] | None = None
    cal_obsnum: Annotated[int, Field(ge=0)] | None = None
    cal_subobsnum: Annotated[int, Field(ge=0)] | None = None
    cal_scannum: Annotated[int, Field(ge=0)] | None = None

    @property
    def uid(self) -> str | None:
        """Calibration raw-observation UID, or None for an incomplete quartet."""
        values = (
            self.cal_master,
            self.cal_obsnum,
            self.cal_subobsnum,
            self.cal_scannum,
        )
        return (
            None
            if any(value is None for value in values)
            else make_uid_raw_obs(*values)
        )


class DataProdIndexCalRef(FrozenDataModel):
    """Portable calibration reference for one logical interface file."""

    model_config = ConfigDict(extra="forbid")
    interface: str = Field(min_length=1)
    meta: RawObsFileCalObsFileMeta
    target_uid: str | None = Field(
        default=None, description="Resolved raw product; None while unresolved."
    )
    audit: DataProdAudit | None = None

    @model_validator(mode="after")
    def _validate_target(self) -> Self:
        if self.target_uid is not None and self.target_uid != self.meta.uid:
            msg = "Calibration target UID must match the complete reference quartet"
            raise ValueError(msg)
        return self


class DataProdIndex(FrozenDataModel):
    """Inventory of a data product, its sources and related products.

    Observations have one identifier per master and obsnum; raw observations
    have one per acquisition quartet. Other producers supply their own identifiers
    and metadata, including explicit associations to their contributing products.
    """

    # DB ingestion assembles and replaces associations on an existing index.
    model_config = ConfigDict(extra="forbid", frozen=False)

    format_version: Literal[1] = Field(
        default=1, description="Portable index format version."
    )
    uid: str = Field(min_length=1, description="Data-product identifier.")
    meta: dict[str, Any] = Field(
        description="Product metadata, including data_prod_type and name."
    )
    data_items: list[DataProdIndexSource] = Field(
        default_factory=list,
        description="Files, indexes or directories with source metadata.",
    )
    locations: dict[str, DataProdIndexLocation] = Field(
        default_factory=dict,
        description="Storage roots referenced by data_items.",
    )
    assocs: list[DataProdIndexAssoc] = Field(
        default_factory=list, description="Directed relationships to other products."
    )
    calibration_refs: list[DataProdIndexCalRef] = Field(
        default_factory=list,
        description="Per-interface calibration lineage of a raw observation.",
    )
    run: DataProdRun | None = Field(
        default=None, description="Pinned run; None for a product inventory."
    )
    provenance: dict[str, Any] | None = Field(
        default=None,
        description="Producer-defined run provenance, validated by its domain model.",
    )
    namespace: DataProdNamespace | None = Field(
        default=None,
        description="Owning namespace; None for singletons and unpublished proposals.",
    )
    actors: dict[Annotated[UUID, Field(strict=False)], DataProdActor] = Field(
        default_factory=dict, description="Actors referenced by this document."
    )
    audit: DataProdAudit | None = None
    products: list[DataProdIndex] = Field(
        default_factory=list,
        description="Associated product records included in a version-group inventory.",
    )
    tags: dict[PathComponent, DataProdTag] = Field(
        default_factory=dict, description="Current version selections of this group."
    )

    @property
    def reference(self) -> DataProdRef:
        """Portable product identity for associated-product lookups."""
        return DataProdRef(
            uid=self.uid,
            namespace_uid=None if self.namespace is None else self.namespace.uid,
        )

    def get_primary(
        self, lookup: Callable[[DataProdRef], DataProdIndex] | None = None
    ) -> DataProdIndex:
        """Resolve the primary raw observation or observation.

        Parameters
        ----------
        lookup : callable or None, optional
            Retrieve an associated index by its DataProdRef identity. For an
            in-memory collection, pass ``products.__getitem__``. Required for
            derived products; logical ``dp_raw_obs`` and ``dp_obs`` return self.

        Returns
        -------
        DataProdIndex
            Terminal logical product reached through ``is_primary`` associations.

        Raises
        ------
        ValueError
            A nonterminal product has no unique primary link, lookup is missing,
            a lookup returns a different UID, or the links form a cycle.
        KeyError
            The lookup cannot find an associated product.
        """

        def get_next(product):
            if product.meta["data_prod_type"] in {"dp_raw_obs", "dp_obs"}:
                return None
            assoc = _select_primary_assoc(product.uid, product.assocs)
            if lookup is None:
                msg = "Primary resolution requires an associated-product lookup"
                raise ValueError(msg)
            reference = DataProdRef(uid=assoc.uid, namespace_uid=assoc.namespace_uid)
            target = lookup(reference)
            if target.reference != reference:
                msg = f"Primary lookup returned a different product for {assoc.uid!r}"
                raise ValueError(msg)
            return target

        return _resolve_primary(self, get_next)

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        if not self.meta.get("data_prod_type") or not self.meta.get("name"):
            msg = "Product metadata requires data_prod_type and name"
            raise ValueError(msg)
        if self.meta["data_prod_type"] in {"dp_obs", "dp_raw_obs"}:
            # ty does not yet honor frozen=False on this inherited model field.
            self.uid = normalize_primary_uid(self.uid)  # ty: ignore[invalid-assignment]
        if self.meta["data_prod_type"] == "dp_obs":
            keys = ("master", "obsnum")
            if any(key not in self.meta for key in keys):
                msg = "Observation metadata requires master and obsnum"
                raise ValueError(msg)
            expected = make_obs_product_uid(**{key: self.meta[key] for key in keys})
            if self.uid != expected:
                msg = f"Observation UID must be {expected!r}"
                raise ValueError(msg)
        if self.meta["data_prod_type"] == "dp_raw_obs":
            keys = ("master", "obsnum", "subobsnum", "scannum")
            if any(key not in self.meta for key in keys):
                msg = "Raw-observation metadata requires the complete quartet"
                raise ValueError(msg)
            expected = make_raw_obs_product_uid(**{key: self.meta[key] for key in keys})
            if self.uid != expected:
                msg = f"Raw-observation UID must be {expected!r}"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        if self.calibration_refs:
            if self.meta["data_prod_type"] != "dp_raw_obs":
                msg = "Only raw observations carry interface calibration references"
                raise ValueError(msg)
            interfaces = [entry.interface for entry in self.calibration_refs]
            if len(set(interfaces)) != len(interfaces):
                msg = "Calibration references require distinct interfaces"
                raise ValueError(msg)
        sources = set()
        for source in self.data_items:
            if source.location not in self.locations:
                msg = f"Undefined storage location {source.location!r}"
                raise ValueError(msg)
            key = (source.location, source.source_uri)
            if key in sources:
                msg = f"Source is listed more than once: {key!r}"
                raise ValueError(msg)
            sources.add(key)
        return self

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read a YAML index without opening its payloads.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML file or existing directory containing ``index.yaml``.

        Returns
        -------
        DataProdIndex
            Validated inventory. Relative storage roots are resolved against the
            index directory and keyed by their absolute URI. Source paths remain
            relative to those roots.

        Raises
        ------
        OSError
            The index cannot be read.
        pydantic.ValidationError
            The index does not satisfy the product-index contract.
        """
        path = Path(path)
        if path.is_dir():
            path /= "index.yaml"
        index = cls.model_validate(yaml_load(path))
        return index._resolve_locations(path.parent)

    def _resolve_locations(self, parent):
        for key, location in list(self.locations.items()):
            if (
                urlsplit(location.root_uri).scheme
                or Path(location.root_uri).is_absolute()
            ):
                continue
            root = (parent / location.root_uri).resolve().as_uri()
            self.locations[root] = location.model_copy(update={"root_uri": root})
            del self.locations[key]
            self.data_items[:] = [
                item.model_copy(update={"location": root})
                if item.location == key
                else item
                for item in self.data_items
            ]
        self.products[:] = [child._resolve_locations(parent) for child in self.products]
        return self

    def get_source_uri(self, source: DataProdIndexSource) -> str:
        """Resolve an artifact reference against its registered location.

        Parameters
        ----------
        source : DataProdIndexSource
            Source from this product. Its location root must be absolute;
            `read` resolves relative roots for filesystem indexes.

        Returns
        -------
        str
            Absolute URI or local path. Absolute source references are unchanged.

        Raises
        ------
        ValueError
            The location root is relative and has no filesystem index base.
        """
        uri = source.source_uri
        if urlsplit(uri).scheme or Path(uri).is_absolute():
            return uri
        root = self.locations[source.location].root_uri
        if not (urlsplit(root).scheme or Path(root).is_absolute()):
            msg = f"Artifact location requires an absolute root: {root!r}"
            raise ValueError(msg)
        return f"{root.rstrip('/')}/{uri}"

    def write(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write the index to a YAML file or product directory.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination YAML filename, or directory for ``index.yaml``.
            Missing parent directories are created.
        overwrite : bool, optional
            Replace an existing index atomically; defaults to False.

        Returns
        -------
        pathlib.Path
            Written YAML filename. Payload files are not copied or exported.

        Raises
        ------
        FileExistsError
            The index exists and overwrite is False.
        """
        path = Path(path)
        if path.is_dir() or path.suffix not in (".yaml", ".yml"):
            path /= "index.yaml"
        return _write_yaml(self, path, overwrite=overwrite)


def _write_yaml(record, path, *, overwrite=False):
    contents = yaml_dump(record.model_dump(mode="json", exclude_none=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".yaml", delete=False
    ) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
            if overwrite:
                tmp.replace(path)
            else:
                # Exclusive publication without exposing an incomplete YAML.
                path.hardlink_to(tmp)
        finally:
            tmp.unlink(missing_ok=True)
    _sync_directory(path.parent)
    return path


def _sync_directory(path: Path) -> None:
    """Synchronize directory entries on POSIX filesystems."""
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
