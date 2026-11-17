"""TolTECA product records and portable index documents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from ..core.data_prod import DataProdBase, DataProdIndexBase, DataProdRef
from ..core.data_prod._io import _read_yaml
from ..core.data_prod._primary import _resolve_primary, _select_primary_assoc
from .calibration import DataProdCalRef  # noqa: TC001 -- Pydantic resolves fields
from .identity import (
    make_obs_product_uid,
    make_raw_obs_product_uid,
    normalize_primary_uid,
)
from .metadata import ObsMeta, ProductMetadata, RawObsMeta, VersionGroupMeta
from .review import DataProdReview
from .types import LOGICAL_PRODUCT_TYPES, PRIMARY_PRODUCT_TYPES, NamedGroupUid

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["DataProd", "DataProdIndex"]


class DataProd(DataProdBase[ProductMetadata]):
    """TolTECA product identity, typed metadata and associations.

    Observations, raw acquisitions, named collections, calibration groups and
    version groups have global identity: their namespace_uid must be None.
    Derived products can acquire their namespace when materialized.
    """

    cal_refs: list[DataProdCalRef] = Field(
        default_factory=list,
        description=(
            "Recorded calibration evidence for raw products only, with at most "
            "one reference per interface."
        ),
    )

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        if not self.meta.data_prod_type or not self.meta.name:
            msg = "Product metadata requires data_prod_type and name"
            raise ValueError(msg)
        if self.meta.data_prod_type in {"dp_obs", "dp_raw_obs"}:
            self.uid = normalize_primary_uid(self.uid)
        if isinstance(self.meta, ObsMeta):
            expected = make_obs_product_uid(self.meta.master, self.meta.obsnum)
            if self.uid != expected:
                msg = f"Observation UID must be {expected!r}"
                raise ValueError(msg)
        if isinstance(self.meta, RawObsMeta):
            expected = make_raw_obs_product_uid(
                self.meta.master,
                self.meta.obsnum,
                self.meta.subobsnum,
                self.meta.scannum,
            )
            if self.uid != expected:
                msg = f"Raw-observation UID must be {expected!r}"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_named_identity(self) -> Self:
        if self.meta.data_prod_type == "dp_named_group":
            self.uid = str(TypeAdapter(NamedGroupUid).validate_python(self.uid))
        return self

    @model_validator(mode="after")
    def _validate_namespace(self) -> Self:
        if (
            self.meta.data_prod_type in LOGICAL_PRODUCT_TYPES
            and self.namespace_uid is not None
        ):
            msg = (
                f"{self.meta.data_prod_type} is a global logical product; "
                "namespace_uid must be None"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_cal_refs(self) -> Self:
        if self.cal_refs:
            if self.meta.data_prod_type != "dp_raw_obs":
                msg = "Only raw observations carry interface calibration references"
                raise ValueError(msg)
            interfaces = [entry.interface for entry in self.cal_refs]
            if len(set(interfaces)) != len(interfaces):
                msg = "Calibration references require distinct interfaces"
                raise ValueError(msg)
        return self

    def get_primary(
        self, lookup: Callable[[DataProdRef], DataProd] | None = None
    ) -> DataProd:
        """Resolve the primary observation, acquisition or named collection.

        Parameters
        ----------
        lookup : callable or None, optional
            Retrieve an associated product by its DataProdRef identity. For an
            in-memory collection, pass ``products.__getitem__``. Required for
            derived products; ``dp_raw_obs``, ``dp_obs`` and ``dp_named_group``
            return self without traversing membership.

        Returns
        -------
        DataProd
            Terminal logical product reached through ``is_primary`` associations.

        Raises
        ------
        ValueError
            A nonterminal product has no unique primary link, lookup is missing,
            a lookup returns a different UID or namespace, or the links form a
            cycle of complete product references.
        KeyError
            The lookup cannot find an associated product.
        """

        def get_next(product):
            if product.meta.data_prod_type in PRIMARY_PRODUCT_TYPES:
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

        return _resolve_primary(self, get_next, lambda product: product.reference)


class DataProdIndex(DataProdIndexBase[DataProd]):
    """Portable document for a TolTECA product and its supporting records."""

    document: Literal["data_prod_index"] = Field(
        default="data_prod_index",
        description="Document kind; a live version-group header has its own kind.",
    )
    # FastAPI inspects field types outside their model context, where inherited
    # Self cannot resolve (fastapi/fastapi#14341). Bind the concrete document type.
    products: list[DataProdIndex] = Field(default_factory=list)
    reviews: list[DataProdReview] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_reviews(self) -> Self:
        if any(review.product != self.product.reference for review in self.reviews):
            msg = "Index reviews must refer to the index's exact product"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_version_members(self) -> Self:
        meta = self.product.meta
        if not isinstance(meta, VersionGroupMeta):
            return self
        members = {
            DataProdRef(uid=assoc.uid, namespace_uid=assoc.namespace_uid)
            for assoc in self.product.assocs
            if assoc.data_prod_assoc_type == "dpa_version_group_data_prod"
        }
        for index in self.products:
            if (
                index.product.reference in members
                and index.product.meta.data_prod_type != meta.content_data_prod_type
            ):
                msg = (
                    f"Version member {index.product.uid!r} must have product type "
                    f"{meta.content_data_prod_type!r}"
                )
                raise ValueError(msg)
        return self

    @classmethod
    def read(cls, path, *, run_uids=None):
        """Read a portable index with relative sources bound to its own directory.

        Parameters
        ----------
        path : str or pathlib.Path
            Index file or directory containing ``index.yaml``.
        run_uids : iterable of UUID or None, optional
            For a version group, return only these realizations, their membership
            edges and tags selecting their exact UID and namespace. Primary and
            other non-membership relationships are retained. None expands the
            complete committed membership. Absent or uncommitted requested
            members are omitted. Payloads stay unopened.

        Returns
        -------
        DataProdIndex
            Group and selected members, or one product index.

        Raises
        ------
        ValueError
            The header is not a version group, an included member has a different
            product type, or a committed membership record disagrees with the
            requested run or its realized product.
        """
        path = Path(path)
        if path.is_dir():
            path /= "index.yaml"
        selected = dict.fromkeys(run_uids) if run_uids is not None else None
        from ._versions import (  # noqa: PLC0415 -- uses DataProdIndex
            GroupHeader,
            parse_index_document,
        )

        document = parse_index_document(_read_yaml(path))
        if isinstance(document, GroupHeader):
            index = document.expand(path.parent, run_uids=selected)
        else:
            index = document
            index._bind_locations(path.parent.resolve())
        if (
            selected is not None
            and index.product.meta.data_prod_type == "dp_version_group"
        ):
            index.products = [
                p for p in index.products if p.run and p.run.uid in selected
            ]
            members = {p.product.reference for p in index.products}
            index.product.assocs = [
                a
                for a in index.product.assocs
                if a.data_prod_assoc_type != "dpa_version_group_data_prod"
                or DataProdRef(uid=a.uid, namespace_uid=a.namespace_uid) in members
            ]
            index.product.tags = {
                label: tag
                for label, tag in index.product.tags.items()
                if tag.current in members
            }
        return index
