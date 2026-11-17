"""Acquisition references used by raw-observation contexts and calibrations."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator

from ..core.data_prod import DataProdRef
from ..data_prod.identity import normalize_primary_uid

__all__ = ["RawObsProcRef", "RawObsRef"]


def _validate_raw_obs_ref(value: DataProdRef) -> DataProdRef:
    uid = normalize_primary_uid(value.uid)
    if uid.count("-") != 3:  # noqa: PLR2004 -- master plus the three acquisition numbers
        msg = "A raw-observation reference requires the full acquisition quartet"
        raise ValueError(msg)
    return value if uid == value.uid else value.model_copy(update={"uid": uid})


type RawObsRef = Annotated[DataProdRef, AfterValidator(_validate_raw_obs_ref)]
"""Product reference with a canonical acquisition quartet, including master.

Accepts a DataProdRef or its mapping representation. Normalizes case and numeric
padding through the existing primary-identity contract, preserving namespace.
Observation-only identifiers, tags and processed-product identifiers are invalid.
Validation does not look up the referenced product.
"""


def _validate_proc_ref(value: DataProdRef) -> DataProdRef:
    group, sep, run_uid = value.uid.rpartition("/")
    if value.namespace_uid is None or not sep or not group:
        msg = "A model requires a namespaced processed-product realization"
        raise ValueError(msg)
    UUID(run_uid)
    return value


type RawObsProcRef = Annotated[DataProdRef, AfterValidator(_validate_proc_ref)]
"""Exact namespaced processed-product reference ending in a run UUID.

Tags, version-group identities and unqualified paths are not resolved selections.
Product type and acquisition ownership are checked against the pinned index.
"""
