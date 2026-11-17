"""Build portable observation inventories from unordered acquisition files."""

from __future__ import annotations

from copy import replace
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from tollan.utils.fileloc import FileLoc, fileloc

from ..core.data_prod import DataProdAssoc, DataProdItem, DataProdSource, Location
from ..lmt.metadata import LmtTelMetadata
from ..lmt_base.metadata import LmtRawObsMetadata
from ..source import SourceSourceLoc
from ..toltec.hwpr import ToltecHwprAccessor
from ..toltec.kids.core import ToltecKidsAccessor
from ..toltec.metadata import ToltecRawObsMetadata
from ..toltec.source import open_toltec_source
from ..toltec.types import ToltecDataKind
from .batch import DataProdBatch
from .identity import make_obs_product_uid, make_raw_obs_product_uid
from .index import DataProd, DataProdIndex
from .lineage import _cal_ref_targets, collect_cal_refs, merge_cal_ref_metadata
from .metadata import (
    ObsMeta,
    RawObsCalObsContext,
    RawObsMeta,
    _load_source_metadata,
    _retain_calibration_master,
    _retort,
)
from .types import DataKind

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ..core.source import SourceReader

__all__ = ["build_raw_obs_batch", "read_raw_obs_index"]


def _include_raw_metadata(previous, meta, kind, obs_datetime=None):
    """Collate interface facts for both portable and relational raw products."""
    header = {}
    if isinstance(meta, LmtRawObsMetadata):
        obs_datetime = meta.t0.to_datetime(timezone=UTC)
    if isinstance(meta, LmtTelMetadata):
        header.update(
            obs_goal=meta.obs_goal.value,
            obs_pgm=meta.obs_pgm.value,
            integration_time=float(meta.t_int.to_value("s")),
        )
    if obs_datetime is not None:
        header["obs_datetime"] = min(
            t for t in (previous.obs_datetime, obs_datetime) if t is not None
        )
    return replace(previous, **header, data_kind=previous.data_kind | kind)


def _raw_obs_index(
    accessor: SourceReader,
    source: str | Path | FileLoc,
    *,
    locations: Mapping[str, Location],
) -> DataProdIndex:
    """Inspect one open acquisition, preserving headers separately from blocks."""
    meta = accessor.meta
    if not isinstance(meta, LmtRawObsMetadata):
        msg = "Raw ingestion requires complete acquisition metadata"
        raise TypeError(msg)
    identity = {
        key: getattr(meta, key)
        for key in (
            "master",
            "obsnum",
            "subobsnum",
            "scannum",
        )
    }
    uid = make_raw_obs_product_uid(**identity)
    history_metadata = ()
    if isinstance(accessor, ToltecKidsAccessor):
        assert isinstance(meta, ToltecRawObsMetadata)
        kind = {
            ToltecDataKind.vnasweep: DataKind.VnaSweep,
            ToltecDataKind.targsweep: DataKind.TargetSweep,
            ToltecDataKind.tune: DataKind.Tune,
            ToltecDataKind.raw_timestream: DataKind.RawTimeStream,
        }[accessor.data_kind]
        history_metadata = (accessor.block_index.meta,)
    elif isinstance(accessor, ToltecHwprAccessor):
        kind = DataKind.Hwpr
    elif isinstance(meta, LmtTelMetadata):
        kind = (
            DataKind.LmtTel2
            if Path(meta.filepath_orig).name.startswith("tel_toltec2_")
            else DataKind.LmtTel
        )
    else:
        msg = "Unsupported raw acquisition accessor"
        raise TypeError(msg)
    locations = _normalize_locations(locations)
    reference = _source_reference(source, locations=locations)
    product = DataProd(
        uid=uid,
        meta=_include_raw_metadata(
            RawObsMeta(name=uid, data_prod_type="dp_raw_obs", **identity),
            meta,
            kind.value,
        ),
        data_items=[
            DataProdItem(
                meta=meta.model_dump(mode="json", round_trip=True),
                source=reference,
            )
        ],
    )
    product.cal_refs = list(
        collect_cal_refs(
            product,
            metadata=history_metadata,
        ).values()
    )
    return DataProdIndex(
        locations={reference.location: locations[reference.location]}, product=product
    )


def _normalize_locations(locations):
    result = {}
    for label, location in locations.items():
        if location.is_relative:
            msg = f"Raw source location {label!r} requires an absolute root"
            raise ValueError(msg)
        result[label] = location.model_copy(
            deep=True, update={"root": location.resolve().url}
        )
    return result


def _source_reference(source, *, locations):
    """Use one declared root and the same relative locator in both ingestion routes."""
    uri = fileloc(source).url
    physical_uri = None
    matches = []
    for label, location in locations.items():
        root = urlsplit(location.resolve().url)
        prefix = root._replace(path=root.path.rstrip("/") + "/").geturl()
        if uri.startswith(prefix):
            # A symlink *inside* the archive is a valid named item in that archive.
            matches.append((label, uri.removeprefix(prefix)))
        elif _local_file_uri(root) and _local_file_uri(urlsplit(uri)):
            # Discovery may have resolved a mount alias. Use physical paths only
            # for comparison; retain the declared root and a portable relative path.
            if physical_uri is None:
                physical_uri = Path.from_uri(uri).resolve().as_uri()
            physical_root = Path.from_uri(root.geturl()).resolve().as_uri()
            prefix = physical_root.rstrip("/") + "/"
            if physical_uri.startswith(prefix):
                matches.append((label, physical_uri.removeprefix(prefix)))
    if len(matches) != 1:
        msg = (
            f"Source {uri!r} must match exactly one declared location; "
            f"matched {len(matches)}"
        )
        raise ValueError(msg)
    label, relative = matches[0]
    return DataProdSource.model_validate(
        {
            "path": unquote(relative),
            "location": label,
            "availability_state": "available",
        }
    )


def _local_file_uri(uri):
    return (
        uri.scheme == "file"
        and uri.netloc in {"", "localhost"}
        and not uri.query
        and not uri.fragment
    )


def read_raw_obs_index(
    source: object, *, locations: Mapping[str, Location]
) -> DataProdIndex:
    """Inspect one file and describe its raw acquisition.

    Parameters
    ----------
    source : path-like or SourceSourceLoc
        Acquisition file, or a prepared source with an explicit location.
    locations : mapping of str to Location
        Named absolute archive roots. The source must match exactly one root.

    Returns
    -------
    DataProdIndex
        One raw product with its file item, authoritative header metadata and
        acquired block history. External calibration references remain unresolved.
        Owned files close before return; borrowed sources remain caller-owned.
        No detector I/Q arrays are loaded or reduced.

    Raises
    ------
    TypeError
        The source is not a dataset with complete raw acquisition metadata.
    ValueError
        The file is unsupported or its acquisition metadata/block history is invalid.
    OSError
        The file cannot be opened.

    See Also
    --------
    build_raw_obs_batch : Group files and resolve their recorded relationships.
    """
    item = SourceSourceLoc.model_validate(source)
    try:
        with open_toltec_source(item) as accessor:
            return _raw_obs_index(accessor, _source_location(item), locations=locations)
    except (TypeError, ValueError, OSError) as exc:
        exc.add_note(f"While inspecting acquisition source: {item.source_loc.url}")
        raise


def _source_location(item: SourceSourceLoc) -> str | Path | FileLoc:
    return (
        item.source
        if isinstance(item.source, (str, Path, FileLoc))
        else item.source_loc
    )


def _merge_raw_index(current, incoming):
    for key, location in incoming.locations.items():
        if key in current.locations and current.locations[key].root != location.root:
            msg = f"Conflicting storage location {key!r}"
            raise ValueError(msg)
    current.locations.update(incoming.locations)
    for item in incoming.product.data_items:
        uri = incoming.get_source_uri(item.source)
        old = next(
            (
                i
                for i in current.product.data_items
                if current.get_source_uri(i.source) == uri
            ),
            None,
        )
        meta = _load_source_metadata(item.meta)
        if old is None:
            current.product.data_items.append(item)
        else:
            meta = _retain_calibration_master(meta, _load_source_metadata(old.meta))
            old.meta = _retort.dump(meta)
            old.source = item.source
        current.product.meta = _include_raw_metadata(
            current.product.meta,
            meta,
            incoming.product.meta.data_kind,
        )
    refs = {ref.interface: ref for ref in current.product.cal_refs}
    for ref in incoming.product.cal_refs:
        old = refs.get(ref.interface)
        refs[ref.interface] = (
            ref
            if old is None
            else old.model_copy(
                update={
                    "meta": merge_cal_ref_metadata(old.meta, ref.meta),
                }
            )
        )
    current.product.cal_refs = [refs[key] for key in sorted(refs)]


def _resolve_inventory(indexes):
    raw = {
        uid: index.product
        for uid, index in indexes.items()
        if index.product.meta.data_prod_type == "dp_raw_obs"
    }
    targets = {
        uid: {ref.interface for ref in product.cal_refs} for uid, product in raw.items()
    }
    for uid, product in raw.items():
        refs = collect_cal_refs(product, targets=targets)
        product.cal_refs = [refs[key] for key in sorted(refs)]
        groups = _cal_ref_targets(uid, product.cal_refs)
        assocs = {
            a.uid: a
            for a in product.assocs
            if a.data_prod_assoc_type == "dpa_raw_obs_cal_obs"
        }
        for target, interfaces in sorted(groups.items()):
            assoc = assocs.get(target)
            if assoc is None:
                assoc = DataProdAssoc(
                    data_prod_assoc_type="dpa_raw_obs_cal_obs", uid=target
                )
                product.assocs.append(assoc)
            assoc.context = {
                **assoc.context,
                **_retort.dump(RawObsCalObsContext(interfaces=interfaces)),
            }
    for uid, product in sorted(raw.items()):
        meta = product.meta
        obs_uid = make_obs_product_uid(meta.master, meta.obsnum)
        if obs_uid not in indexes:
            indexes[obs_uid] = DataProdIndex(
                product=DataProd(
                    uid=obs_uid,
                    meta=ObsMeta(
                        name=obs_uid,
                        data_prod_type="dp_obs",
                        master=meta.master,
                        obsnum=meta.obsnum,
                    ),
                )
            )
        observation = indexes[obs_uid].product
        if not any(
            a.uid == uid and a.data_prod_assoc_type == "dpa_obs_raw_obs"
            for a in observation.assocs
        ):
            observation.assocs.append(
                DataProdAssoc(
                    uid=uid,
                    data_prod_assoc_type="dpa_obs_raw_obs",
                )
            )
        observation.meta = replace(
            observation.meta,
            data_kind=observation.meta.data_kind | meta.data_kind,
            obs_datetime=min(
                (
                    t
                    for t in (
                        observation.meta.obs_datetime,
                        meta.obs_datetime,
                    )
                    if t is not None
                ),
                default=None,
            ),
        )


def build_raw_obs_batch(
    sources: Iterable[object],
    *,
    locations: Mapping[str, Location],
    existing: DataProdBatch | None = None,
) -> DataProdBatch:
    """Describe unordered raw files as a portable batch of product indexes.

    Parameters
    ----------
    sources : iterable of path-like or SourceSourceLoc
        KIDs, HWPR and TolTEC telescope files. Files may span acquisitions,
        observations and interfaces. No grouping or calibration identities are
        required from the caller. Use discovery or globbing to select the files.
    locations : mapping of str to Location
        Named absolute archive roots. Every source must match exactly one root;
        each item uses that name and a relative source URI. Use the same root
        declarations when registering these files directly in DPDB.
    existing : DataProdBatch or None, optional
        Previously prepared raw/observation indexes. None starts an empty batch.
        Sources, annotations, acquired history and batch metadata are copied and
        retained. Newly supplied files enrich the copy and resolve dependencies.
        Only unqualified dp_raw_obs and dp_obs indexes are accepted, once per UID.

    Returns
    -------
    DataProdBatch
        Raw-acquisition and observation indexes in UID order, ready for writing
        through DataProdStore.write_index or registering with DPDB's
        ingest_data_prod_batch. Raw products group files by their header quartet;
        observations group acquisitions by master and observation number.
        Calibration uses recorded references and acquired block order. Missing or
        ambiguous targets retain unresolved cal_refs. Input order does not choose
        calibration. Supplied objects are not modified.

    Raises
    ------
    TypeError
        A source is not a dataset with complete raw acquisition metadata.
    ValueError
        A source has no unique declared root, existing entries are unsupported or
        duplicated, or metadata/location declarations conflict with existing facts.
    OSError
        A selected file cannot be read. No files or database records are written.

    Notes
    -----
    Each unique source is inspected once. Owned files close before continuing;
    borrowed sources remain caller-owned. Inspection reads headers and LO/tone
    axes for block identification, not detector I/Q samples. This operation does
    not fit, publish, allocate a workflow run, or
    search an archive for missing dependencies. Reopen saved indexes with
    DataProdIndex.read; wrapping them in a batch preserves their resolution context.
    """
    locations = _normalize_locations(locations)
    result = existing.model_copy(deep=True) if existing is not None else DataProdBatch()
    indexes = {}
    for index in result.items:
        if (
            not isinstance(index, DataProdIndex)
            or index.product.meta.data_prod_type not in {"dp_raw_obs", "dp_obs"}
            or index.product.namespace_uid is not None
        ):
            msg = "Existing batch requires unqualified raw/observation product indexes"
            raise ValueError(msg)
        uid = index.product.uid
        if uid in indexes:
            msg = f"Duplicate product in existing batch: {uid!r}"
            raise ValueError(msg)
        for key, location in index.locations.items():
            location.root = index.resolve_location(key).url
            if key in locations and location.root != locations[key].root:
                msg = f"Conflicting storage location {key!r}"
                raise ValueError(msg)
        indexes[uid] = index
    unique = {}
    for source in sources:
        item = SourceSourceLoc.model_validate(source)
        uri = fileloc(_source_location(item)).url
        _source_reference(uri, locations=locations)
        unique.setdefault(uri, item)
    for _, source in sorted(unique.items()):
        index = read_raw_obs_index(source, locations=locations)
        uid = index.product.uid
        if uid in indexes:
            _merge_raw_index(indexes[uid], index)
        else:
            indexes[uid] = index
    _resolve_inventory(indexes)
    return result.model_copy(
        update={"items": [indexes[uid] for uid in sorted(indexes)]}
    )
