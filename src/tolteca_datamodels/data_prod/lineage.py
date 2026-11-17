"""Recorded acquisition anchors and block calibration history.

These operations perform no I/O and do not mutate their inputs. Database callers
supply indexed lookup results; file workflows supply known acquisition identities
and interfaces. Both use the same normalization, merging and resolution rules.
Block histories supplement the acquisition anchor with acquired block order.
The same resolver consumes portable records prepared from files or DPDB.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import (  # noqa: TC003 -- public annotations
    Collection,
    Iterable,
    Mapping,
)
from typing import Literal

from ..lmt_base.funcs import make_uid_raw_obs
from ..toltec.metadata import ToltecRawObsBlocksMetadata, ToltecRawObsMetadata
from ..toltec.types import ToltecDataKind, ToltecInfo
from .calibration import (
    DataProdCalRef,
    RawObsBlockHistory,
    RawObsBlockRelation,
    RawObsFileCalObsFileMeta,
    RawObsHistoryBlock,
)
from .index import DataProd  # noqa: TC001 -- public annotations
from .metadata import RoachInterfaceMeta

__all__ = [
    "CalRefResolutionReason",
    "cal_ref_candidate_uids",
    "cal_ref_from_metadata",
    "collect_cal_refs",
    "merge_cal_ref_metadata",
    "resolve_block_relation",
    "resolve_cal_ref",
]

type CalRefResolutionReason = Literal[
    "incomplete_reference", "ambiguous_reference", "missing_target"
]
"""Why a recorded calibration reference could not resolve in the supplied lookup."""


def cal_ref_from_metadata(meta: object) -> DataProdCalRef | None:
    """Normalize the calibration reference recorded by one interface.

    Parameters
    ----------
    meta : object
        Opened-file ToltecRawObsMetadata or acquisition-record RoachInterfaceMeta.
        ToltecRawObsBlocksMetadata also supplies the inspected acquired index.
        Other metadata types contribute no calibration reference.

    Returns
    -------
    DataProdCalRef or None
        Interface and known calibration quartet, without a resolved target or
        invented audit attribution. None means no interface/reference evidence.
        A VNA uses its own quartet even if its TargSweep values are stale.
        Inspected block metadata contributes acquired roles as well. The original
        metadata is preserved, including missing calibration master.

    Raises
    ------
    ValueError
        An attached index is a selection rather than an acquired prefix, or
        contains data other than identified raw sweep/timestream blocks.
    """
    if (
        not isinstance(meta, (RoachInterfaceMeta, ToltecRawObsMetadata))
        or not meta.interface
    ):
        return None
    prefix = "" if meta.obs_type == "vna" else "cal_"
    values = {
        f"cal_{key}": getattr(meta, f"{prefix}{key}")
        for key in ("master", "obsnum", "subobsnum", "scannum")
    }
    history = _block_history_from_metadata(meta)
    if history is None and all(value is None for value in values.values()):
        return None
    if values["cal_master"] is not None:
        values["cal_master"] = values["cal_master"].lower()
    return DataProdCalRef(
        interface=meta.interface,
        meta=RawObsFileCalObsFileMeta(**values, block_history=history),
    )


def _block_history_from_metadata(meta):
    if not isinstance(meta, ToltecRawObsBlocksMetadata) or meta.block_index is None:
        return None
    index = meta.block_index
    if index.block.values.tolist() != list(range(meta.n_blocks)):
        msg = "Block history requires the complete acquired prefix, not a selection"
        raise ValueError(msg)
    if meta.n_blocks == 0:
        return None
    blocks = []
    for value in index.data_kind.values:
        kind = ToltecDataKind(int(value))
        if kind & ToltecDataKind.raw_sweep and not kind & ToltecDataKind.raw_timestream:
            role = "sweep"
        elif kind == ToltecDataKind.raw_timestream:
            role = "timestream"
        else:
            msg = "Block history requires identified raw sweep/timestream blocks"
            raise ValueError(msg)
        blocks.append(RawObsHistoryBlock(kind=role))
    initial = (
        "root"
        if meta.obs_type == "vna"
        else (
            "recorded"
            if any(
                value is not None
                for value in (meta.cal_obsnum, meta.cal_subobsnum, meta.cal_scannum)
            )
            else "unknown"
        )
    )
    return RawObsBlockHistory(initial=initial, blocks=tuple(blocks))


def merge_cal_ref_metadata(
    previous: RawObsFileCalObsFileMeta, incoming: RawObsFileCalObsFileMeta
) -> RawObsFileCalObsFileMeta:
    """Enrich calibration identity and block history without replacing facts.

    Parameters
    ----------
    previous, incoming : RawObsFileCalObsFileMeta
        Evidence for the same acquisition interface. Missing values contribute
        no new fact. Histories may append blocks or resolve an unknown initial state.
        The caller establishes the common source/interface identity.

    Returns
    -------
    RawObsFileCalObsFileMeta
        Combined evidence; neither input is modified.

    Raises
    ------
    ValueError
        The inputs disagree on an established identity component, initial state,
        or block role.
    """
    values = previous.model_dump()
    for key, value in incoming.model_dump(exclude={"block_history"}).items():
        if values[key] is not None and value is not None and values[key] != value:
            msg = f"Conflicting calibration reference field {key!r}"
            raise ValueError(msg)
        if value is not None:
            values[key] = value
    values["block_history"] = _merge_block_history(
        previous.block_history, incoming.block_history
    )
    return RawObsFileCalObsFileMeta(**values)


def cal_ref_candidate_uids(meta: RawObsFileCalObsFileMeta) -> tuple[str, ...]:
    """Return exact identities to look up for a recorded calibration quartet.

    Parameters
    ----------
    meta : RawObsFileCalObsFileMeta
        Known calibration identity fields.

    Returns
    -------
    tuple of str
        One UID for a complete quartet; possible TolTEC-master UIDs when only
        master is missing; empty when a numeric component is missing. These are
        lookup keys, not evidence that a target exists or a master is known.
    """
    if meta.uid is not None:
        return (meta.uid,)
    triple = meta.cal_obsnum, meta.cal_subobsnum, meta.cal_scannum
    if meta.cal_master is None and all(value is not None for value in triple):
        candidates = []
        for master in ToltecInfo.masters:
            uid = make_uid_raw_obs(master, *triple)
            assert uid is not None
            candidates.append(uid)
        return tuple(candidates)
    return ()


def resolve_cal_ref(
    reference: DataProdCalRef,
    targets: Mapping[str, Collection[str]],
) -> tuple[DataProdCalRef, CalRefResolutionReason | None]:
    """Resolve a recorded parent against known acquisitions and interfaces.

    Parameters
    ----------
    reference : DataProdCalRef
        Recorded interface reference. An already resolved reference is retained.
    targets : mapping of str to collection of str
        Known raw-acquisition UIDs and their known calibration interfaces. A
        complete recorded quartet needs only its target product to exist. A
        missing master requires exactly one matching triple on the same interface.
        Callers supply actual target evidence, never a science-master default.

    Returns
    -------
    tuple
        Updated reference and unresolved reason, or None for the reason on
        success. Reasons distinguish incomplete identity, multiple matching
        masters, and an absent target. Partial and ambiguous evidence is retained.
        Resolution does not certify payload availability or sweep-block contents.
    """
    if reference.target_uid is not None:
        return reference, None
    meta = reference.meta
    uid = meta.uid
    if uid is not None:
        if uid not in targets:
            return reference, "missing_target"
        return reference.model_copy(update={"target_uid": uid}), None
    matches = [
        uid
        for uid in cal_ref_candidate_uids(meta)
        if uid in targets and reference.interface in targets[uid]
    ]
    if len(matches) != 1:
        return reference, "ambiguous_reference" if matches else "incomplete_reference"
    [uid] = matches
    resolved = meta.model_copy(update={"cal_master": uid.split("-", 1)[0]})
    return reference.model_copy(update={"meta": resolved, "target_uid": uid}), None


def collect_cal_refs(
    product: DataProd,
    *,
    metadata: Iterable[object] = (),
    targets: Mapping[str, Collection[str]] | None = None,
) -> dict[str, DataProdCalRef]:
    """Collect one raw acquisition's lineage from stored facts and source metadata.

    Parameters
    ----------
    product : DataProd
        Raw-acquisition record. Its portable cal_refs supply previously
        learned facts and attribution; any calibration DPAs must agree with them.
    metadata : iterable of object, optional
        Typed source metadata for this acquisition, such as opened KIDs-file
        metadata. Empty by default. Recognized interface evidence enriches stored
        facts; metadata for another acquisition is rejected. Other types are ignored.
    targets : mapping of str to collection of str or None, optional
        Target evidence accepted by resolve_cal_ref. None performs
        no target lookup. An empty mapping explicitly supplies no known targets.

    Returns
    -------
    dict of str to DataProdCalRef
        References keyed by interface, including incomplete references. The
        product, source metadata and supplied target mapping remain unchanged.

    Raises
    ------
    ValueError
        The product is not raw, source identity differs, evidence conflicts, or
        a calibration DPA contradicts the recorded interface reference.

    Notes
    -----
    Only the supplied facts are examined; no files or database are opened.
    Consumers requiring a complete identity must reject an incomplete result.
    A complete quartet without target_uid records identity without establishing
    that the target product is present in the supplied lookup.
    """
    if product.meta.data_prod_type != "dp_raw_obs":
        msg = "Calibration lineage requires a raw-observation product"
        raise ValueError(msg)
    references = {entry.interface: entry for entry in product.cal_refs}
    for source in metadata:
        incoming = cal_ref_from_metadata(source)
        if incoming is None:
            continue
        # The supported source types have acquisition identity fields.
        assert isinstance(source, (RoachInterfaceMeta, ToltecRawObsMetadata))
        uid = make_uid_raw_obs(
            source.master.lower(), source.obsnum, source.subobsnum, source.scannum
        )
        if uid != product.uid:
            msg = f"Calibration source metadata belongs to {uid!r}, not {product.uid!r}"
            raise ValueError(msg)
        previous = references.get(incoming.interface)
        references[incoming.interface] = (
            incoming
            if previous is None
            else previous.model_copy(
                update={"meta": merge_cal_ref_metadata(previous.meta, incoming.meta)}
            )
        )
    if targets is not None:
        references = {
            interface: resolve_cal_ref(entry, targets)[0]
            for interface, entry in references.items()
        }
    for assoc in product.assocs:
        if assoc.data_prod_assoc_type != "dpa_raw_obs_cal_obs":
            continue
        for interface in assoc.context.get("interfaces", ()):
            entry = references.get(interface)
            if (
                entry is not None
                and entry.meta.uid is not None
                and (entry.meta.uid != assoc.uid or assoc.namespace_uid is not None)
            ):
                msg = f"Conflicting calibration lineage and DPA for {interface!r}"
                raise ValueError(msg)
    return references


def _merge_block_history(
    previous: RawObsBlockHistory | None, incoming: RawObsBlockHistory | None
) -> RawObsBlockHistory | None:
    if previous is None:
        return incoming
    if incoming is None:
        return previous
    initial = previous.initial
    if initial == "unknown":
        initial = incoming.initial
    elif incoming.initial not in ("unknown", initial):
        msg = "Conflicting block-history initial state"
        raise ValueError(msg)
    blocks = list(previous.blocks)
    for index, entry in enumerate(incoming.blocks):
        if index == len(blocks):
            blocks.append(entry)
            continue
        old = blocks[index]
        if old.kind != entry.kind:
            msg = f"Conflicting block kind at {index}"
            raise ValueError(msg)
    return RawObsBlockHistory(initial=initial, blocks=tuple(blocks))


def resolve_block_relation(
    product: DataProd,
    interface: str,
    block: int,
    *,
    products: Mapping[str, DataProd] | None = None,
    metadata: Iterable[object] = (),
) -> RawObsBlockRelation | None:
    """Resolve a block's calibration or setup lineage to an exact sweep.

    Parameters
    ----------
    product : DataProd
        Source raw acquisition with portable block-history evidence.
    interface : str
        Detector interface shared by the source and its calibration sweeps.
    block : int
        Zero-based acquired source block; never a caller-chosen target block.
    products : mapping of str to DataProd or None, optional
        Referenced raw products keyed by UID. Only the recorded dependency path
        is traversed. None supplies no additional acquisitions. Include the
        target history even when its complete quartet is already known.
    metadata : iterable of object, optional
        Additional source acquisition metadata, as in collect_cal_refs.

    Returns
    -------
    RawObsBlockRelation or None
        Exact target and relation kind: calibration for a timestream,
        setup_lineage for a sweep. None means an established root.
        Unresolved references raise instead of returning None.

    Raises
    ------
    ValueError
        Evidence is missing, ambiguous, inconsistent or cyclic; the selected
        block is absent; or the result would be a self/forward local dependency.

    Notes
    -----
    No I/O or mutation occurs. Timestream-only referenced acquisitions inherit
    their incoming state. Unknown boundary evidence never selects a latest fit.
    Payload availability and tone/power compatibility require separate checks.
    """
    known = dict(products or {})
    if any(key != value.uid for key, value in known.items()):
        msg = "Calibration product lookup keys must match product UIDs"
        raise ValueError(msg)
    known[product.uid] = product
    targets = {
        uid: tuple(entry.interface for entry in value.cal_refs)
        for uid, value in known.items()
    }
    uid = product.uid
    selected: int | None = block
    seen = set()
    relation_kind = None
    while True:
        key = uid, selected
        if key in seen:
            msg = "Cyclic block calibration history"
            raise ValueError(msg)
        seen.add(key)
        entry = _history_reference(
            known, uid, interface, targets, metadata if uid == product.uid else ()
        )
        history = entry.meta.block_history
        assert history is not None
        state = history.sweep_before(selected)
        if relation_kind is None:
            relation_kind = (
                "calibration"
                if history.blocks[block].kind == "timestream"
                else "setup_lineage"
            )
        if isinstance(state, int):
            if uid == product.uid and state >= block:
                msg = "A block cannot depend on itself or a later local block"
                raise ValueError(msg)
            return RawObsBlockRelation(
                kind=relation_kind, target_uid=uid, target_block=state
            )
        if state == "root":
            return None
        if state == "unknown":
            msg = f"Unknown active sweep for {uid!r}/{interface}"
            raise ValueError(msg)
        parent = entry.meta.uid
        if parent is None:
            msg = f"Missing or incomplete calibration lineage for {interface!r}"
            raise ValueError(msg)
        uid, selected = parent, None


def _history_reference(known, uid, interface, targets, metadata):
    current = known.get(uid)
    if current is None:
        msg = f"Missing calibration product {uid!r}"
        raise ValueError(msg)
    references = collect_cal_refs(current, metadata=metadata, targets=targets)
    entry = references.get(interface)
    if entry is None or entry.meta.block_history is None:
        msg = f"Missing block history for {uid!r}/{interface}"
        raise ValueError(msg)
    if entry.meta.block_history.initial == "root" and entry.meta.uid not in (None, uid):
        msg = f"Established root conflicts with recorded parent for {uid!r}/{interface}"
        raise ValueError(msg)
    return entry


def _cal_ref_targets(source_uid, references):
    """Group resolved external references by target, preserving interface scope."""
    targets = defaultdict(set)
    for ref in references:
        if ref.target_uid is not None and ref.target_uid != source_uid:
            targets[ref.target_uid].add(ref.interface)
    return {uid: sorted(interfaces) for uid, interfaces in sorted(targets.items())}
