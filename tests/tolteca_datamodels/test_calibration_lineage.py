"""Shared lineage rules preserve acquisition facts independently of storage."""

from copy import replace

import pytest

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdCalRef,
    RawObsFileCalObsFileMeta,
)
from tolteca_datamodels.data_prod.lineage import (
    cal_ref_from_metadata,
    collect_cal_refs,
    merge_cal_ref_metadata,
    resolve_cal_ref,
)
from tolteca_datamodels.data_prod.metadata import RawObsMeta, RoachInterfaceMeta


def _source(**changes):
    # Deliberately different science/calibration masters exercise no-default rules.
    return replace(
        RoachInterfaceMeta(
            master="tcs",
            obsnum=145609,
            subobsnum=0,
            scannum=2,
            interface="toltec0",
            obs_type="timestream",
            cal_master=None,
            cal_obsnum=145605,
            cal_subobsnum=0,
            cal_scannum=1,
        ),
        **changes,
    )


def _reference():
    ref = cal_ref_from_metadata(_source())
    assert ref is not None
    return ref


def _product(**changes):
    return DataProd(
        uid="tcs-145609-0-2",
        meta=RawObsMeta(
            data_prod_type="dp_raw_obs",
            name="tcs-145609-0-2",
            master="tcs",
            obsnum=145609,
            subobsnum=0,
            scannum=2,
        ),
        **changes,
    )


def test_vna_starts_its_own_lineage_despite_stale_target_sweep():
    source = _source(obs_type="vna", master="ICS")
    reference = cal_ref_from_metadata(source)
    assert reference is not None
    assert reference.meta.uid == "ics-145609-0-2"
    assert reference.target_uid is None
    assert source.cal_obsnum == 145605
    assert source.master == "ICS"
    assert _reference().meta.cal_master is None


@pytest.mark.parametrize(
    "source",
    [
        object(),
        _source(interface=None),
        _source(cal_obsnum=None, cal_subobsnum=None, cal_scannum=None),
    ],
)
def test_absent_reference_evidence_is_not_fabricated(source):
    assert cal_ref_from_metadata(source) is None


def test_partial_replay_retains_learned_parent_and_conflicting_parent_fails():
    partial = _reference().meta
    complete = merge_cal_ref_metadata(
        partial, RawObsFileCalObsFileMeta(cal_master="ics")
    )
    assert complete.uid == "ics-145605-0-1"
    assert merge_cal_ref_metadata(complete, partial) == complete
    assert partial.cal_master is None
    with pytest.raises(ValueError, match="Conflicting calibration reference"):
        merge_cal_ref_metadata(complete, RawObsFileCalObsFileMeta(cal_master="tcs"))


@pytest.mark.parametrize(
    ("targets", "reason", "uid"),
    [
        ({"ics-145605-0-1": ("toltec0",)}, None, "ics-145605-0-1"),
        ({"ics-145605-0-1": ("toltec1",)}, "incomplete_reference", None),
        (
            {"ics-145605-0-1": ("toltec0",), "tcs-145605-0-1": ("toltec0",)},
            "ambiguous_reference",
            None,
        ),
        (
            {"ics-145605-0-1": ("toltec0",), "tcs-145605-0-1": ("toltec1",)},
            None,
            "ics-145605-0-1",
        ),
        ({}, "incomplete_reference", None),
    ],
)
def test_missing_master_requires_unique_same_interface_evidence(targets, reason, uid):
    original = _reference()
    resolved, actual_reason = resolve_cal_ref(original, targets)
    assert (actual_reason, resolved.target_uid) == (reason, uid)
    assert resolved.meta.uid == uid
    assert original.meta.cal_master is None
    assert original.target_uid is None


def test_missing_numeric_component_is_not_inferred_from_available_targets():
    source = _source(cal_scannum=None)
    reference = cal_ref_from_metadata(source)
    assert reference is not None
    resolved, reason = resolve_cal_ref(reference, {"ics-145605-0-1": ("toltec0",)})
    assert resolved == reference
    assert reason == "incomplete_reference"


def test_complete_recorded_identity_needs_only_a_logical_target():
    reference = cal_ref_from_metadata(_source(cal_master="ics"))
    assert reference is not None
    missing, reason = resolve_cal_ref(reference, {})
    assert missing == reference
    assert reason == "missing_target"
    resolved, reason = resolve_cal_ref(reference, {"ics-145605-0-1": ()})
    assert reason is None
    assert resolved.target_uid == "ics-145605-0-1"
    # A learned identity survives an incomplete lookup on a later invocation.
    assert resolve_cal_ref(resolved, {}) == (resolved, None)


def test_file_evidence_merges_with_stored_facts_without_rewriting_either():
    stored = DataProdCalRef(
        interface="toltec0", meta=RawObsFileCalObsFileMeta(cal_master="ics")
    )
    product = _product(cal_refs=[stored])
    source = _source()
    before = product.model_dump_json()
    [reference] = collect_cal_refs(
        product, metadata=(source,), targets={"ics-145605-0-1": ("toltec0",)}
    ).values()
    assert reference.target_uid == "ics-145605-0-1"
    assert product.model_dump_json() == before
    assert source.cal_master is None
    with pytest.raises(ValueError, match="Conflicting calibration reference"):
        collect_cal_refs(product, metadata=(_source(cal_master="tcs"),))


def test_metadata_for_another_acquisition_cannot_supply_context_lineage():
    with pytest.raises(ValueError, match="metadata belongs to"):
        collect_cal_refs(_product(), metadata=(_source(scannum=3),))
