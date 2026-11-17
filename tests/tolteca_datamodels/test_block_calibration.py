"""Recorded block history: Tune chains, mixed fanout and evidence enrichment."""

import pytest

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdCalRef,
    RawObsBlockHistory,
    RawObsBlockRelation,
    RawObsFileCalObsFileMeta,
    RawObsHistoryBlock,
)
from tolteca_datamodels.data_prod.lineage import (
    merge_cal_ref_metadata,
    resolve_block_relation,
)
from tolteca_datamodels.data_prod.metadata import RawObsMeta
from tolteca_datamodels.raw_obs_context import (
    RawObsContext,
    RawObsDataItem,
    SweepCalDataItem,
    SweepCalItem,
)


def product(obs, roles, *, parent=None, initial=None):
    """Explicit acquired-block facts for the user's block-order examples."""
    return DataProd(
        uid=f"tcs-{obs}-0-0",
        meta=RawObsMeta(
            name=str(obs),
            data_prod_type="dp_raw_obs",
            master="tcs",
            obsnum=obs,
            subobsnum=0,
            scannum=0,
        ),
        cal_refs=[
            DataProdCalRef(
                interface="toltec0",
                meta=RawObsFileCalObsFileMeta(
                    cal_master="tcs" if parent is not None else None,
                    cal_obsnum=parent,
                    cal_subobsnum=0 if parent is not None else None,
                    cal_scannum=0 if parent is not None else None,
                    block_history=RawObsBlockHistory(
                        initial=initial or ("root" if parent is None else "recorded"),
                        blocks=tuple(
                            RawObsHistoryBlock(
                                kind="sweep" if role == "s" else "timestream",
                            )
                            for role in roles
                        ),
                    ),
                ),
            )
        ],
    )


def test_two_tunes_and_timestream_form_the_recorded_chain():
    r0, r1, r2 = product(0, "ss"), product(1, "ss", parent=0), product(2, "t", parent=1)
    lookup = {p.uid: p for p in (r0, r1, r2)}
    actual, kinds = [], []
    cursor = (r2.uid, 0)
    while cursor:
        actual.append(cursor)
        uid, block = cursor
        relation = resolve_block_relation(
            lookup[uid], "toltec0", block, products=lookup
        )
        if relation is None:
            break
        kinds.append(relation.kind)
        cursor = relation.target_uid, relation.target_block
    assert kinds == ["calibration", "setup_lineage", "setup_lineage", "setup_lineage"]
    assert actual == [(r2.uid, 0), (r1.uid, 1), (r1.uid, 0), (r0.uid, 1), (r0.uid, 0)]


def test_mixed_fanout_and_cross_acquisition_final_active_sweep():
    mixed = product(0, "ststt")
    continuation = product(1, "t", parent=0)
    later = product(2, "t", parent=1)
    lookup = {p.uid: p for p in (mixed, continuation)}
    ctx = RawObsContext.from_product(
        mixed,
        data_items=tuple(
            RawObsDataItem(interface="toltec0", block=b) for b in (1, 3, 4)
        ),
    )
    assert [i.cal_items[0].data_items[0].block for i in ctx.data_items] == [0, 2, 2]
    ctx.validate_lineage(mixed)
    assert resolve_block_relation(
        later, "toltec0", 0, products=lookup
    ) == RawObsBlockRelation(kind="calibration", target_uid=mixed.uid, target_block=2)
    # A correct acquisition with the wrong sweep was previously accepted.
    wrong = ctx.model_copy(
        update={
            "data_items": (
                ctx.data_items[0].model_copy(
                    update={
                        "cal_items": (
                            SweepCalItem(
                                type="sweep",
                                raw_obs=mixed.reference,
                                data_items=(
                                    SweepCalDataItem(interface="toltec0", block=2),
                                ),
                            ),
                        )
                    }
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="differs from recorded block"):
        wrong.validate_lineage(mixed)
    common = RawObsContext(
        raw_obs=mixed.reference,
        data_items=tuple(RawObsDataItem(interface="toltec0", block=b) for b in (1, 3)),
        cal_items=ctx.data_items[0].cal_items,
    )
    with pytest.raises(ValueError, match="differs from recorded block"):
        common.validate_lineage(mixed)


def test_sweep_setup_lineage_does_not_become_a_cal_item():
    tune = product(0, "ss")
    relation = resolve_block_relation(tune, "toltec0", 1)
    assert relation == RawObsBlockRelation(
        kind="setup_lineage", target_uid=tune.uid, target_block=0
    )
    context = RawObsContext.from_product(
        tune, data_items=(RawObsDataItem(interface="toltec0", block=1),)
    )
    assert context.data_items[0].cal_items == ()
    context.validate_lineage(tune)
    incorrect = context.model_copy(
        update={
            "cal_items": (
                SweepCalItem(
                    type="sweep",
                    raw_obs=tune.reference,
                    data_items=(SweepCalDataItem(interface="toltec0", block=0),),
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="differs from recorded block"):
        incorrect.validate_lineage(tune)


def test_only_the_acquired_tune_prefix_contributes_history():
    partial = product(0, "s", initial="unknown")
    later = product(1, "t", parent=0)
    # No fit, model receipt or allocated second slot is needed to identify block 0.
    assert resolve_block_relation(
        later, "toltec0", 0, products={partial.uid: partial}
    ) == (
        RawObsBlockRelation(kind="calibration", target_uid=partial.uid, target_block=0)
    )
    with pytest.raises(ValueError, match="not acquired"):
        resolve_block_relation(partial, "toltec0", 1)


def test_progressive_history_merges_without_erasing_or_replacing_facts():
    old = product(0, "s", initial="unknown").cal_refs[0].meta
    new = product(0, "sst").cal_refs[0].meta
    merged = merge_cal_ref_metadata(old, new)
    assert merged == new
    assert merge_cal_ref_metadata(merged, old) == new
    assert merge_cal_ref_metadata(merged, RawObsFileCalObsFileMeta()) == new
    with pytest.raises(ValueError, match="Conflicting block kind"):
        merge_cal_ref_metadata(merged, product(0, "st").cal_refs[0].meta)
    with pytest.raises(ValueError, match="Conflicting block-history initial"):
        merge_cal_ref_metadata(merged, product(0, "ss", parent=1).cal_refs[0].meta)


def test_unknown_missing_and_cyclic_histories_do_not_become_roots():
    unknown = product(0, "t", initial="unknown")
    with pytest.raises(ValueError, match="Unknown active"):
        resolve_block_relation(unknown, "toltec0", 0)
    missing = product(0, "t", parent=99)
    with pytest.raises(ValueError, match="Missing calibration product"):
        resolve_block_relation(missing, "toltec0", 0)
    cycle = product(0, "t", parent=0)
    with pytest.raises(ValueError, match="Cyclic"):
        resolve_block_relation(cycle, "toltec0", 0)
    self_cal = product(0, "s", parent=0)
    with pytest.raises(ValueError, match="depend on itself"):
        resolve_block_relation(self_cal, "toltec0", 0)


def test_sweep_preparation_does_not_require_its_setup_ancestors():
    sweep = product(0, "s", parent=99)
    context = RawObsContext.from_product(
        sweep, data_items=(RawObsDataItem(interface="toltec0", block=0),)
    )
    assert context.data_items[0].cal_items == ()
    context.validate_lineage(sweep)
    # Resolving the setup relation itself still requires the recorded parent.
    with pytest.raises(ValueError, match="Missing calibration product"):
        resolve_block_relation(sweep, "toltec0", 0)
