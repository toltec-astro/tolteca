"""Recorded acquisition lineage and sweep bindings at both supported scopes."""

import json
from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdAssoc,
    DataProdCalRef,
    DataProdIndex,
    DataProdItem,
    DataProdRef,
    DataProdSource,
    Location,
    RawObsBlockHistory,
    RawObsFileCalObsFileMeta,
    RawObsHistoryBlock,
)
from tolteca_datamodels.data_prod.lineage import (
    cal_ref_from_metadata,
    collect_cal_refs,
    resolve_block_relation,
)
from tolteca_datamodels.data_prod.metadata import NamedGroupMeta, RawObsMeta
from tolteca_datamodels.raw_obs_context import (
    RawObsContext,
    RawObsDataItem,
    RawObsRef,
    SweepCalDataItem,
    SweepCalItem,
)
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.types import ToltecDataKind


def _sweep(*interfaces, uid="tcs-145609-0-1", block=1):
    return SweepCalItem(
        type="sweep",
        raw_obs=DataProdRef(uid=uid),
        data_items=tuple(
            SweepCalDataItem(interface=interface, block=block)
            for interface in interfaces
        ),
    )


def _product(parents=None):
    if parents is None:
        parents = {"toltec0": ("tcs", 145609, 0, 1)}
    return DataProd(
        uid="tcs-145609-0-2",
        meta=RawObsMeta(
            name="tcs-145609-0-2",
            data_prod_type="dp_raw_obs",
            master="tcs",
            obsnum=145609,
            subobsnum=0,
            scannum=2,
        ),
        cal_refs=[
            DataProdCalRef(
                interface=interface,
                meta=RawObsFileCalObsFileMeta(
                    cal_master=master,
                    cal_obsnum=obsnum,
                    cal_subobsnum=subobsnum,
                    cal_scannum=scannum,
                    block_history=RawObsBlockHistory(
                        initial="recorded",
                        blocks=(RawObsHistoryBlock(kind="timestream"),),
                    ),
                ),
            )
            for interface, (master, obsnum, subobsnum, scannum) in parents.items()
        ],
    )


def _targets():
    """Portable acquired-block fixtures for network and scope checks."""
    targets = {}
    for uid, count in (("tcs-145609-0-1", 2), ("ics-145605-0-1", 1)):
        master, obsnum, subobsnum, scannum = uid.split("-")
        targets[uid] = DataProd(
            uid=uid,
            meta=RawObsMeta(
                name=uid,
                data_prod_type="dp_raw_obs",
                master=master,
                obsnum=int(obsnum),
                subobsnum=int(subobsnum),
                scannum=int(scannum),
            ),
            cal_refs=[
                DataProdCalRef(
                    interface=interface,
                    meta=RawObsFileCalObsFileMeta(
                        block_history=RawObsBlockHistory(
                            initial="root",
                            blocks=tuple(
                                RawObsHistoryBlock(kind="sweep") for _ in range(count)
                            ),
                        ),
                    ),
                )
                for interface in ("toltec0", "toltec1")
            ],
        )
    return targets


def test_common_sweep_and_telescope_round_trip():
    common = _sweep("toltec0", "toltec1")
    context = RawObsContext(
        raw_obs=DataProdRef(uid="tcs-145609-0-2"),
        data_items=tuple(
            RawObsDataItem(interface=interface, block=None if interface == "tel" else 0)
            for interface in ("toltec1", "tel", "toltec0")
        ),
        cal_items=(common,),
    )
    restored = RawObsContext.model_validate_json(context.model_dump_json())
    restored.validate_lineage(
        _product(dict.fromkeys(("toltec0", "toltec1"), ("tcs", 145609, 0, 1))),
        cal_products=_targets(),
    )
    assert restored == context
    assert tuple(item.interface for item in restored.data_items) == (
        "toltec1",
        "tel",
        "toltec0",
    )
    assert not restored.data_items[1].cal_items
    assert restored.cal_items[0].raw_obs.uid == "tcs-145609-0-1"
    assert restored.cal_items[0].data_items[0].block == 1
    assert "filepath" not in context.model_dump_json()


def test_context_derives_each_networks_parent_from_recorded_lineage():
    # A deliberately divergent record checks that no shared parent is assumed.
    raw = _product({"toltec0": ("tcs", 145609, 0, 1), "toltec1": ("ics", 145605, 0, 1)})
    before = raw.model_dump_json()
    context = RawObsContext.from_product(
        raw,
        data_items=(
            RawObsDataItem(interface="toltec0", block=0),
            RawObsDataItem(interface="tel"),
            RawObsDataItem(interface="toltec1", block=0),
        ),
        cal_products=_targets(),
    )
    restored = RawObsContext.model_validate_json(context.model_dump_json())
    restored.validate_lineage(raw, cal_products=_targets())
    assert restored.data_items[0].cal_items[0].raw_obs.uid == "tcs-145609-0-1"
    assert not restored.data_items[1].cal_items
    assert restored.data_items[2].cal_items[0].raw_obs.uid == "ics-145605-0-1"
    assert restored.data_items[2].cal_items[0].data_items[0].block == 0
    assert raw.model_dump_json() == before


@pytest.mark.parametrize("scope", ["local", "common"])
def test_arbitrary_sweep_parent_is_rejected_against_recorded_lineage(scope):
    sweep = _sweep("toltec0", uid="tcs-145605-0-1")
    context = RawObsContext(
        raw_obs=_product().reference,
        data_items=(
            RawObsDataItem(
                interface="toltec0",
                block=0,
                cal_items=(sweep,) if scope == "local" else (),
            ),
        ),
        cal_items=(sweep,) if scope == "common" else (),
    )
    restored = RawObsContext.model_validate_json(context.model_dump_json())
    with pytest.raises(ValueError, match="differs from recorded block calibration"):
        restored.validate_lineage(_product(), cal_products=_targets())


@pytest.mark.parametrize("parents", [{}, {"toltec0": (None, 145609, 0, 1)}])
def test_missing_or_incomplete_lineage_does_not_guess_a_parent(parents):
    with pytest.raises(ValueError, match="Missing .* (calibration lineage|history)"):
        RawObsContext.from_product(
            _product(parents),
            data_items=(RawObsDataItem(interface="toltec0", block=0),),
        )


def test_dpa_projection_must_agree_with_recorded_interface_reference():
    raw = _product()
    raw.assocs.append(
        DataProdAssoc(
            data_prod_assoc_type="dpa_raw_obs_cal_obs",
            uid="tcs-145605-0-1",
            context={"tag": "raw_obs_cal_obs", "interfaces": ["toltec0"]},
        )
    )
    with pytest.raises(ValueError, match="Conflicting calibration lineage and DPA"):
        RawObsContext.from_product(
            raw, data_items=(RawObsDataItem(interface="toltec0", block=0),)
        )


def test_detector_selection_requires_block_and_binding():
    with pytest.raises(ValueError, match="requires a source block"):
        RawObsContext.from_product(
            _product(), data_items=(RawObsDataItem(interface="toltec0"),)
        )
    unbound = RawObsContext(
        raw_obs=_product().reference,
        data_items=(RawObsDataItem(interface="toltec0", block=0),),
    )
    with pytest.raises(ValueError, match="differs from recorded block"):
        unbound.validate_lineage(_product(), cal_products=_targets())


def test_lineage_must_belong_to_the_context_acquisition():
    context = RawObsContext(
        raw_obs=DataProdRef(uid="tcs-145610-0-2"),
        data_items=(RawObsDataItem(interface="toltec0", block=0),),
    )
    with pytest.raises(ValueError, match="does not match"):
        context.validate_lineage(_product())
    with pytest.raises(ValueError, match="requires a raw-observation product"):
        RawObsContext.from_product(
            DataProd(
                uid="b7f65f98-4f32-4e0a-a66a-cd278670f7cc",
                meta=NamedGroupMeta(data_prod_type="dp_named_group", name="group"),
            ),
            data_items=(RawObsDataItem(interface="toltec0", block=0),),
        )


@pytest.mark.parametrize("scope", ["local", "common", "both"])
def test_overlapping_sweep_bindings_fail_even_when_identical(scope):
    sweep = _sweep("toltec0")
    with pytest.raises(ValidationError, match="Conflicting sweep"):
        RawObsContext(
            raw_obs=DataProdRef(uid="tcs-145609-0-2"),
            data_items=(
                RawObsDataItem(
                    interface="toltec0",
                    block=0,
                    cal_items=(sweep, sweep)
                    if scope == "local"
                    else (sweep,)
                    if scope == "both"
                    else (),
                ),
            ),
            cal_items=(sweep, sweep)
            if scope == "common"
            else (sweep,)
            if scope == "both"
            else (),
        )


@pytest.mark.parametrize("interfaces", [("toltec1",), ("toltec0", "toltec1")])
def test_local_calibration_cannot_target_another_network(interfaces):
    with pytest.raises(ValidationError, match="select only its interface"):
        RawObsDataItem(interface="toltec0", block=0, cal_items=(_sweep(*interfaces),))


def test_common_calibration_must_target_selected_data():
    with pytest.raises(
        ValidationError, match="unselected detector interface 'toltec1'"
    ):
        RawObsContext(
            raw_obs=DataProdRef(uid="tcs-145609-0-2"),
            data_items=(RawObsDataItem(interface="toltec0", block=0),),
            cal_items=(_sweep("toltec1"),),
        )


def test_duplicate_interfaces_do_not_create_multiple_science_or_sweep_items():
    with pytest.raises(ValidationError, match="distinct source selections"):
        RawObsContext(
            raw_obs=DataProdRef(uid="tcs-145609-0-2"),
            data_items=(RawObsDataItem(interface="toltec0", block=0),) * 2,
        )
    with pytest.raises(ValidationError, match="distinct interfaces"):
        _sweep("toltec0", "toltec0")


@pytest.mark.parametrize("block", [-1, "last", "1", True])
def test_sweep_block_must_be_an_exact_nonnegative_integer(block):
    with pytest.raises(ValidationError):
        _sweep("toltec0", block=block)


@pytest.mark.parametrize(
    "uid", ["145609-0-1", "tcs-145609", "tcs-145609-0-1_proc", "main"]
)
def test_raw_reference_excludes_partial_identity_and_product_selections(uid):
    with pytest.raises(ValidationError):
        TypeAdapter(RawObsRef).validate_python(DataProdRef(uid=uid))


def test_reference_normalization_preserves_namespace_and_supplied_reference():
    ref = DataProdRef(
        uid="TCS-00145609-000-0001",
        namespace_uid=UUID("00000000-0000-0000-0000-000000000001"),
    )
    normalized = TypeAdapter(RawObsRef).validate_python(ref)
    assert normalized.uid == "tcs-145609-0-1"
    assert normalized.namespace_uid == ref.namespace_uid
    assert ref.uid == "TCS-00145609-000-0001"


def test_initial_slice_does_not_accept_unimplemented_calibration_roles_or_fields():
    payload = {
        "type": "sweep",
        "raw_obs": {"uid": "tcs-145609-0-1"},
        "data_items": [{"interface": "toltec0", "block": 1}],
    }
    with pytest.raises(ValidationError, match="Input should be 'sweep'"):
        SweepCalItem.model_validate_json(json.dumps(payload | {"type": "model"}))
    with pytest.raises(ValidationError, match="Extra inputs"):
        SweepCalItem.model_validate_json(json.dumps(payload | {"fit": True}))
    with pytest.raises(ValidationError, match="at least 1 item"):
        RawObsContext(raw_obs=DataProdRef(uid="tcs-145609-0-2"), data_items=())
    with pytest.raises(ValidationError, match="at least 1 item"):
        _sweep()


def test_real_raw_and_tune_context_keeps_indexes_and_payloads_separate(data_lmt_path):
    root = data_lmt_path / "toltec/tcs/toltec0"
    names = (
        "toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
        "toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc",
    )
    indexes = []
    metadata = []
    for name in names:
        with open_toltec_kids(root / name) as data:
            meta = data.toltec_kids.block_index.meta
            assert meta is not None
            assert meta.uid_raw_obs is not None
            metadata.append(meta)
            if meta.scannum == 1:
                blocks = data.toltec_kids.block_index.index
                assert blocks.block.values.tolist() == [0, 1]
                assert (
                    ToltecDataKind(int(blocks.data_kind[1])) & ToltecDataKind.raw_sweep
                )
        indexes.append(
            DataProdIndex(
                product=DataProd(
                    uid=meta.uid_raw_obs,
                    meta=RawObsMeta(
                        name=meta.uid_raw_obs,
                        data_prod_type="dp_raw_obs",
                        master=meta.master,
                        obsnum=meta.obsnum,
                        subobsnum=meta.subobsnum,
                        scannum=meta.scannum,
                    ),
                    data_items=[
                        DataProdItem(
                            meta={"interface": meta.interface},
                            source=DataProdSource.model_validate(
                                {"location": "raw", "path": name}
                            ),
                        )
                    ],
                ),
                locations={"raw": Location(root=str(root))},
            )
        )
    indexes = [
        index.model_copy(
            update={
                "product": index.product.model_copy(
                    update={
                        "cal_refs": list(
                            collect_cal_refs(index.product, metadata=(meta,)).values()
                        ),
                    }
                )
            }
        )
        for index, meta in zip(indexes, metadata, strict=True)
    ]
    raw, tune = indexes
    assert (
        metadata[0].cal_obsnum,
        metadata[0].cal_subobsnum,
        metadata[0].cal_scannum,
    ) == (145609, 0, 1)
    assert metadata[0].cal_master is None
    before = [index.model_dump_json() for index in indexes]
    context = RawObsContext.from_product(
        raw.product,
        data_items=(RawObsDataItem(interface="toltec0", block=0),),
        metadata=metadata[:1],
        cal_products={tune.product.uid: tune.product},
    )
    sweep = context.data_items[0].cal_items[0]
    assert sweep.raw_obs.uid == tune.product.uid
    assert sweep.data_items[0].block == 1
    relation = resolve_block_relation(tune.product, "toltec0", 1)
    assert relation.kind == "setup_lineage"
    assert (relation.target_uid, relation.target_block) == (tune.product.uid, 0)
    context.validate_lineage(raw.product, cal_products={tune.product.uid: tune.product})
    assert [index.model_dump_json() for index in indexes] == before


def test_inspected_history_uses_acquired_prefix_not_tune_capacity(data_lmt_path):
    path = (
        data_lmt_path
        / "toltec/tcs/toltec0"
        / ("toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc")
    )
    with open_toltec_kids(path) as data:
        scalar = cal_ref_from_metadata(data.toltec_kids.meta)
        meta = data.toltec_kids.block_index.meta
    assert scalar.meta.block_history is None
    assert meta.n_blocks_max == 2
    # An earlier inspection snapshot sees just the first acquired block.
    prefix = meta.model_copy(update={"n_blocks": 1})
    prefix._block_index = meta.block_index.isel(block=slice(0, 1))
    history = cal_ref_from_metadata(prefix).meta.block_history
    assert len(history.blocks) == 1
    assert history.sweep_before() == 0
    with pytest.raises(ValueError, match="not acquired"):
        history.sweep_before(1)
    # A selection cannot relabel acquisition block 1 as block 0.
    selected = meta.model_copy(update={"n_blocks": 1})
    selected._block_index = meta.block_index.isel(block=slice(1, 2))
    with pytest.raises(ValueError, match="complete acquired prefix"):
        cal_ref_from_metadata(selected)
