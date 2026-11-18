"""Exact fit selection and workflow provenance on acquired Tune/TS inputs."""

from contextlib import ExitStack, nullcontext
from uuid import uuid4

import pytest
from pydantic import ValidationError

from tests.data_prod_helpers import publish_model as _publish_model
from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdIndex,
    DataProdRef,
    DataProdStore,
    WorkflowRun,
    resolve_index,
)
from tolteca_datamodels.data_prod.lineage import cal_ref_from_metadata
from tolteca_datamodels.data_prod.metadata import RawObsMeta
from tolteca_datamodels.raw_obs_context import (
    KidsModelBinding,
    RawObsContext,
    RawObsDataItem,
    SweepCalDataItem,
    SweepCalItem,
    prepare_run_inputs,
    read_run_inputs,
)
from tolteca_datamodels.toltec import open_toltec_kids


@pytest.fixture
def raw_context(data_lmt_path):
    products = {}
    for filename in (
        "toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc",
        "toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
    ):
        with open_toltec_kids(data_lmt_path / "toltec/tcs/toltec0" / filename) as ds:
            meta = ds.toltec_kids.block_index.meta
            ref = cal_ref_from_metadata(meta)
            assert meta.uid_raw_obs is not None and ref is not None
            products[meta.uid_raw_obs] = DataProd(
                uid=meta.uid_raw_obs,
                meta=RawObsMeta(
                    name=meta.uid_raw_obs,
                    data_prod_type="dp_raw_obs",
                    master=meta.master,
                    obsnum=meta.obsnum,
                    subobsnum=meta.subobsnum,
                    scannum=meta.scannum,
                ),
                cal_refs=[ref],
            )
    raw = products["tcs-145609-0-2"]
    context = RawObsContext.from_product(
        raw,
        data_items=(RawObsDataItem(interface="toltec0", block=0),),
        cal_products=products,
    )
    return context, products


def _bind(context, index):
    cal = context.data_items[0].cal_items[0]
    [item] = cal.data_items
    return context.with_model(
        SweepCalItem(
            type="sweep",
            raw_obs=cal.raw_obs,
            data_items=(
                SweepCalDataItem(
                    interface=item.interface,
                    block=item.block,
                    model=KidsModelBinding(
                        product=index.product.reference,
                        producer="kidscpp.fit",
                        name="model",
                    ),
                ),
            ),
        ),
        interface="toltec0",
        block=0,
    )


@pytest.mark.parametrize(
    "ref",
    [
        DataProdRef(uid="tcs-145609-0-1_proc"),
        DataProdRef(uid="tcs-145609-0-1_proc/main", namespace_uid=uuid4()),
        DataProdRef(uid=f"tcs-145609-0-1_proc/{uuid4()}"),
    ],
)
def test_model_requires_namespace_and_realization(ref):
    with pytest.raises(ValidationError):
        KidsModelBinding(product=ref, producer="kidscpp.fit", name="model")


@pytest.mark.parametrize("failed", [False, True])
def test_saved_run_pins_model_when_tag_moves(raw_context, tmp_path, failed):
    context, raw = raw_context
    original = context.model_dump_json()
    store = DataProdStore(tmp_path)
    first = _publish_model(store, "fit-first")
    selected = _bind(context, first)
    inputs = prepare_run_inputs(
        [selected], raw_products=raw, model_indexes={first.product.reference: first}
    )
    outcome = (
        pytest.raises(RuntimeError, match="consumer failed")
        if failed
        else nullcontext()
    )
    with (
        ExitStack() as _run_records,
        outcome,
        WorkflowRun.create(
            context={"inputs": inputs, "workflow": "test-consumer"}
        ) as run,
    ):
        _run_records.callback(store.write_run, run)
        store.write_run(run)
        assert read_run_inputs(store.read_run(run.uid)) == (selected,)
        if failed:
            msg = "consumer failed"
            raise RuntimeError(msg)
    second = _publish_model(store, "fit-second")
    assert first.product.reference != second.product.reference
    current = DataProdIndex.read(
        resolve_index(store.get_product_path("tcs-145609-0-1", "dp_raw_obs_proc"))
    )
    assert current.product.reference == second.product.reference
    inputs[0]["data_items"].clear()
    saved = store.read_run(run.uid)
    assert saved.status == ("failed" if failed else "completed")
    assert read_run_inputs(saved) == (selected,)
    assert saved.context["workflow"] == "test-consumer"
    assert context.model_dump_json() == original
    alternative = _bind(context, second)
    assert alternative != selected
    assert (
        alternative.data_items[0].cal_items[0].raw_obs
        == selected.data_items[0].cal_items[0].raw_obs
    )
    assert (
        prepare_run_inputs(
            [read_run_inputs(saved)[0]],
            raw_products=raw,
            model_indexes={first.product.reference: first},
        )
        == saved.context["inputs"]
    )
    with pytest.raises(KeyError):
        prepare_run_inputs(
            [selected],
            raw_products=raw,
            model_indexes={second.product.reference: second},
        )


@pytest.mark.parametrize(
    "change", ["raw", "block", "interface", "run", "namespace", "duplicate", "primary"]
)
def test_model_index_must_match_recorded_sweep(raw_context, tmp_path, change):
    context, raw = raw_context
    index = _publish_model(DataProdStore(tmp_path), "fit")
    selected = _bind(context, index)
    altered = index.model_copy(deep=True)
    [artifact] = [
        item for item in altered.product.data_items if item.meta.get("name") == "model"
    ]
    if change == "raw":
        altered.product.meta.obsnum += 1
    elif change == "primary":
        altered.product.assocs[0].uid = "tcs-145608-0-1"
    elif change == "block":
        artifact.meta["scope"] = "blocks/0"
    elif change == "interface":
        artifact.meta["interface"] = "toltec1"
    elif change == "run":
        artifact.meta["run_uid"] = str(uuid4())
    elif change == "namespace":
        altered.product.namespace_uid = uuid4()
    elif change == "duplicate":
        duplicate = artifact.model_copy(deep=True)
        duplicate.source = type(duplicate.source).model_validate(
            {**duplicate.source.model_dump(round_trip=True), "path": "another.ecsv"}
        )
        altered.product.data_items.append(duplicate)
    with pytest.raises(ValueError):
        prepare_run_inputs(
            [selected],
            raw_products=raw,
            model_indexes={index.product.reference: altered},
        )


def test_unresolved_and_changed_calibrations_cannot_be_saved(raw_context, tmp_path):
    context, raw = raw_context
    with pytest.raises(ValueError, match="exact model"):
        prepare_run_inputs([context], raw_products=raw, model_indexes={})
    model = _publish_model(DataProdStore(tmp_path), "fit")
    selected = _bind(context, model)
    [cal] = selected.data_items[0].cal_items
    wrong_block = cal.model_copy(
        update={"data_items": (cal.data_items[0].model_copy(update={"block": 0}),)}
    )
    with pytest.raises(ValueError, match="changes the recorded"):
        context.with_model(wrong_block, interface="toltec0", block=0)
    with pytest.raises(ValueError, match="not selected"):
        context.with_model(cal, interface="toltec0", block=1)
    missing = context.model_copy(
        update={"data_items": (RawObsDataItem(interface="toltec0", block=0),)}
    )
    with pytest.raises(ValueError, match="recorded block calibration"):
        prepare_run_inputs([missing], raw_products=raw, model_indexes={})
    with pytest.raises(ValueError, match="At least one"):
        prepare_run_inputs([], raw_products=raw, model_indexes={})


def test_local_model_choice_preserves_other_common_networks(
    tmp_path,
):
    cal = SweepCalItem(
        type="sweep",
        raw_obs=DataProdRef(uid="tcs-145609-0-1"),
        data_items=tuple(
            SweepCalDataItem(interface=f"toltec{i}", block=1) for i in (0, 1)
        ),
    )
    context = RawObsContext(
        raw_obs=DataProdRef(uid="tcs-145609-0-2"),
        data_items=(
            RawObsDataItem(interface="toltec0", block=0),
            RawObsDataItem(interface="toltec1", block=0),
            RawObsDataItem(interface="tel"),
        ),
        cal_items=(cal,),
    )
    first = _publish_model(DataProdStore(tmp_path), "fit")
    fitted = SweepCalItem(
        type="sweep",
        raw_obs=cal.raw_obs,
        data_items=(
            cal.data_items[0].model_copy(
                update={
                    "model": KidsModelBinding(
                        product=first.product.reference,
                        producer="kidscpp.fit",
                        name="model",
                    )
                }
            ),
        ),
    )
    result = context.with_model(fitted, interface="toltec0", block=0)
    assert not result.cal_items
    assert result.data_items[0].cal_items == (fitted,)
    kept = result.data_items[1].cal_items[0]
    assert isinstance(kept, SweepCalItem)
    assert kept.data_items == (cal.data_items[1],)
    assert not result.data_items[2].cal_items
    assert context.cal_items == (cal,)
