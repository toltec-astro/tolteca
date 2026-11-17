"""Portable calibration evidence and its ownership by raw products."""

import pytest
from pydantic import ValidationError

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdBatch,
    DataProdCalRef,
    DataProdIndex,
    RawObsBlockHistory,
    RawObsFileCalObsFileMeta,
    RawObsHistoryBlock,
)
from tolteca_datamodels.data_prod.metadata import (
    NamedGroupMeta,
    RawObsMeta,
    ScienceMeta,
)


@pytest.fixture
def raw_metadata():
    return RawObsMeta(
        name="raw",
        data_prod_type="dp_raw_obs",
        master="tcs",
        obsnum=145609,
        subobsnum=0,
        scannum=1,
    )


@pytest.fixture
def complete_reference_metadata():
    return RawObsFileCalObsFileMeta(
        cal_master="tcs", cal_obsnum=145608, cal_subobsnum=0, cal_scannum=1
    )


def test_partial_reference_retains_only_known_identity():
    reference = DataProdCalRef(
        interface="toltec0", meta=RawObsFileCalObsFileMeta(cal_obsnum=145608)
    )
    assert reference.meta.cal_obsnum == 145608
    assert (
        reference.meta.cal_master,
        reference.meta.cal_subobsnum,
        reference.meta.cal_scannum,
        reference.meta.block_history,
        reference.meta.uid,
        reference.target_uid,
        reference.audit,
    ) == (None,) * 7


@pytest.mark.parametrize("target_uid", [None, "tcs-145608-0-1"])
def test_complete_reference_can_be_unresolved_or_resolved(
    complete_reference_metadata, target_uid
):
    reference = DataProdCalRef(
        interface="toltec0", meta=complete_reference_metadata, target_uid=target_uid
    )
    assert reference.meta.uid == "tcs-145608-0-1"
    assert reference.target_uid == target_uid


@pytest.mark.parametrize("master", [None, "ics"])
def test_target_requires_a_complete_matching_quartet(master):
    meta = RawObsFileCalObsFileMeta(
        cal_master=master, cal_obsnum=145608, cal_subobsnum=0, cal_scannum=1
    )
    with pytest.raises(ValidationError, match="complete reference quartet"):
        DataProdCalRef(interface="toltec0", meta=meta, target_uid="tcs-145608-0-1")


def test_raw_product_keeps_one_reference_per_interface(raw_metadata):
    first = DataProdCalRef(
        interface="toltec0", meta=RawObsFileCalObsFileMeta(cal_obsnum=145608)
    )
    second = DataProdCalRef(
        interface="toltec1", meta=RawObsFileCalObsFileMeta(cal_obsnum=145607)
    )
    product = DataProd(
        uid="tcs-145609-0-1", meta=raw_metadata, cal_refs=[first, second]
    )
    assert product.cal_refs == [first, second]
    with pytest.raises(ValidationError, match="distinct interfaces"):
        DataProd(uid=product.uid, meta=raw_metadata, cal_refs=[first, first])


def test_nonraw_product_cannot_carry_calibration_evidence(complete_reference_metadata):
    with pytest.raises(ValidationError, match="Only raw observations"):
        DataProd(
            uid="science/result",
            meta=ScienceMeta(name="science", data_prod_type="dp_science"),
            cal_refs=[
                DataProdCalRef(interface="toltec0", meta=complete_reference_metadata)
            ],
        )


@pytest.mark.parametrize(
    ("blocks", "error_type"),
    [((), "too_short"), ([RawObsHistoryBlock(kind="sweep")], "tuple_type")],
)
def test_history_requires_a_nonempty_tuple_in_python(blocks, error_type):
    with pytest.raises(ValidationError) as exc:
        RawObsBlockHistory.model_validate({"initial": "unknown", "blocks": blocks})
    [error] = exc.value.errors()
    assert (error["loc"], error["type"]) == (("blocks",), error_type)


def test_nested_index_and_batch_preserve_partial_reference_and_history(
    tmp_path, raw_metadata
):
    reference = DataProdCalRef(
        interface="toltec0",
        meta=RawObsFileCalObsFileMeta(
            cal_obsnum=145608,
            block_history=RawObsBlockHistory(
                initial="recorded",
                blocks=(
                    RawObsHistoryBlock(kind="sweep"),
                    RawObsHistoryBlock(kind="timestream"),
                ),
            ),
        ),
    )
    raw = DataProdIndex(
        product=DataProd(uid="tcs-145609-0-1", meta=raw_metadata, cal_refs=[reference])
    )
    parent = DataProdIndex(
        product=DataProd(
            uid="b7f65f98-4f32-4e0a-a66a-cd278670f7cc",
            meta=NamedGroupMeta(name="collection", data_prod_type="dp_named_group"),
        ),
        products=[raw],
    )
    restored = DataProdIndex.read(parent.write(tmp_path / "index.yaml"))
    assert restored.products[0].product.cal_refs == [reference]

    batch = DataProdBatch(items=[parent])
    restored_batch = DataProdBatch.model_validate_json(batch.model_dump_json())
    assert restored_batch == batch
