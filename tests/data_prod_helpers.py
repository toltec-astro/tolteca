"""Shared data-product fixtures built from tolteca_datamodels records."""

from contextlib import ExitStack

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdAssoc,
    DataProdIndex,
    DataProdItem,
    DataProdSource,
    Location,
)
from tolteca_datamodels.data_prod import WorkflowRun as ExecutionRun
from tolteca_datamodels.data_prod.metadata import RawObsProcMeta

__all__ = ["publish_model"]


def publish_model(store, label):
    """Index-only publication fixture; native payload validation is separate."""
    uid = "tcs-145609-0-1"
    index = DataProdIndex(
        product=DataProd(
            uid=f"{uid}_proc",
            meta=RawObsProcMeta(
                data_prod_type="dp_raw_obs_proc",
                name=f"{uid}_proc",
                master="tcs",
                obsnum=145609,
                subobsnum=0,
                scannum=1,
            ),
            assocs=[
                DataProdAssoc(
                    data_prod_assoc_type="dpa_raw_obs_proc_raw_obs",
                    uid=uid,
                    is_primary=True,
                )
            ],
        )
    )
    with (
        ExitStack() as _run_records,
        ExecutionRun.create() as invocation,
        store.materialize(
            index.model_copy(update={"run": invocation}), run_name=label
        ) as out,
    ):
        _run_records.callback(store.write_run, invocation)
        out.locations["self"] = Location(root=".", location_type="local")
        out.product.data_items.append(
            DataProdItem(
                meta={
                    "type": "artifact",
                    "format": "ecsv",
                    "interface": "toltec0",
                    "scope": "blocks/1",
                    "producer": "kidscpp.fit",
                    "name": "model",
                    "run_uid": str(invocation.uid),
                },
                source=DataProdSource.model_validate(
                    {"location": "self", "path": "fit.ecsv"}
                ),
            )
        )
    return DataProdIndex.read(out.base_path)
