#!/usr/bin/env python
"""Bind a saved Python Tune fit and materialize a consuming acquisition stub."""

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

from tolteca_datamodels.data_prod import (
    DataProdIndex,
    DataProdStore,
    Location,
    WorkflowRun,
    build_raw_obs_batch,
)
from tolteca_datamodels.raw_obs_context import (
    RawObsContext,
    RawObsDataItem,
    prepare_run_inputs,
)
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids.model_binding import prepare_model_binding
from tolteca_reduce.stubs import materialize_model_stubs


def prepare_stubs(tune, timestream, *, model_index, output_path, namespace):
    """Save the recorded calibration binding and its model stub, without fitting."""
    tune, timestream = Path(tune).resolve(), Path(timestream).resolve()
    model = DataProdIndex.read(model_index)
    store = DataProdStore(output_path, namespace=namespace)
    batch = build_raw_obs_batch(
        [tune, timestream],
        locations={"raw": Location(root="/", location_type="local")},
    )
    raw_indexes = {
        index.product.uid: DataProdIndex.read(store.write_index(index))
        for index in batch.items
    }
    products = {
        uid: index.product
        for uid, index in raw_indexes.items()
        if index.product.meta.data_prod_type == "dp_raw_obs"
    }
    with open_toltec_kids(timestream) as raw:
        meta = raw.toltec_kids.meta
        interface, uid = meta.interface, meta.uid_raw_obs
    context = RawObsContext.from_product(
        products[uid],
        data_items=(RawObsDataItem(interface=interface, block=0),),
        cal_products=products,
    )
    [sweep] = context.data_items[0].cal_items
    bound = prepare_model_binding(
        model,
        sweep=sweep,
        source=tune,
        producer="tolteca_kids.model_fit:ModelFit",
        name="model",
    )
    context = context.with_model(bound, interface=interface, block=0)
    indexes = {index.product.reference: index for index in raw_indexes.values()}
    indexes[model.product.reference] = model
    inputs = prepare_run_inputs([context], raw_products=products, model_indexes=indexes)
    with ExitStack() as stack, WorkflowRun.create(context={"inputs": inputs}) as run:
        stack.callback(store.write_run, run)
        context, index = materialize_model_stubs(
            context, indexes=indexes, store=store, run=run
        )
    indexes[index.product.reference] = index
    inputs = prepare_run_inputs(
        [context],
        raw_products=products,
        model_indexes=indexes,
        artifact_indexes=indexes,
    )
    # The stub is an output of preparation and an input to a later consumer.
    # Preserve both records; do not rewrite the preparation run's frozen inputs.
    with WorkflowRun.create(
        context={"workflow": "model-input-check", "inputs": inputs}
    ) as consumer:
        pass
    store.write_run(consumer)
    return {
        "index": str(index.base_path / "index.yaml"),
        "preparation_run_uid": str(run.uid),
        "run_uid": str(consumer.uid),
        "context": context.model_dump(mode="json"),
        "data_item": {
            "meta": {"interface": interface},
            "filepath": str(timestream),
            "stubs": [
                {
                    "filepath": str(application.artifact.resolve_path(index)),
                    "mode": application.mode,
                }
                for application in context.stubs
            ],
        },
    }


def main():
    """Prepare one network's single-block timestream input for Citlali."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tune", type=Path)
    parser.add_argument("timestream", type=Path)
    parser.add_argument("--model_index", type=Path, required=True)
    parser.add_argument(
        "--output_path", type=Path, default=Path("comparison/dataprod_toltec")
    )
    parser.add_argument("--namespace", default="tune-comparison")
    sys.stdout.write(json.dumps(prepare_stubs(**vars(parser.parse_args())), indent=2))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
