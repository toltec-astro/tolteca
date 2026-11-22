#!/usr/bin/env python
"""Recompute post-solving diagnostics from a pinned saved product, without solving."""

import argparse
import copy
import json
import sys
from pathlib import Path

from tollan.utils.fileloc import fileloc

from tolteca_datamodels.data_prod import DataProdIndex, DataProdStore, WorkflowRun
from tolteca_datamodels.raw_obs_context import SweepCalItem
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.model_fit import ModelFit
from tolteca_kids.product import open_kids_product
from tolteca_kids.timestream_diagnostics import TimestreamDiagnostics
from tolteca_kids.timestream_model import prepare_fitted_groups
from tolteca_kids.timestream_plot import TimestreamPlot
from tolteca_kids.timestream_reduce import TimestreamReduce
from tolteca_kids.timestream_solve import TimestreamSolve


def diagnose_product(  # noqa: PLR0913 -- explicit sources and diagnostic controls
    source,
    model,
    output,
    namespace,
    *,
    evidence=None,
    nperseg=1024,
    fit_residual_max=0.05,
):
    """Publish new diagnostics while retaining accepted states and original lineage."""
    source, model = DataProdIndex.read(source), DataProdIndex.read(model)
    if source.run is None:
        msg = "A solved product must have a recorded workflow run"
        raise ValueError(msg)
    context = copy.deepcopy(source.run.context)
    context.update(
        workflow="timestream-diagnostics",
        source_product=source.product.reference.model_dump(mode="json"),
        source_index=str(source.base_path / "index.yaml"),
    )
    store = DataProdStore(output, namespace=namespace)
    with (
        open_kids_product(source) as tree,
        open_kids_product(model) as fitted,
        WorkflowRun.create(context=context) as run,
    ):
        for block in tree.toltec_kids.blocks:
            node = block.tree
            saved = TimestreamSolve.load_context(
                node["tolteca_kids"].attrs[TimestreamSolve.get_context_key()]
            )
            if saved.model is None or saved.model.product != model.product.reference:
                msg = "Diagnostic model differs from the solved product's exact binding"
                raise ValueError(msg)
            cal = SweepCalItem.model_validate(
                node["solved_groups"].attrs["calibration"]
            )
            [target] = cal.data_items
            if target.model != saved.model:
                msg = "Saved solve context and calibration disagree"
                raise ValueError(msg)
            fit_block = fitted.toltec_kids.blocks[target.block]
            view = ModelFit.view(fit_block.tree)
            quality = fit_block.tree[f"{ModelFit.result_path()}/quality"].to_dataset()
            refreshed = prepare_fitted_groups(view, quality)
            previous = node["solved_groups"]
            if set(refreshed["groups"].children) != set(previous["groups"].children):
                msg = "Saved solutions and selected fit have different groups"
                raise ValueError(msg)
            for name, group in refreshed["groups"].children.items():
                if group.attrs["model"] != previous[f"groups/{name}"].attrs["model"]:
                    msg = "Saved solutions and selected fit have different models"
                    raise ValueError(msg)
                refreshed[f"groups/{name}"] = previous[f"groups/{name}"]
            refreshed.attrs.update(previous.attrs)
            artifact = saved.model.resolve(
                model,
                raw_obs=cal.raw_obs,
                interface=target.interface,
                block=target.block,
            )
            refreshed.attrs["calibration_source"] = str(
                fileloc(model.get_source_uri(artifact.source)).path
            )
            node["solved_groups"] = refreshed
            # Reattach the inexpensive signal view; do not rerun TimestreamSolve.
            TimestreamReduce(psd=None)(node)
            TimestreamDiagnostics(
                evidence=evidence, nperseg=nperseg, fit_residual_max=fit_residual_max
            )(node)
            TimestreamPlot()(node)
        output_step = DataProdOutput(
            path=store.path,
            namespace=namespace,
            registration=False,
            run_name={"generator": "counter", "key": "diagnostics"},
            run=run,
        )
        output_step(tree)
        result = DataProdOutput.get_context(tree)
        if not result.completed:
            msg = f"Diagnostics publication failed: {result.failures}"
            raise RuntimeError(msg)
    store.write_run(run)
    assert result.data is not None
    return result.data / "index.yaml"


def main():
    """Accept explicit source/model indexes and a product-store destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--fit_residual_max", type=float, default=0.05)
    args = parser.parse_args()
    result = diagnose_product(
        args.source,
        args.model,
        args.output,
        args.namespace,
        evidence=args.evidence,
        nperseg=args.nperseg,
        fit_residual_max=args.fit_residual_max,
    )
    sys.stdout.write(json.dumps({"index": str(result)}) + "\n")


if __name__ == "__main__":
    main()
