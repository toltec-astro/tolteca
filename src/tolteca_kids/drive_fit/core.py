"""Configured drive-fit steps and product publication."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

from tollan.config import ConfigHandler, SubConfigKeyTransformer
from tollan.pipeline import SequentialPipeline, StepFailure, StepFailureKind

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_datamodels.data_prod_store import DataProdStore

from .config import DriveFitConfig
from .input import prepare_drive_fit_inputs
from .output import DriveFitOutputStep
from .plot import DriveFitPlotStep
from .product import make_drive_fit_product_index
from .step import DriveFitStep

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import xarray as xr
    from tollan.utils.fileloc import FileLoc

__all__ = ["DriveFit"]


class DriveFit(
    SubConfigKeyTransformer[Literal["drive_fit"]], ConfigHandler[DriveFitConfig]
):
    """Fit sequences using root ``drive_fit`` configuration and typed steps.

    Parameters
    ----------
    runtime_context : tollan.config.RuntimeContext
        Runtime containing fit, plot and output settings under ``drive_fit``.
        The sibling ``kids`` section is independent.
    """

    def __call__(self, sources: Iterable[str | Path | FileLoc]) -> xr.DataTree:
        """Prepare inputs, fit each network, and run configured plot/output steps.

        Parameters
        ----------
        sources : iterable of str, pathlib.Path or FileLoc
            Native target/tune sweeps for one sequence and acquisition master.
            Each network must have distinct attenuations and consistent channels.

        Returns
        -------
        xarray.DataTree
            Working collection with interface-local step contexts and results.
            DriveFitOutputStep.get_context exposes the path when output is enabled.
            Input resources are closed before returning.
        """
        return self.process(prepare_drive_fit_inputs(sources))

    def process(self, data: xr.DataTree) -> xr.DataTree:
        """Run the workflow on a prepared collection of sequence identities.

        Parameters
        ----------
        data : xarray.DataTree
            Collection returned by prepare_drive_fit_inputs.

        Returns
        -------
        xarray.DataTree
            The same collection, with scientific results and invocation records.
        """
        steps, snapshots = {}, {}
        for interface, tree in data["interfaces"].children.items():
            handler = type(self)(
                self._rc.with_config(
                    validation_context={
                        str(k): v for k, v in tree["inputs"].attrs.items()
                    },
                )
            )
            cfg = handler.config
            if cfg.output != self.config.output:
                msg = "Output settings must agree across the selected interfaces"
                raise ValueError(msg)
            steps[interface] = [DriveFitStep(cfg.fit), DriveFitPlotStep(cfg.plot)]
            snapshots[interface] = cfg.model_dump(mode="json")
        data.attrs["config"] = snapshots
        return _execute_workflow(
            data,
            self._rc,
            self.config.output,
            make_drive_fit_product_index(data),
            steps,
        )


def _execute_workflow(data, runtime, output, index, steps):
    store = DataProdStore(output.path, namespace=output.namespace)
    primary = next(a.uid for a in index.assocs if a.is_primary)
    scope = (
        store.publication(
            index,
            primary_uid=primary,
            run=output.run,
            tag=output.tag,
            runtime_info=runtime.runtime_info,
        )
        if output.enabled
        else nullcontext()
    )
    with scope as pending:
        for interface, pipeline in steps.items():
            tree = data[f"interfaces/{interface}"]
            SequentialPipeline(steps=pipeline)(tree)
            for step in pipeline:
                if (
                    step.config.enabled
                    and not step.get_context(tree, alias=step.config.alias).completed
                ):
                    msg = f"{interface}: {type(step).__name__} did not complete"
                    raise ValueError(msg)
        if output.enabled:
            DriveFitOutputStep(output, publication=pending, index=index)(data)
    if output.enabled and output.registration:
        # Optional backend; file-only workflows do not initialize DPDB.
        from tolteca_db.core import DataProdHandler  # noqa: PLC0415

        context = DriveFitOutputStep.get_context(data)
        path = context.data
        assert path is not None
        published = DataProdIndex.read(path)
        assert published.run is not None
        try:
            DataProdHandler(runtime).index.ingest(
                [
                    path / "index.yaml",
                    path.parent / "index.yaml",
                    store.namespace_path / "namespace.yaml",
                ],
                run_uids=[published.run.uid],
            )
        except Exception as exc:
            context.completed = False
            context.failures.append(
                StepFailure(
                    kind=StepFailureKind.exception,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                )
            )
            raise
        context.registered = True
    return data
