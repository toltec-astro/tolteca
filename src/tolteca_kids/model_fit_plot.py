"""Plot step for completed Python ModelFit results and saved diagnostics."""

from pydantic import ConfigDict, Field
from tollan.config.types import AbsAnyPath  # noqa: TC002 -- Pydantic field type.
from tollan.pipeline import (
    ArtifactFile,
    Step,
    StepConfig,
    StepContext,
    StepFailure,
    StepFailureKind,
)
from tollan.plot.plotly import PlotlyViz

from .model_fit import ModelFit
from .model_fit_viz import ModelFitPlotArtifacts


class ModelFitPlotConfig(StepConfig):
    """Model-fit diagnostics using the standard retained-artifact lifecycle.

    Parameters
    ----------
    artifacts : ModelFitPlotArtifacts, optional
        Figure selection and export settings; interactive HTML by default.
    model_fit_alias : str or None, optional
        Saved Python ModelFit invocation to inspect. None selects the primary.
    save_path : pathlib.Path or None, optional
        Optional immediate export directory. Product output also collects the
        retained artifacts. Use separate destinations for different blocks.
    show : bool, optional
        Display locally through the common Plotly viewer, default False.
    """

    model_config = ConfigDict(extra="forbid")
    artifacts: ModelFitPlotArtifacts = Field(default_factory=ModelFitPlotArtifacts)
    model_fit_alias: str | None = None
    save_path: AbsAnyPath | None = None
    show: bool = False


class ModelFitPlotContext(StepContext["ModelFitPlot", ModelFitPlotConfig]):
    """Retained figures, execution record and optional immediate export receipts."""

    files: list[ArtifactFile] = Field(default_factory=list)


class ModelFitPlot(Step[ModelFitPlotConfig, ModelFitPlotContext]):
    """Inspect saved Python fit results without fitting or selecting primaries."""

    @classmethod
    def run(cls, data, context):
        """Retain summary and detailed artifacts for a completed fit result.

        Parameters
        ----------
        data : xarray.DataTree
            One block with a saved Python ModelFit result and quality table.
        context : ModelFitPlotContext
            Plot configuration, retained artifacts and export receipts.

        Returns
        -------
        None
            Missing Python results record a missing-input failure. Configured
            immediate export and display use the retained figures.
        """
        config = context.config
        path = ModelFit.result_path(alias=config.model_fit_alias)
        if path not in data.children or "quality" not in data[path].children:
            context.failures.append(
                StepFailure(
                    kind=StepFailureKind.missing_input,
                    message=(
                        "ModelFitPlot requires a Python ModelFit result with quality "
                        "diagnostics; native kids.fit/1 is not this model contract."
                    ),
                )
            )
            return
        scope = data[path]
        view = ModelFit.view(data, alias=config.model_fit_alias)
        power = scope["power"].to_dataset() if "power" in scope.children else None
        context.artifacts = config.artifacts(view, scope["quality"].to_dataset(), power)
        meta = data.toltec_kids.block.meta
        summary = context.artifacts[PlotlyViz(alias="summary")].data
        summary.update_layout(
            meta={
                **summary.layout.meta,
                "acquisition": (
                    f"toltec{meta.roach} · "
                    f"{meta.obsnum}-{meta.subobsnum}-{meta.scannum} "
                    f"· block {meta.block}"
                ),
            }
        )
        if config.save_path is not None:
            context.files = config.artifacts.write(
                context.artifacts, path=config.save_path
            )
        if config.show:
            config.artifacts.show(context.artifacts)
