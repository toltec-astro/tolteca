"""Configured v2 finding diagnostics and their publication artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field
from tollan.pipeline import (
    ArtifactContext,
    ArtifactFile,
    ArtifactSpec,
    Step,
    StepConfig,
    StepContext,
    StepFailure,
    StepFailureKind,
)
from tollan.plot.plotly import PlotlyViz

from .kids_find import KidsFind
from .kids_plot import make_kids_find_figs
from .sweep_check import SweepCheck

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr


class KidsFindPlotArtifacts(ArtifactSpec[ArtifactContext]):
    """Peak, detection and match diagnostics with shared Plotly export settings."""

    figure: PlotlyViz = Field(default_factory=PlotlyViz)

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """Export the retained finding figures.

        Parameters
        ----------
        data : ArtifactContext
            Named figures produced from the completed finding result.
        path : pathlib.Path
            Product artifact directory.

        Returns
        -------
        list of ArtifactFile
            Export receipts in the configured formats.
        """
        receipts = []
        for artifact in data.values():
            # The plot step assigns each figure its v2 diagnostic name.
            name = artifact.spec.alias
            assert name is not None
            receipts.extend(artifact.spec.write(artifact.data, path=path / name))
        return receipts


class KidsFindPlotConfig(StepConfig):
    """Finding diagnostics; product output exports figures when enabled.

    Parameters
    ----------
    artifacts : KidsFindPlotArtifacts, optional
        Shared figure export settings.
    show : bool, optional
        Display figures locally, default False.
    """

    model_config = ConfigDict(extra="forbid")
    artifacts: KidsFindPlotArtifacts = Field(default_factory=KidsFindPlotArtifacts)
    show: bool = False


class KidsFindPlotContext(StepContext["KidsFindPlot", KidsFindPlotConfig]):
    """Retained figures and execution status."""


class KidsFindPlot(Step[KidsFindPlotConfig, KidsFindPlotContext]):
    """Create v2-style diagnostics for a completed finding step."""

    @classmethod
    def run(cls, data: xr.DataTree, context: KidsFindPlotContext) -> None:
        """Construct figures without rerunning finding or matching.

        Parameters
        ----------
        data : xarray.DataTree
            Block containing saved finding and sweep-check results.
        context : KidsFindPlotContext
            Figure settings and retained artifacts.

        Returns
        -------
        None
            Missing finding results record a missing-input failure.
        """
        if (
            KidsFind.result_path() not in data.children
            or SweepCheck.result_path() not in data.children
        ):
            context.failures.append(
                StepFailure(
                    kind=StepFailureKind.missing_input,
                    message=(
                        "Finding plots require saved finding and sweep-check results."
                    ),
                )
            )
            return
        found = KidsFind.view(data)
        checked = SweepCheck.view(data)
        figures = make_kids_find_figs(checked.sweep.data_source, found)
        context.artifacts = ArtifactContext()
        for name, figure in figures.items():
            context.artifacts.add(
                context.config.artifacts.figure.model_copy(update={"alias": name}),
                figure,
            )
        if context.config.show:
            for figure in figures.values():
                figure.show()
