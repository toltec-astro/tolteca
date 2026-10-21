"""Commit diagnostics constructed from persisted matching and settings arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

from plotly.subplots import make_subplots
from pydantic import Field
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

from .step import DriveFitCommitStep

if TYPE_CHECKING:
    from pathlib import Path

    import plotly.graph_objects as go
    import xarray as xr

    from .step import DriveFitCommitResult


class DriveFitCommitViz(PlotlyViz):
    """Amplitude corrections, selected drives and tone matching diagnostics."""

    def __call__(self, result: DriveFitCommitResult) -> go.Figure:
        """Construct a figure from a saved or live commit.

        Parameters
        ----------
        result : DriveFitCommitResult
            Commit arrays with accessible storage until this call returns.

        Returns
        -------
        plotly.graph_objects.Figure
            Loaded diagnostics with no raw-data or matcher dependencies.
        """
        data = result.data.load()
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "Tone amplitude correction",
                "Selected drive attenuation",
                "Tone matching residual",
                "Fixed-total power error after attenuation quantization",
            ),
        )
        for mask, name in (
            (~data.amp_flag.values, "Accepted"),
            (data.amp_flag.values, "Filled"),
        ):
            for row, field in enumerate(
                ("amp_best", "adrv_best", "dist_shifted", "power_error_db"), 1
            ):
                if field not in data:
                    continue
                values = data[field].values[mask]
                if field == "dist_shifted":
                    values = values / 1e3
                fig.add_scatter(
                    x=data.f_chan.values[mask] / 1e6,
                    y=values,
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    showlegend=row == 1,
                    row=row,
                    col=1,
                )
        for row, label in enumerate(
            (
                "Relative amplitude",
                "Attenuation (dB)",
                "Residual (kHz)",
                "Power error (dB)",
            ),
            1,
        ):
            fig.update_yaxes(title_text=label, row=row, col=1)
            fig.update_xaxes(title_text="Target frequency (MHz)", row=row, col=1)
        step = data.attrs["config"]["adrv_ref_step"]
        fig.update_yaxes(
            range=[-0.02 * step, 1.02 * step],
            tickvals=[0, step / 2, step],
            tickformat=".3f",
            row=4,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dot", row=4, col=1)
        fig.update_layout(
            height=1150,
            title=(
                f"{data.attrs['interface']} drive-fit commit · "
                f"global attenuation {data.attrs['atten_drive_global']:g} dB"
            ),
        )
        return fig


class DriveFitCommitPlotArtifacts(ArtifactSpec[ArtifactContext]):
    """Commit summary figure with export settings."""

    summary: DriveFitCommitViz = Field(default_factory=DriveFitCommitViz)

    def __call__(self, result: DriveFitCommitResult) -> ArtifactContext:
        """Prepare a loaded figure artifact.

        Parameters
        ----------
        result : DriveFitCommitResult
            Completed commit arrays.

        Returns
        -------
        ArtifactContext
            Figure paired with its effective specification.
        """
        context = ArtifactContext()
        context.add(self.summary, self.summary(result))
        return context

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """Write the retained commit figure.

        Parameters
        ----------
        data : ArtifactContext
            Constructed plot artifact.
        path : pathlib.Path
            Directory for the drive_fit_commit figure stem.

        Returns
        -------
        list of ArtifactFile
            Exported figure files.
        """
        return [
            receipt
            for artifact in data.values()
            for receipt in artifact.spec.write(
                artifact.data, path=path / "drive_fit_commit"
            )
        ]


class DriveFitCommitPlotStepConfig(StepConfig):
    """Commit diagnostic construction and export settings."""

    artifacts: DriveFitCommitPlotArtifacts = Field(
        default_factory=DriveFitCommitPlotArtifacts
    )


class DriveFitCommitPlotStepContext(
    StepContext["DriveFitCommitPlotStep", DriveFitCommitPlotStepConfig]
):
    """Commit plot execution record and retained figure artifacts."""


class DriveFitCommitPlotStep(
    Step[DriveFitCommitPlotStepConfig, DriveFitCommitPlotStepContext]
):
    """Construct commit plots independently of raw sources and matching."""

    @classmethod
    def run(cls, data: xr.DataTree, context: DriveFitCommitPlotStepContext) -> None:
        """Prepare plot artifacts from saved commit arrays.

        Parameters
        ----------
        data : xarray.DataTree
            Interface containing drive_fit_commit results.
        context : DriveFitCommitPlotStepContext
            Plot settings and destination for artifacts.

        Returns
        -------
        None
            Missing results record a failure without producing artifacts.
        """
        if "drive_fit_commit" not in data.children:
            context.failures.append(
                StepFailure(
                    kind=StepFailureKind.missing_input,
                    message="Commit plotting requires a drive_fit_commit result.",
                )
            )
            return
        context.artifacts = context.config.artifacts(DriveFitCommitStep.view(data))
