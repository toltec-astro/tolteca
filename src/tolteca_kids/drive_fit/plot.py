"""Plotly artifacts built from saved drive-fit results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
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
from tollan.plot.plotly import PlotlyViz, make_subplots

from .fit import A_MAX, A_MIN
from .step import DriveFitStep
from .sweep_plot import DriveFitSweepArtifacts

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from .result import DriveFitResult


_ACCEPTED = "#087f5b"
_FLAGGED = "#c25b17"
_CURVE = "#3467a1"


def _select(result, channels):
    data = result.data if channels is None else result.data.sel(chan=channels)
    if not data.sizes["chan"]:
        msg = "Select at least one fitted channel to plot"
        raise ValueError(msg)
    return data


def _counts(data):
    finite = np.isfinite(data.p_best.values)
    accepted = finite & (data.p_flag.values == 0)
    return accepted, finite & ~accepted, ~finite


def _caption(data):
    accepted, flagged, missing = _counts(data)
    return (
        f"{data.attrs['interface']} · {data.sizes['chan']} channels · "
        f"{accepted.sum()} accepted · {flagged.sum()} flagged selections · "
        f"{missing.sum()} without a selection"
    )


class DriveFitViz(PlotlyViz):
    """Network summary of selected drives, following the v2 summary panels.

    Parameters
    ----------
    channels : list of int or None, optional
        Recorded channel IDs to summarize. None includes every fitted channel.
        Histograms and frequency/Qr panels distinguish accepted from flagged
        finite selections. Channels without a selection appear in the counts.
    """

    channels: list[int] | None = None

    def __call__(self, result: DriveFitResult) -> go.Figure:
        """Construct a network summary without repeating fits.

        Parameters
        ----------
        result : DriveFitResult
            Saved selections; keep their storage open until this call returns.

        Returns
        -------
        plotly.graph_objects.Figure
            Drive/nonlinearity histograms and drive versus frequency/Qr panels.
            Drive attenuation is displayed as positive dB, equal to -p_best.
        """
        data = _select(result, self.channels)
        figure = make_subplots(
            2,
            2,
            horizontal_spacing=0.12,
            vertical_spacing=0.20,
            subplot_titles=[
                "Selected drive attenuation",
                "Selected nonlinearity",
                "Drive attenuation versus tone frequency",
                "Drive attenuation versus loaded Qr",
            ],
        )
        accepted, flagged, _ = _counts(data)
        attenuation = -data.p_best.values
        powers = -data.power.values
        drive_bins = np.arange(powers.min(), powers.max() + 0.25, 0.25)
        a_bins = np.linspace(A_MIN, A_MAX, 35)
        for mask, label, color in (
            (accepted, "Accepted", _ACCEPTED),
            (flagged, "Flagged selection", _FLAGGED),
        ):
            for column, values, bins in (
                (1, attenuation, drive_bins),
                (2, data.a_best.values, a_bins),
            ):
                figure.add_bar(
                    x=(bins[:-1] + bins[1:]) / 2,
                    y=np.histogram(values[mask], bins=bins)[0],
                    width=np.diff(bins),
                    marker_color=color,
                    name=label,
                    legendgroup=label,
                    showlegend=column == 1,
                    hovertemplate=(
                        "%{x:.3g}: %{y} channels<extra>%{fullData.name}</extra>"
                    ),
                    row=1,
                    col=column,
                )
            for column, values in (
                (1, data.f_chan.values / 1e6),
                (2, data.Qr_best.values),
            ):
                figure.add_scatter(
                    x=values[mask],
                    y=attenuation[mask],
                    mode="markers",
                    marker={"color": color, "size": 6, "opacity": 0.8},
                    name=label,
                    legendgroup=label,
                    showlegend=False,
                    customdata=data.chan.values[mask],
                    hovertemplate=(
                        "Channel %{customdata}<br>x=%{x:.5g}<br>"
                        "Attenuation=%{y:.3f} dB<extra>%{fullData.name}</extra>"
                    ),
                    row=2,
                    col=column,
                )
        figure.add_vline(
            x=data.attrs["config"]["a_ref"],
            line_dash="dash",
            line_color="#52606d",
            row=1,
            col=2,
        )
        for row, column, xlabel, ylabel in (
            (1, 1, "Drive attenuation (dB)", "Channel count"),
            (1, 2, "Nonlinearity a", "Channel count"),
            (2, 1, "Tone frequency (MHz)", "Drive attenuation (dB)"),
            (2, 2, "Loaded quality factor Qr", "Drive attenuation (dB)"),
        ):
            figure.update_xaxes(title_text=xlabel, row=row, col=column)
            figure.update_yaxes(title_text=ylabel, row=row, col=column)
        figure.update_layout(
            title={"text": "Drive fit summary<br><sup>" + _caption(data) + "</sup>"},
            template="plotly_white",
            height=850,
            barmode="stack",
            legend={"orientation": "h", "y": 1.10, "x": 0},
            margin={"t": 150, "b": 80, "l": 80, "r": 45},
        )
        return figure


class DriveFitGridViz(PlotlyViz):
    """Paginated channel curves with one channel per panel.

    Parameters
    ----------
    channels : list of int or None, optional
        Recorded IDs in requested order. None includes every fitted channel.
    page_size : int, optional
        Channels per page, default 12. The final page clears unused panels.
    columns : int, optional
        Grid columns, default 4.

    Notes
    -----
    Curves retain completed fits, including nonlinearity outliers. The initial
    vertical range covers the drive-selection interval; autoscale reveals all
    values. Failed fits leave gaps. Titles report excluded and failed counts;
    hover shows Qr and the separate phase-quality flag. The page selector uses
    channel IDs, including IDs that are noncontiguous or out of order.
    """

    channels: list[int] | None = None
    page_size: int = Field(default=12, ge=1)
    columns: int = Field(default=4, ge=1)

    def __call__(self, result: DriveFitResult) -> go.Figure:
        """Construct channel pages using saved scientific arrays.

        Parameters
        ----------
        result : DriveFitResult
            Fit arrays; keep their storage open until construction finishes.

        Returns
        -------
        plotly.graph_objects.Figure
            Self-contained paginated grid with selected-point markers and a_ref
            guides. No raw reads or fitting occur, including when changing pages.
        """
        data = _select(result, self.channels)[
            [
                "a",
                "Qr",
                "fit_flag",
                "phase_flag",
                "p_best",
                "a_best",
                "p_flag",
                "f_chan",
            ]
        ].load()
        size = min(self.page_size, data.sizes["chan"])
        columns = min(self.columns, size)
        rows = (size + columns - 1) // columns
        figure = make_subplots(
            rows,
            columns,
            horizontal_spacing=0.065,
            vertical_spacing=min(0.13, 0.5 / rows),
        )
        a_ref = data.attrs["config"]["a_ref"]
        figure.update_xaxes(
            title_text="Negative attenuation (dB)",
            range=[data.power.min().item() - 0.5, data.power.max().item() + 0.5],
            autorange=False,
        )
        figure.update_yaxes(
            title_text="a",
            range=[min(A_MIN, a_ref) - 0.05, max(A_MAX, a_ref) + 0.05],
            autorange=False,
        )
        frames = []
        for start in range(0, data.sizes["chan"], size):
            traces, annotations = _channel_page(
                data, start, size, figure.layout.annotations
            )
            frames.append(
                go.Frame(
                    name=str(start),
                    data=traces,
                    layout={"annotations": annotations},
                    traces=list(range(len(traces))),
                )
            )
        figure.add_traces(frames[0].data)
        figure.update_layout(annotations=frames[0].layout.annotations)
        if len(frames) > 1:
            figure.frames = frames
            ids = data.chan.values
            figure.update_layout(
                sliders=[
                    {
                        "currentvalue": {
                            "prefix": "Channel page: ",
                            "font": {"size": 14},
                        },
                        "pad": {"t": 45},
                        "ticklen": 3,
                        "steps": [
                            {
                                "label": (
                                    f"{ids[start]} to "
                                    f"{ids[min(start + size, len(ids)) - 1]}"
                                ),
                                "method": "animate",
                                "args": [
                                    [str(start)],
                                    {
                                        "mode": "immediate",
                                        "frame": {"duration": 0, "redraw": True},
                                        "transition": {"duration": 0},
                                    },
                                ],
                            }
                            for start in range(0, len(ids), size)
                        ],
                    }
                ]
            )
        figure.update_layout(
            title={
                "text": "Drive fit · channel details<br><sup>"
                + _caption(data)
                + "<br>y-axis starts at the selection interval; "
                "autoscale for outliers</sup>"
            },
            template="plotly_white",
            showlegend=False,
            height=270 * rows + 190,
            margin={"t": 130, "b": 110, "l": 65, "r": 35},
            meta={"channels": data.chan.values.tolist(), "page_size": size},
        )
        return figure


def _channel_page(data, start, size, titles):
    traces, annotations = [], []
    power = data.power.values
    for offset in range(size):
        suffix = str(offset + 1) if offset else ""
        axes = {"xaxis": "x" + suffix, "yaxis": "y" + suffix}
        title = titles[offset].to_plotly_json()
        if start + offset >= data.sizes["chan"]:
            traces.extend(go.Scatter(x=[], y=[], **axes) for _ in range(3))
            title["text"] = ""
        else:
            row = data.isel(chan=start + offset)
            complete = row.fit_flag.values == 0
            a = np.where(complete, row.a.values, np.nan)
            excluded = complete & ((a < A_MIN) | (a > A_MAX) | ~np.isfinite(a))
            selected = np.isfinite(row.p_best.item())
            accepted = selected and row.p_flag.item() == 0
            status = (
                "accepted" if accepted else "flagged" if selected else "no selection"
            )
            title["text"] = (
                f"Ch {row.chan.item()} · {row.f_chan.item() / 1e6:.1f} MHz · {status}"
                f"<br><sup>{(~complete).sum()} failed · "
                f"{excluded.sum()} excluded a</sup>"
            )
            title["font"] = {"size": 12}
            traces.extend(
                [
                    go.Scatter(
                        x=power,
                        y=a,
                        mode="lines+markers",
                        connectgaps=False,
                        line={"color": _CURVE, "width": 1.5},
                        marker={"size": 4},
                        customdata=np.column_stack(
                            (row.Qr.values, row.phase_flag.values)
                        ),
                        hovertemplate=(
                            "Negative attenuation=%{x:.2f} dB<br>a=%{y:.5g}<br>"
                            "Qr=%{customdata[0]:.5g}<br>"
                            "phase_flag=%{customdata[1]}<extra></extra>"
                        ),
                        **axes,
                    ),
                    go.Scatter(
                        x=[row.p_best.item()] if selected else [],
                        y=[row.a_best.item()] if selected else [],
                        mode="markers",
                        marker={
                            "symbol": "x",
                            "size": 11,
                            "color": _ACCEPTED if accepted else _FLAGGED,
                        },
                        hovertemplate=(
                            f"{status}<br>Negative attenuation=%{{x:.3f}} dB"
                            "<br>a=%{y:.4g}<extra></extra>"
                        ),
                        **axes,
                    ),
                    go.Scatter(
                        x=[power.min(), power.max()],
                        y=[data.attrs["config"]["a_ref"]] * 2,
                        mode="lines",
                        line={"color": "#6b7280", "dash": "dash", "width": 1},
                        hoverinfo="skip",
                        **axes,
                    ),
                ]
            )
        annotations.append(title)
    return traces, annotations


class DriveFitPlotArtifacts(ArtifactSpec[ArtifactContext]):
    """Network summary, channel curves and per-power fit diagnostics."""

    summary: DriveFitViz = Field(default_factory=DriveFitViz)
    grid: DriveFitGridViz = Field(default_factory=DriveFitGridViz)
    sweeps: DriveFitSweepArtifacts = Field(default_factory=DriveFitSweepArtifacts)

    def __call__(self, result: DriveFitResult) -> ArtifactContext:
        """Prepare the configured summary, channel grid and sweep artifacts.

        Parameters
        ----------
        result : DriveFitResult
            Saved or newly computed fit arrays.

        Returns
        -------
        ArtifactContext
            Loaded figures paired with their effective specifications.
        """
        artifacts = ArtifactContext()
        artifacts.add(self.summary, self.summary(result))
        artifacts.add(self.grid, self.grid(result))
        artifacts.add(self.sweeps, self.sweeps(result))
        return artifacts

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """Export the retained figures in the configured formats.

        Parameters
        ----------
        data : ArtifactContext
            Previously constructed plots.
        path : pathlib.Path
            Caller-owned output directory.

        Returns
        -------
        list of ArtifactFile
            Summary, channel grid, sweep browser and linked channel assets.
        """
        return [
            receipt
            for artifact in data.values()
            for receipt in artifact.spec.write(
                artifact.data,
                path=(
                    path
                    if isinstance(artifact.spec, DriveFitSweepArtifacts)
                    else path
                    / (
                        "drive_fit_grid"
                        if isinstance(artifact.spec, DriveFitGridViz)
                        else "drive_fit_summary"
                    )
                ),
            )
        ]


class DriveFitPlotStepConfig(StepConfig):
    """Plot settings for a saved drive fit.

    Parameters
    ----------
    artifacts : DriveFitPlotArtifacts, optional
        Figure construction and export settings.
    fit_alias : str or None, optional
        Fit result to plot; None selects the primary result.
    """

    model_config = ConfigDict(extra="forbid")
    artifacts: DriveFitPlotArtifacts = Field(default_factory=DriveFitPlotArtifacts)
    fit_alias: str | None = None


class DriveFitPlotStepContext(StepContext["DriveFitPlotStep", DriveFitPlotStepConfig]):
    """Plot invocation record and retained figure artifacts."""


class DriveFitPlotStep(Step[DriveFitPlotStepConfig, DriveFitPlotStepContext]):
    """Prepare diagnostics from a live or reopened fit result."""

    @classmethod
    def run(cls, data: xr.DataTree, context: DriveFitPlotStepContext) -> None:
        """Construct plots independently of raw inputs and live fit contexts.

        Parameters
        ----------
        data : xarray.DataTree
            Interface tree containing the requested fit result.
        context : DriveFitPlotStepContext
            Figure settings and destination for retained artifacts.

        Returns
        -------
        None
            Missing results record a failure; successful plots remain on context.
        """
        path = DriveFitStep.result_path(alias=context.config.fit_alias)
        if path not in data.children:
            context.failures.append(
                StepFailure(
                    kind=StepFailureKind.missing_input,
                    message=f"Drive-fit plotting requires the result {path!r}.",
                )
            )
            return
        context.artifacts = context.config.artifacts(
            DriveFitStep.view(data, alias=context.config.fit_alias)
        )
