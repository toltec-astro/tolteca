"""Plotly diagnostics for a completed sweep check.

Construct a figure directly from a ``SweepCheckView``::

    figure = S21FrequencyGridViz(page_size=10)(check, channels=[0, 1, 2])

Use the returned figure in ``dcc.Graph`` or call the configured Viz's ``write``
or ``show`` method. Construction loads the requested plotted values; keep the
source open until it returns. The figure can then outlive the source. Channel
selection limits channel arrays; unified D21 diagnostics still cover the block.

``SweepCheckArtifacts`` constructs all four figures and retains each with its
configuration for typed lookup and later export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import Field
from tollan.pipeline import ArtifactContext, ArtifactFile, ArtifactSpec
from tollan.plot.plotly import (
    PlotlyFigExport,
    PlotlyFigExportFormat,
    PlotlyFigShowInDash,
    PlotlyViz,
)

from .sweep import ReducedSweepView
from .sweep_check import SweepBitMask

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .sweep_check_view import SweepCheckView

__all__ = [
    "ChannelSummaryViz",
    "IQGridViz",
    "S21FrequencyGridViz",
    "S21FrequencyViz",
    "SweepCheckArtifacts",
]


def _channels(view, channels):
    """Select channels before loading corrected sweep arrays."""
    sweep = view.sweep
    chan_dim = sweep.f_chan.dims[0]
    count = sweep.I.sizes[chan_dim]
    selected = list(range(count)) if channels is None else list(channels)
    ds = sweep.data_source.isel({chan_dim: selected}).compute()
    loaded = ReducedSweepView(ds)
    return selected, loaded, loaded.mapper.get_arr(ds, loaded.mapper.schema.chan).values


def _array(view, field, channels=None):
    """Load one mapped statistic, selecting its channel axis when requested."""
    arr = view.mapper.get_arr(view.data_source, field)
    if channels is not None:
        arr = arr.isel({view.mapper.get_name(view.mapper.schema.chan_dim): channels})
    return arr.values


def _channel_summary(view, channels, *, height):
    """Production channel flags, levels, RMS and chunk-statistic panels."""
    schema = view.mapper.schema
    channel_axis = view.sweep.mapper.get_arr(
        view.sweep.data_source, view.sweep.mapper.schema.chan
    ).values
    selected = list(range(len(channel_axis))) if channels is None else list(channels)
    ids = channel_axis[selected]
    flags = _array(view, schema.bitmask_chan, selected)
    panels = [
        (
            "Channel flags",
            np.vstack([(flags & flag) > 0 for flag in SweepBitMask]).astype(int),
            [flag.name for flag in SweepBitMask],
        ),
    ]
    for title, field in (
        ("Channel level (dB)", schema.chan_level),
        ("Channel range (dB)", schema.chan_range),
        ("Channel RMS (dB)", schema.chan_rms_mean),
        ("Chunk RMS (dB)", schema.chunk_rms_mean),
        ("Chunk skew", schema.chunk_skew),
        ("Chunk kurtosis", schema.chunk_kurtosis),
        ("Chunk baseline", schema.mask_chunk_baseline),
    ):
        values = _array(view, field, selected)
        z = values[None, :] if values.ndim == 1 else values.T
        panels.append((title, z.astype(float), list(range(z.shape[0]))))
    fig = make_subplots(
        rows=len(panels) + 1,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[p[0] for p in panels] + ["Channel frequency (MHz)"],
        row_heights=[2, 1, 1, 1, 1, 1, 1, 1, 1],
        vertical_spacing=0.035,
    )
    for row, (title, z, y) in enumerate(panels, 1):
        fig.add_trace(
            go.Heatmap(
                x=ids, y=y, z=z, name=title, colorscale="RdYlGn_r", showscale=False
            ),
            row=row,
            col=1,
        )
    f = view.sweep.f_chan.isel({view.sweep.f_chan.dims[0]: selected}).u.to("MHz").values
    fig.add_trace(
        go.Scatter(x=ids, y=f, mode="markers", name="Channel frequency"),
        row=len(panels) + 1,
        col=1,
    )
    fig.update_xaxes(title_text="Channel ID", row=len(panels) + 1, col=1)
    fig.update_layout(
        title="Sweep check · channel summary", height=height, showlegend=False
    )
    return fig


def _s21_frequency(view, channels, *, height, sample_stride):
    """Production S21, D21, baseline and chunk-statistic traces."""
    selected, sweep, ids = _channels(view, channels)
    schema = view.mapper.schema
    signal = sweep.I.values + 1j * sweep.Q.values
    uncertainty = sweep.unc_I.values + 1j * sweep.unc_Q.values
    amplitude = 20 * np.log10(np.abs(signal))
    rms = 20 * np.log10(np.e) * np.abs(uncertainty / signal)
    frequency = sweep.frequency.u.to("MHz").values
    baseline = _array(view, schema.mask_baseline, selected)
    chunks_f = _array(view, schema.f_chunks, selected) / 1e6
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "D21 and baseline",
            "D21 baseline mask",
            "Sweep amplitude",
            "Chunk statistics",
        ],
    )
    chunk_stats = [
        (field, label, _array(view, field, selected))
        for field, label in (
            (schema.chunk_skew, "Chunk skew"),
            (schema.chunk_kurtosis, "Chunk kurtosis"),
            (schema.mask_chunk_baseline, "Chunk baseline"),
        )
    ]
    for i, channel in enumerate(ids):
        stride = slice(None, None, sample_stride)
        fig.add_trace(
            go.Scattergl(
                x=frequency[i, stride],
                y=amplitude[i, stride],
                mode="markers",
                error_y={"type": "data", "array": rms[i, stride], "width": 0},
                marker={
                    "size": 4,
                    "color": np.where(baseline[i, stride], "#1971c2", "black"),
                },
                name=f"Channel {channel}",
            ),
            row=3,
            col=1,
        )
        for field, label, values in chunk_stats:
            y = values[i].astype(float)
            if field is schema.mask_chunk_baseline:
                y = y - (3 + 3 * (i % 2))
            fig.add_trace(
                go.Scattergl(
                    x=chunks_f[i],
                    y=y,
                    mode="lines+markers",
                    line={"shape": "hvh"},
                    name=f"{label} · {channel}",
                ),
                row=4,
                col=1,
            )
    f = view.d21_frequency.to_value("MHz")
    for y, label in ((view.d21, "D21"), (view.d21_baseline, "D21 baseline")):
        fig.add_trace(
            go.Scattergl(x=f, y=y.to_value("1/Hz"), mode="lines", name=label),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=view.d21_f_chunks.to_value("MHz"),
            y=view.d21_chunk_baseline.to_value("1/Hz"),
            mode="markers",
            name="D21 chunk baseline",
            error_y={
                "type": "data",
                "array": view.d21_chunk_baseline_rms.to_value("1/Hz"),
                "width": 0,
            },
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=f,
            y=view.d21_mask_baseline.astype(int),
            mode="lines",
            name="D21 baseline mask",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Frequency (MHz)", row=4, col=1)
    fig.update_yaxes(title_text="|S21| (dB)", row=3, col=1)
    fig.update_yaxes(title_text="|D21| (1/Hz)", row=1, col=1)
    fig.update_layout(title="Sweep check · S21 and D21", height=height)
    return fig


def _signal_grid(view, channels, *, iq, page_size, columns, height):  # noqa: PLR0913 -- explicit shared panel settings
    """Page through amplitude panels or overlaid IQ channels with uncertainties."""
    _, sweep, ids = _channels(view, channels)
    signal = sweep.I.values + 1j * sweep.Q.values
    uncertainty = sweep.unc_I.values + 1j * sweep.unc_Q.values
    frequency = sweep.frequency.u.to("MHz").values
    amplitude = 20 * np.log10(np.abs(signal))
    rms = 20 * np.log10(np.e) * np.abs(uncertainty / signal)
    page_size = min(page_size, len(ids))
    if not page_size:
        msg = "Select at least one channel"
        raise ValueError(msg)
    columns = 1 if iq else min(columns, page_size)
    rows = 1 if iq else (page_size + columns - 1) // columns

    def page(start):
        page_ids = ids[start : start + page_size]
        fig = make_subplots(
            rows=rows,
            cols=columns,
            subplot_titles=None if iq else [f"Channel {c}" for c in page_ids],
        )
        for offset in range(page_size):
            index = start + offset
            present = index < len(ids)
            x = (signal[index].real if iq else frequency[index]) if present else []
            y = (signal[index].imag if iq else amplitude[index]) if present else []
            ey = (uncertainty[index].imag if iq else rms[index]) if present else []
            ex = uncertainty[index].real if present and iq else []
            row, col = (0, 0) if iq else divmod(offset, columns)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker={"size": 3},
                    name=f"Channel {ids[index]}" if present else "",
                    showlegend=iq and present,
                    error_y={"type": "data", "array": ey, "width": 0},
                    error_x={"type": "data", "array": ex, "width": 0, "visible": iq},
                ),
                row=row + 1,
                col=col + 1,
            )
        fig.update_xaxes(title_text="I" if iq else "Frequency (MHz)", autorange=True)
        fig.update_yaxes(title_text="Q" if iq else "|S21| (dB)", autorange=True)
        if iq:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    fig = page(0)
    frames = []
    for start in range(page_size, len(ids), page_size):
        p = page(start)
        frames.append(go.Frame(name=str(start), data=p.data, layout=p.layout))
    if frames:
        frames.insert(0, go.Frame(name="0", data=fig.data, layout=fig.layout))
        fig.frames = frames
        fig.update_layout(
            sliders=[
                {
                    "currentvalue": {"prefix": "Channels: "},
                    "steps": [
                        {
                            "label": (
                                f"{ids[start]}-"
                                f"{ids[min(start + page_size, len(ids)) - 1]}"
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
                        for start in range(0, len(ids), page_size)
                    ],
                }
            ]
        )
    fig.update_layout(
        title="Sweep check · IQ" if iq else "Sweep check · amplitude",
        height=height if height is not None else max(350, rows * 260),
    )
    return fig


class ChannelSummaryViz(PlotlyViz):
    """Channel quality flags, levels, RMS and chunk statistics.

    Parameters
    ----------
    height : int, optional
        Figure height in pixels, by default 1400.
    """

    height: int = Field(default=1400, gt=0, description="Figure height in pixels.")

    def __call__(
        self, view: SweepCheckView, *, channels: Sequence[int] | None = None
    ) -> go.Figure:
        """Build a figure from a completed sweep check.

        Parameters
        ----------
        view : SweepCheckView
            Completed check; its source must remain open until this call returns.
        channels : sequence of int or None, optional
            Positional channel indices. None selects all channels. Plotted labels
            use the stored channel IDs.

        Returns
        -------
        plotly.graph_objects.Figure
            Loaded figure suitable for Dash, standalone display or export.
        """
        return _channel_summary(view, channels, height=self.height)


class S21FrequencyViz(PlotlyViz):
    """Sweep amplitudes, uncertainties, D21 and baseline diagnostics.

    Parameters
    ----------
    height : int, optional
        Figure height in pixels, by default 1000.
    sample_stride : int, optional
        Plot every Nth sweep sample, by default 4. D21 and chunk statistics
        retain their full frequency grids.
    """

    height: int = Field(default=1000, gt=0, description="Figure height in pixels.")
    sample_stride: int = Field(
        default=4, gt=0, description="Plot every Nth sweep sample."
    )

    def __call__(
        self, view: SweepCheckView, *, channels: Sequence[int] | None = None
    ) -> go.Figure:
        """Build a figure from a completed sweep check.

        Parameters
        ----------
        view : SweepCheckView
            Completed check; its source must remain open until this call returns.
        channels : sequence of int or None, optional
            Positional channel indices. None selects all channels. Plotted labels
            use the stored channel IDs.

        Returns
        -------
        plotly.graph_objects.Figure
            Loaded figure suitable for Dash, standalone display or export.
        """
        return _s21_frequency(
            view, channels, height=self.height, sample_stride=self.sample_stride
        )


class S21FrequencyGridViz(PlotlyViz):
    """Paged amplitude panels with sweep uncertainties.

    Parameters
    ----------
    page_size : int, optional
        Channels per page, by default 25.
    columns : int, optional
        Maximum panels per row, by default 5.
    height : int or None, optional
        Figure height in pixels. None scales the height with the number of rows.
    """

    page_size: int = Field(default=25, gt=0, description="Channels per page.")
    columns: int = Field(default=5, gt=0, description="Maximum panels per row.")
    height: int | None = Field(
        default=None, gt=0, description="Height in pixels; None scales with panel rows."
    )

    def __call__(
        self, view: SweepCheckView, *, channels: Sequence[int] | None = None
    ) -> go.Figure:
        """Build a figure from a completed sweep check.

        Parameters
        ----------
        view : SweepCheckView
            Completed check; its source must remain open until this call returns.
        channels : sequence of int or None, optional
            Positional channel indices. None selects all channels. Plotted labels
            use the stored channel IDs.

        Returns
        -------
        plotly.graph_objects.Figure
            Loaded figure suitable for Dash, standalone display or export.
        """
        return _signal_grid(
            view,
            channels,
            iq=False,
            page_size=self.page_size,
            columns=self.columns,
            height=self.height,
        )


class IQGridViz(PlotlyViz):
    """Paged IQ overlays with I and Q uncertainties.

    Parameters
    ----------
    page_size : int, optional
        Channels per page, by default 25.
    height : int, optional
        Figure height in pixels, by default 800.
    """

    page_size: int = Field(default=25, gt=0, description="Channels per page.")
    height: int = Field(default=800, gt=0, description="Figure height in pixels.")

    def __call__(
        self, view: SweepCheckView, *, channels: Sequence[int] | None = None
    ) -> go.Figure:
        """Build a figure from a completed sweep check.

        Parameters
        ----------
        view : SweepCheckView
            Completed check; its source must remain open until this call returns.
        channels : sequence of int or None, optional
            Positional channel indices. None selects all channels. Plotted labels
            use the stored channel IDs.

        Returns
        -------
        plotly.graph_objects.Figure
            Loaded figure suitable for Dash, standalone display or export.
        """
        return _signal_grid(
            view,
            channels,
            iq=True,
            page_size=self.page_size,
            columns=1,
            height=self.height,
        )


class SweepCheckArtifacts(ArtifactSpec[ArtifactContext]):
    """Four sweep diagnostics with shared export and local-display settings.

    Configure each named Viz to customize its construction. ``save_formats`` and
    ``save_plotly`` apply to all four figures; each Viz's ``save`` controls whether
    that individual figure is exported. ``save=False`` disables all exports.
    """

    chan_summary: ChannelSummaryViz = Field(default_factory=ChannelSummaryViz)
    s21_f: S21FrequencyViz = Field(default_factory=S21FrequencyViz)
    s21_f_grid: S21FrequencyGridViz = Field(default_factory=S21FrequencyGridViz)
    i_q_grid: IQGridViz = Field(default_factory=IQGridViz)
    save: bool = Field(default=True, description="Enable artifact export.")
    save_formats: list[PlotlyFigExportFormat] = Field(
        default=["html"], description="Formats for every figure."
    )
    save_plotly: PlotlyFigExport = Field(default_factory=PlotlyFigExport)
    show_plotly: PlotlyFigShowInDash = Field(default_factory=PlotlyFigShowInDash)

    def __call__(
        self, view: SweepCheckView, *, channels: Sequence[int] | None = None
    ) -> ArtifactContext:
        """Construct and retain the four diagnostic figures.

        Parameters
        ----------
        view : SweepCheckView
            Completed check with an open source.
        channels : sequence of int or None, optional
            Positional channel indices; None selects all channels.

        Returns
        -------
        tollan.pipeline.ArtifactContext
            Loaded figures with effective Viz settings. Use a Viz class, or its
            configured instance when aliased, to retrieve an entry.
        """
        artifacts = ArtifactContext()
        for spec in self._figures().values():
            effective = spec.model_copy(
                update={
                    "save": self.save and spec.save,
                    "save_formats": self.save_formats,
                    "save_plotly": self.save_plotly,
                    "show_plotly": self.show_plotly,
                }
            )
            artifacts.add(effective, effective(view, channels=channels))
        return artifacts

    def _figures(self):
        return {
            "chan_summary": self.chan_summary,
            "s21_f": self.s21_f,
            "s21_f_grid": self.s21_f_grid,
            "i_q_grid": self.i_q_grid,
        }

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """Export retained figures beneath a caller-supplied directory.

        Parameters
        ----------
        data : tollan.pipeline.ArtifactContext
            Figures returned by this spec. Each retained Viz owns its export
            settings; no scientific input or recomputation is required.
        path : pathlib.Path
            Destination directory. Figure field names form filename stems.

        Returns
        -------
        list of tollan.pipeline.ArtifactFile
            Files written using the retained specs' enabled formats.
        """
        files = []
        for name, spec in self._figures().items():
            artifact = data[spec]
            files.extend(artifact.spec.write(artifact.data, path=path / name))
        return files

    def show(self, data: ArtifactContext) -> None:
        """Display retained figures together in a local Dash app.

        Parameters
        ----------
        data : tollan.pipeline.ArtifactContext
            Figures returned by this spec.

        Returns
        -------
        None
            Returns after the local display server is stopped.
        """
        self.show_plotly(
            {name: data[spec].data for name, spec in self._figures().items()},
            title="Sweep check",
        )
