"""Plotly diagnostics for a completed sweep check.

Construct a figure directly from a ``SweepCheckView``::

    figure = S21FrequencyGridViz(page_size=10)(check, channels=[0, 1, 2])

Use the returned figure in ``dcc.Graph`` or call the configured Viz's ``write``
or ``show`` method. Construction loads the requested plotted values; keep the
source open until it returns. The figure can then outlive the source. Channel
selection limits channel arrays; unified D21 diagnostics still cover the block.

``SweepCheckArtifacts`` constructs all available figures and retains each with its
configuration for typed lookup and later export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from pydantic import Field
from tollan.pipeline import ArtifactContext, ArtifactFile, ArtifactSpec
from tollan.plot.plotly import (
    PlotlyFigExport,
    PlotlyFigExportFormat,
    PlotlyFigShowInDash,
    PlotlyViz,
    SubplotGrid,
    make_range,
    make_subplot_layout,
    make_subplots,
    update_subplot_layout,
)

from ._plot import COLOR_PALETTE, FIG_LAYOUT
from .readout_power_viz import RoachReadoutPowerViz
from .sweep import ReducedSweepView
from .sweep_check import SweepBitMask

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    from .sweep_check_view import SweepCheckView

__all__ = [
    "ChannelSummaryViz",
    "IQGridViz",
    "RoachReadoutPowerViz",
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


def _array(view, field, channels=None, *, unit=None):
    """Load one mapped statistic, selecting its channel axis when requested."""
    arr = view.mapper.get_arr(view.data_source, field)
    if field is view.mapper.schema.drive_power:
        arr = arr.sel(element="drive_atten")
    if channels is not None:
        arr = arr.isel({view.mapper.get_name(view.mapper.schema.chan_dim): channels})
    return (arr.u.to(unit) if unit is not None else arr).values


def _channel_summary(view, channels, *, height):
    """Channel flags, scalar statistics and chunk diagnostics on shared axes."""
    schema = view.mapper.schema
    channel_axis = view.sweep.mapper.get_arr(
        view.sweep.data_source, view.sweep.mapper.schema.chan
    ).values
    selected = list(range(len(channel_axis))) if channels is None else list(channels)
    ids = channel_axis[selected]
    flags = _array(view, schema.bitmask_chan, selected)
    grid = SubplotGrid(fig_layout=FIG_LAYOUT)
    flag_fig = make_subplots(1, 1, fig_layout=FIG_LAYOUT)
    flag_fig.add_heatmap(
        x=ids,
        z=np.vstack([(flags & flag) > 0 for flag in SweepBitMask]).astype(int),
        y=[flag.name for flag in SweepBitMask],
        colorscale="rdylgn_r",
        zmin=0,
        zmax=1,
    )
    grid.add_subplot(1, 1, fig=flag_fig, row_height=0.5)
    channel_stats = [
        ("Channel Level", schema.chan_level, 0.15, {}),
        ("Channel Range", schema.chan_range, 0.15, {}),
        ("Channel RMS", schema.chan_rms_mean, 0.15, {}),
    ]
    if view.readout_power is not None:
        channel_stats.extend(
            [
                ("Drive Atten", schema.tone_amps_db, 0.15, {}),
                ("Drive Power (dbm)", schema.drive_power, 0.15, {}),
            ]
        )
    for name, field, row_height, trace_kw in [
        *channel_stats,
        ("Chunk RMS", schema.chunk_rms_mean, 0.25, {}),
        ("Chunk Skew", schema.chunk_skew, 0.25, {"zmin": 0, "zmax": 2}),
        ("Chunk Kurtosis", schema.chunk_kurtosis, 0.25, {"zmin": -2, "zmax": 2}),
        ("Chunk Baseline", schema.mask_chunk_baseline, 0.25, {}),
    ]:
        values = _array(view, field, selected)
        fig = make_subplots(1, 1, fig_layout=FIG_LAYOUT)
        if values.ndim == 1:
            z, y = values[None, :], [name]
        else:
            z = values.T
            center = (len(z) - 1) // 2
            y = [str(i) if i != center else f"{name} {i}" for i in range(len(z))]
            fig.update_yaxes(tickvals=[center])
        fig.add_heatmap(
            x=ids, y=y, z=z.astype(float), colorscale="rdylgn_r", **trace_kw
        )
        grid.add_subplot(grid.shape[0] + 1, 1, fig=fig, row_height=row_height)
    fig = make_subplots(1, 1, fig_layout=FIG_LAYOUT)
    f = view.sweep.f_chan.isel({view.sweep.f_chan.dims[0]: selected}).u.to("MHz").values
    fig.add_scatter(
        x=ids,
        y=f,
        mode="markers",
        marker={"size": 4, "color": "blue"},
        name="Chan Freq",
    )
    fig.update_xaxes(title="Channel ID")
    fig.update_yaxes(title="Chan Frequency (MHz)")
    grid.add_subplot(grid.shape[0] + 1, 1, fig=fig, row_height=0.5)
    fig = grid.make_figure(
        shared_xaxes="all",
        vertical_spacing=40 / height,
        fig_layout={"height": height},
    )
    fig.update_xaxes(
        rangeslider={
            "autorange": True,
            "range": [0, len(channel_axis)],
            "thickness": 0.05,
        },
        row=grid.shape[0],
        col=1,
    )
    return fig


def _s21_frequency(view, channels, *, height, sample_stride):
    """S21 uncertainties, D21 baseline and chunk diagnostics."""
    selected, sweep, ids = _channels(view, channels)
    schema = view.mapper.schema
    signal = sweep.I.values + 1j * sweep.Q.values
    uncertainty = sweep.unc_I.values + 1j * sweep.unc_Q.values
    amplitude = 20 * np.log10(np.abs(signal))
    rms = 20 * np.log10(np.e) * np.abs(uncertainty / signal)
    frequency = sweep.frequency.u.to("MHz").values
    baseline = _array(view, schema.mask_baseline, selected)
    chunks_f = _array(view, schema.f_chunks, selected, unit="MHz")
    fig = make_subplots(
        4,
        1,
        shared_xaxes="all",
        vertical_spacing=40 / height,
        row_heights=[0.5, 0.5, 1, 1],
        fig_layout=FIG_LAYOUT | {"showlegend": False, "height": height},
    )
    chunk_stats = [
        (field, label, _array(view, field, selected))
        for field, label in (
            (schema.chunk_skew, "Chunk Skew"),
            (schema.chunk_kurtosis, "Chunk Kurtosis"),
            (schema.mask_chunk_baseline, "Chunk Baseline"),
        )
    ]
    colors = COLOR_PALETTE.cycle_alternated(1, 0.25, 0.5, 0.25)
    for i, channel in enumerate(ids):
        stride = slice(None, None, sample_stride)
        color, error_color = next(colors), next(colors)
        fig.add_scattergl(
            x=frequency[i, stride],
            y=amplitude[i, stride],
            mode="markers",
            error_y={
                "type": "data",
                "array": rms[i, stride],
                "width": 0,
                "color": error_color,
            },
            marker={
                "size": 4,
                "color": [color if m else "black" for m in baseline[i, stride]],
            },
            name=f"Channel {channel}",
            row=3,
            col=1,
        )
        for field, label, values in chunk_stats:
            y = values[i].astype(float)
            line = {"shape": "hvh"}
            if field is schema.mask_chunk_baseline:
                y = y - (3 + 3 * (i % 2))
                line["color"] = "black"
            fig.add_scattergl(
                x=chunks_f[i],
                y=y,
                mode="lines+markers",
                line=line,
                marker={"size": 4},
                name=label,
                row=4,
                col=1,
            )
    f = view.d21_frequency.to_value("MHz")
    for y, label in ((view.d21, "D21"), (view.d21_baseline, "D21 baseline")):
        fig.add_scatter(
            x=f, y=y.to_value("1/Hz"), mode="lines", name=label, row=1, col=1
        )
    fig.add_scatter(
        x=view.d21_f_chunks.to_value("MHz"),
        y=view.d21_chunk_baseline.to_value("1/Hz"),
        mode="markers",
        marker={"color": "black", "size": 4},
        name="D21 chunk baseline",
        error_y={
            "type": "data",
            "array": view.d21_chunk_baseline_rms.to_value("1/Hz"),
            "width": 0,
            "color": "black",
        },
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=f,
        y=view.d21_mask_baseline.astype(int),
        mode="lines",
        name="d21 baseline mask",
        line={"shape": "hvh"},
        showlegend=True,
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Frequency (MHz)", row=3, col=1)
    fig.update_yaxes(title_text="|S21| (dB)", row=3, col=1)
    fig.update_yaxes(
        title_text="|D21| (Hz^(-1))", autorange=False, range=[-1, 5], row=1, col=1
    )
    return fig


def _make_paged_figure(fig, make_panels, pages, *, name):
    """Install initial panels and frame-specific data and axes."""
    for traces, layout, position in make_panels(0, init=True):
        for trace in traces:
            fig.add_trace(trace, **position)
        update_subplot_layout(fig, layout, **position)
    frames, steps = [], []
    for frame_name, start in pages:
        data, frame_layout = [], {"annotations": []}
        for traces, layout, position in make_panels(start, init=False):
            data.extend(traces)
            frame_layout.update(make_subplot_layout(fig, layout, **position))
        frames.append(
            go.Frame(
                name=frame_name,
                data=data,
                traces=list(range(len(data))),
                layout=frame_layout,
            )
        )
        steps.append(
            {
                "args": [
                    [frame_name],
                    {
                        "frame": {"duration": 30, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 30},
                    },
                ],
                "label": frame_name,
                "method": "animate",
            }
        )
    fig.frames = frames
    fig.update_layout(
        title={"text": name},
        margin={"t": 40},
        showlegend=False,
        sliders=[{"pad": {"t": 40, "b": 20}, "steps": steps}],
    )
    return fig


def _page_size(page_size, ids):
    """Limit a page to the selected channels."""
    if not len(ids):
        msg = "Select at least one channel"
        raise ValueError(msg)
    return min(page_size, len(ids))


def _amplitude_grid(view, channels, *, page_size, columns, height):
    """Amplitude panels with per-channel ranges and half-dB grid lines."""
    _, sweep, ids = _channels(view, channels)
    signal = sweep.I.values + 1j * sweep.Q.values
    uncertainty = sweep.unc_I.values + 1j * sweep.unc_Q.values
    frequency = sweep.frequency.u.to("MHz").values
    amplitude = 20 * np.log10(np.abs(signal))
    rms = 20 * np.log10(np.e) * np.abs(uncertainty / signal)
    page_size = _page_size(page_size, ids)
    columns = min(columns, page_size)
    rows = (page_size + columns - 1) // columns
    fig = make_subplots(
        rows,
        columns,
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
        fig_layout=FIG_LAYOUT | {"width": 1200, "height": height or rows * 240},
    )
    colors = COLOR_PALETTE.cycles(1, 0.5)

    def panels(start, *, init):
        for slot in range(rows * columns):
            row, col = divmod(slot, columns)
            index = start + slot
            present = slot < page_size and index < len(ids)
            color, error_color = next(colors) if present else (None, None)
            x, y, ey = (
                (frequency[index], amplitude[index], rms[index])
                if present
                else ([], [], [])
            )
            xrange = make_range(x) if present else [0, 1]
            yrange = make_range(y) if present else [0, 1]
            trace: dict[str, Any] = {
                "type": "scatter",
                "x": x,
                "y": y,
                "error_y": {
                    "type": "data",
                    "array": ey,
                    "width": 0,
                    "color": error_color,
                },
                "name": f"Channel {ids[index]}" if present else "",
                "marker": {"color": color},
            }
            layout: dict[str, Any] = {
                "xaxis": {"range": xrange},
                "yaxis": {"range": yrange},
            }
            if init:
                trace["mode"] = "markers"
                trace["marker"] = {"color": color, "size": 4}
                if row == rows - 1 and col == 0:
                    layout["xaxis"] = {
                        "range": xrange,
                        "title": {"text": "Frequency (MHz)"},
                    }
                    layout["yaxis"] = {"range": yrange, "title": {"text": "S21 (dB)"}}
                ticks = np.arange(
                    int(yrange[0] * 2) / 2 - 20, int(yrange[1] * 2) / 2 + 20, 0.5
                )
                layout["yaxis"] = layout["yaxis"] | {
                    "showgrid": True,
                    "gridcolor": "#dddddd",
                    "tickvals": ticks.tolist(),
                    "ticktext": [
                        f"{v:.0f}" if int(v * 2) % 2 == 0 else "" for v in ticks
                    ],
                }
            yield [trace], layout, {"row": row + 1, "col": col + 1}

    return _make_paged_figure(
        fig,
        panels,
        ((str(start), start) for start in range(0, len(ids), page_size)),
        name="S21-f",
    )


def _iq_grid(view, channels, *, page_size, height):
    """Overlaid IQ channels with equal axis spans and individual uncertainties."""
    _, sweep, ids = _channels(view, channels)
    i_data, q_data = sweep.I.values, sweep.Q.values
    i_unc, q_unc = sweep.unc_I.values, sweep.unc_Q.values
    page_size = _page_size(page_size, ids)
    fig = make_subplots(
        1,
        1,
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
        fig_layout=FIG_LAYOUT | {"width": 1200, "height": height},
    )
    colors = COLOR_PALETTE.cycles(1, 0.5)

    def panels(start, *, init):
        traces = []
        for index in range(start, start + page_size):
            present = index < len(ids)
            color, error_color = next(colors)
            trace: dict[str, Any] = {
                "type": "scatter",
                "x": i_data[index] if present else [],
                "y": q_data[index] if present else [],
                "error_x": {
                    "type": "data",
                    "array": i_unc[index] if present else [],
                    "width": 0,
                    "color": error_color,
                },
                "error_y": {
                    "type": "data",
                    "array": q_unc[index] if present else [],
                    "width": 0,
                    "color": error_color,
                },
                "name": f"Channel {ids[index]}" if present else "",
                "marker": {"color": color},
            }
            if init:
                trace["mode"] = "markers"
                trace["marker"] = {"color": color, "size": 4}
            traces.append(trace)
        xr = np.asarray(make_range(i_data[start : start + page_size]))
        yr = np.asarray(make_range(q_data[start : start + page_size]))
        if np.ptp(xr) > np.ptp(yr):
            yr = xr - np.mean(xr) + np.mean(yr)
        else:
            xr = yr - np.mean(yr) + np.mean(xr)
        layout = {"xaxis": {"range": xr}, "yaxis": {"range": yr}}
        if init:
            layout["xaxis"]["title"] = {"text": "I (adu)"}
            layout["yaxis"].update(
                title={"text": "Q (adu)"}, scaleanchor="x", scaleratio=1
            )
        yield traces, layout, {"row": 1, "col": 1}

    return _make_paged_figure(
        fig,
        panels,
        ((str(i), start) for i, start in enumerate(range(0, len(ids), page_size))),
        name="I-Q",
    )


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
        view : tolteca_kids.sweep_check_view.SweepCheckView
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
        view : tolteca_kids.sweep_check_view.SweepCheckView
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
        Figure height in pixels. None uses 240 pixels per row (1200 for five rows).
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
        view : tolteca_kids.sweep_check_view.SweepCheckView
            Completed check; its source must remain open until this call returns.
        channels : sequence of int or None, optional
            Positional channel indices. None selects all channels. Plotted labels
            use the stored channel IDs.

        Returns
        -------
        plotly.graph_objects.Figure
            Loaded figure suitable for Dash, standalone display or export.
        """
        return _amplitude_grid(
            view,
            channels,
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
        Figure height in pixels, by default 1200.
    """

    page_size: int = Field(default=25, gt=0, description="Channels per page.")
    height: int = Field(default=1200, gt=0, description="Figure height in pixels.")

    def __call__(
        self, view: SweepCheckView, *, channels: Sequence[int] | None = None
    ) -> go.Figure:
        """Build a figure from a completed sweep check.

        Parameters
        ----------
        view : tolteca_kids.sweep_check_view.SweepCheckView
            Completed check; its source must remain open until this call returns.
        channels : sequence of int or None, optional
            Positional channel indices. None selects all channels. Plotted labels
            use the stored channel IDs.

        Returns
        -------
        plotly.graph_objects.Figure
            Loaded figure suitable for Dash, standalone display or export.
        """
        return _iq_grid(
            view,
            channels,
            page_size=self.page_size,
            height=self.height,
        )


class SweepCheckArtifacts(ArtifactSpec[ArtifactContext]):
    """Sweep diagnostics with shared export and local-display settings.

    Configure each named Viz to customize its construction. ``save_formats`` and
    ``save_plotly`` apply to all available figures; each Viz's ``save`` controls whether
    that individual figure is exported. ``save=False`` disables all exports.
    """

    chan_summary: ChannelSummaryViz = Field(default_factory=ChannelSummaryViz)
    s21_f: S21FrequencyViz = Field(default_factory=S21FrequencyViz)
    s21_f_grid: S21FrequencyGridViz = Field(default_factory=S21FrequencyGridViz)
    i_q_grid: IQGridViz = Field(default_factory=IQGridViz)
    roach_tone_power: RoachReadoutPowerViz = Field(default_factory=RoachReadoutPowerViz)
    save: bool = Field(default=True, description="Enable artifact export.")
    save_formats: list[PlotlyFigExportFormat] = Field(
        default=["html"], description="Formats for every figure."
    )
    save_plotly: PlotlyFigExport = Field(default_factory=PlotlyFigExport)
    show_plotly: PlotlyFigShowInDash = Field(default_factory=PlotlyFigShowInDash)

    def __call__(
        self, view: SweepCheckView, *, channels: Sequence[int] | None = None
    ) -> ArtifactContext:
        """Construct and retain the available diagnostic figures.

        Parameters
        ----------
        view : tolteca_kids.sweep_check_view.SweepCheckView
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
            if spec is self.roach_tone_power and view.readout_power is None:
                continue
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
            "roach_tone_power": self.roach_tone_power,
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
            if spec not in data:
                continue
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
            {
                name: data[spec].data
                for name, spec in self._figures().items()
                if spec in data
            },
            title="Sweep check",
        )
