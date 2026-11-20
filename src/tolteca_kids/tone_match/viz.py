"""Pair and lineage diagnostics from completed tone matches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go
from pydantic import Field
from tollan.plot.plotly import PlotlyViz, make_subplots

from tolteca_kids._plot import COLOR_PALETTE, FIG_LAYOUT

if TYPE_CHECKING:
    from .compute import ToneMatchResult
    from .result import ToneMatchOverview


class ToneMatchViz(PlotlyViz):
    """Frequency correspondence and density panels for a query/reference pair."""

    def __call__(
        self, result: ToneMatchResult, *, query_uid: str, reference_uid: str
    ) -> go.Figure:
        """Build the match and pair-density panels from one completed match.

        Parameters
        ----------
        result : ToneMatchResult
            Completed query/reference match.
        query_uid, reference_uid : str
            Acquisition labels displayed on the diagnostic.

        Returns
        -------
        plotly.graph_objects.Figure
            The two production v2 panels with interactive Plotly controls.
        """
        figure = make_subplots(n_rows=2, n_cols=1, vertical_spacing=40 / 1200)
        figure.update_layout(showlegend=False, height=1200)
        result.matched.make_plotly_fig(
            type="match",
            fig=figure,
            panel_kw={"row": 1, "col": 1},
            label_value="Frequency (MHz)",
            label_ref=f"Ref {reference_uid}",
            label_query=f"Current {query_uid}",
        )
        result.matched.make_plotly_fig(
            type="density",
            fig=figure,
            panel_kw={"row": 2, "col": 1},
            label_ref=f"Ref Channel Id of {reference_uid}",
            label_query=f"Current Channel Id of {query_uid}",
        )
        figure.update_layout(
            title={"text": f"ToneMatch summary: {query_uid} with ref {reference_uid}"},
            margin={"t": 100, "b": 100},
        )
        return figure


_ABSENT, _NON_UNIQUE, _UNIQUE, _REFERENCE = range(4)


def _lineage_data(table):
    """Read saved correspondence arrays in acquisition and reference-row order."""
    reference = table.meta["reference"]
    ref_frequency = table[reference["frequency_column"]].to_value("MHz")
    labels, frequencies, residuals, flags = [], [], [], []
    for entry in table.meta["inputs"]:
        uid = entry["key"]
        labels.append(entry["label"])
        if uid == reference["key"]:
            frequency = ref_frequency
            residual = np.zeros(len(table))
            flag = np.full(len(table), _REFERENCE)
        else:
            frequency = table[f"f_{uid}"].to_value("MHz")
            residual = table[f"dist_shifted_{uid}"].to_value("kHz")
            flag = np.where(table[f"mask_match_unique_{uid}"], _UNIQUE, _NON_UNIQUE)
        flags.append(np.where(np.isfinite(frequency), flag, _ABSENT))
        frequencies.append(frequency)
        residuals.append(residual)
    return labels, ref_frequency, *map(np.asarray, (frequencies, residuals, flags))


def _relations(data):
    """One row per saved relation, never the nearest-match projection."""
    table = data.table
    labels, ref_frequency, *_ = _lineage_data(table)
    entries = [
        (
            np.arange(len(table)),
            ref_frequency,
            np.zeros(len(table)),
            np.full(len(table), _REFERENCE),
        )
    ]
    for entry in table.meta["inputs"][1:]:
        pairs = data.pairs[entry["key"]]
        unique = (pairs["n_matched_query"] == 1) & (pairs["n_matched_ref"] == 1)
        entries.append(
            (
                np.asarray(pairs["idx_ref"]),
                pairs["query"].to_value("MHz"),
                pairs["dist_shifted"].to_value("kHz"),
                np.where(unique, _UNIQUE, _NON_UNIQUE),
            )
        )
    return labels, ref_frequency, entries


def _projection(data):
    """Retain representative frequencies, but classify by the complete relation."""
    labels, ref_frequency, frequency, residual, status = _lineage_data(data.table)
    _, _, entries = _relations(data)
    for i, (refs, _, _, states) in enumerate(entries[1:], 1):
        status[i] = _ABSENT
        status[i, refs] = states
    return labels, ref_frequency, frequency, residual, status


def _selection_data(data):
    labels, ref_frequency, frequency, _, status = _projection(data)
    _, _, entries = _relations(data)
    # Ragged vectors retain every assignment of a reference tone. Preparation
    # visits each edge once; tone changes only replace browser trace vectors.
    traces = [[{"x": [], "y": []} for _ in ref_frequency] for _ in range(3)]
    for label, (refs, frequencies, residuals, states) in zip(
        labels, entries, strict=True
    ):
        for ref, f, residual, state in zip(
            refs, frequencies, residuals, states, strict=True
        ):
            for trace, value in (
                (int(state == _NON_UNIQUE), (f - ref_frequency[ref]) * 1000),
                (2, residual),
            ):
                traces[trace][ref]["x"].append(label)
                traces[trace][ref]["y"].append(float(value))
    return {
        "interface": data.table.meta["reference"]["interface"],
        "ref_frequency": ref_frequency.tolist(),
        "frequency": frequency.T.tolist(),
        "status": status.T.tolist(),
        "traces": traces,
    }


class ToneCorrespondenceViz(PlotlyViz):
    """All saved tone correspondences and frequency offsets across acquisitions.

    Parameters
    ----------
    quantity : {'offset', 'residual'}, optional
        Lower-panel quantity, default 'offset': frequency difference from the
        reference, or the saved residual after the fitted shift, in kHz.
    """

    quantity: Literal["offset", "residual"] = "offset"

    def __call__(self, data: ToneMatchOverview) -> go.Figure:
        """Plot all reference rows without rerunning the matching calculation.

        Parameters
        ----------
        data : ToneMatchOverview
            Complete pair relations with a reference projection and input identities.

        Returns
        -------
        plotly.graph_objects.Figure
            Frequency traces and offsets for the whole interface. Diamonds mark
            non-unique matches. Every saved edge connects the reference to its query;
            the plot does not infer links between successive queries.
            The frequency slider shows reference-tone positions and navigates
            both panels across the full band. Short vertical bars mark each
            input's LO centre when ``inputs[].f_lo_center`` is present (Hz).

        Raises
        ------
        ValueError
            The table has no reference tones.
        """
        table = data.table
        if not len(table):
            msg = "The lineage table has no reference tones"
            raise ValueError(msg)
        labels, ref_frequency, entries = _relations(data)
        figure = make_subplots(
            2,
            1,
            shared_xaxes="all",
            fig_layout=FIG_LAYOUT,
            vertical_spacing=0.12,
            subplot_titles=[
                "Frequency correspondences",
                "Offsets across the readout band",
            ],
        )
        # One line trace for all segments avoids a trace per detector.
        x, y, identities = [], [], []
        for i, (rows, frequencies, _, _) in enumerate(entries[1:], 1):
            for row, frequency in zip(rows, frequencies, strict=True):
                x.extend([float(ref_frequency[row]), float(frequency), None])
                y.extend([0, i, None])
                identities.extend([int(row), int(row), None])
        figure.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                customdata=identities,
                meta={"role": "links"},
                mode="lines",
                line={"color": "#adb5bd", "width": 1},
                hoverinfo="skip",
                showlegend=False,
                connectgaps=False,
            ),
            row=1,
            col=1,
        )
        palette = COLOR_PALETTE.cycle()
        for i, label in enumerate(labels):
            color = next(palette)
            rows, frequencies, residuals, states = entries[i]
            symbols = np.where(states == _NON_UNIQUE, "diamond-open", "circle").tolist()
            names = np.asarray(["Absent", "Non-unique", "Unique", "Reference"])[
                states
            ].tolist()
            common = {
                "type": "scattergl",
                "mode": "markers",
                "name": label,
                "legendgroup": label,
                "marker": {
                    "color": color,
                    "size": 6,
                    "symbol": symbols,
                    # WebGL's default zero-width stroke hides open symbols.
                    "line": {
                        "color": color,
                        "width": np.where(states == _NON_UNIQUE, 1, 0).tolist(),
                    },
                },
                "customdata": rows.tolist(),
                "text": names,
            }
            figure.add_trace(
                dict(
                    common,
                    x=frequencies.tolist(),
                    y=[i] * len(rows),
                    meta={"role": "frequency"},
                    hovertemplate="%{fullData.name}<br>Reference row %{customdata}"
                    "<br>%{x:.6f} MHz · %{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            if i == 0:
                continue
            offset = ((frequencies - ref_frequency[rows]) * 1000).tolist()
            shifted = residuals.tolist()
            figure.add_trace(
                dict(
                    common,
                    x=ref_frequency[rows].tolist(),
                    y=offset if self.quantity == "offset" else shifted,
                    showlegend=False,
                    meta={"role": "offset", "offset": offset, "residual": shifted},
                    hovertemplate="%{fullData.name}<br>Reference row %{customdata}"
                    "<br>%{x:.6f} MHz<br>%{y:.3f} kHz · %{text}<extra></extra>",
                ),
                row=2,
                col=1,
            )
        # Plotly's native slider cannot draw scattergl. Supply reference ticks
        # on an overlaid, hidden y axis: the main plot clips them outside [2, 3],
        # while the slider displays [0, 1]. This needs no third subplot or CSS,
        # and keeps the tick scale independent of the measured offsets.
        # Return each tick to the baseline before the next frequency, keeping
        # one continuous path rather than one path per disconnected tick.
        overview_x = np.repeat(np.sort(ref_frequency), 3).tolist()
        figure.add_trace(
            go.Scatter(
                x=overview_x,
                y=[0, 1, 0] * len(ref_frequency),
                mode="lines",
                line={"color": "#868e96", "width": 1},
                connectgaps=False,
                name="Reference tones",
                showlegend=False,
                hoverinfo="skip",
                meta={"role": "overview"},
                xaxis="x2",
                yaxis="y3",
            ),
        )
        lo_rows = [
            (i, entry["f_lo_center"] / 1e6)
            for i, entry in enumerate(table.meta["inputs"])
            if entry.get("f_lo_center") is not None
        ]
        if lo_rows:
            figure.add_trace(
                go.Scattergl(
                    x=[lo for _, lo in lo_rows],
                    y=[i for i, _ in lo_rows],
                    text=[labels[i] for i, _ in lo_rows],
                    mode="markers",
                    marker={
                        "symbol": "line-ns",
                        "size": 20,
                        "color": "#212529",
                        "line": {"width": 2},
                    },
                    name="LO centre",
                    meta={"role": "lo"},
                    hovertemplate="%{text}<br>LO centre %{x:.6f} MHz<extra></extra>",
                ),
                row=1,
                col=1,
            )
        figure.update_xaxes(showticklabels=True, row=1, col=1)
        figure.update_yaxes(
            tickvals=list(range(len(labels))),
            ticktext=labels,
            autorange="reversed",
            row=1,
            col=1,
        )
        band_frequencies = np.concatenate(
            [entry[1] for entry in entries] + [[lo for _, lo in lo_rows]]
        )
        figure.update_xaxes(
            title_text="Frequency (MHz) · reference tones",
            rangeslider={
                "visible": True,
                "thickness": 0.07,
                "autorange": True,
                "range": [
                    float(np.nanmin(band_frequencies)),
                    float(np.nanmax(band_frequencies)),
                ],
                "bgcolor": "#f1f3f5",
                "bordercolor": "#ced4da",
                "borderwidth": 1,
                "yaxis": {"rangemode": "fixed", "range": [-0.1, 1.1]},
            },
            row=2,
            col=1,
        )
        figure.update_yaxes(
            title_text="Offset (kHz)"
            if self.quantity == "offset"
            else "Residual after fitted shift (kHz)",
            fixedrange=False,
            row=2,
            col=1,
        )
        # Plotly.py exposes slider settings only for the unnumbered yaxis.
        # Reserve that axis for the reference ticks; move the upper panel to y3.
        # The lower panel stays on y2 for the browser's quantity-only patches.
        figure.layout.yaxis3 = figure.layout.yaxis
        figure.layout.xaxis.anchor = "y3"
        for trace in figure.data:
            if trace.meta["role"] == "overview":
                trace.yaxis = "y"
            elif trace.yaxis == "y":
                trace.yaxis = "y3"
        figure.layout.yaxis = {
            "anchor": "x2",
            "overlaying": "y2",
            "visible": False,
            "range": [2, 3],
            "fixedrange": True,
        }
        figure.update_layout(
            height=850,
            dragmode="zoom",
            clickmode="event",
            modebar_remove=["select2d", "lasso2d"],
            margin={"l": 145, "r": 25, "t": 85, "b": 60},
            legend={"orientation": "h", "y": 1.16, "x": 0},
            uirevision=table.meta["reference"]["uid_raw_obs"],
        )
        return figure


class ToneLineageViz(PlotlyViz):
    """Coverage, residuals and selected-tone offsets from a saved lineage table.

    Parameters
    ----------
    start, count : int, optional
        First reference row and matrix page size, default 0 and 64. At most 256
        rows are displayed. Summary statistics use all reference rows.
    tone : int, optional
        Reference row whose frequency offsets are plotted, default zero.
        Row positions are local to this result, not persistent detector IDs.
    """

    start: int = Field(default=0, ge=0)
    count: int = Field(default=64, ge=1, le=256)
    tone: int = Field(default=0, ge=0)

    @classmethod
    def make_selection_data(cls, data: ToneMatchOverview) -> dict:
        """Prepare all edge vectors for browser-only tone selection.

        Parameters
        ----------
        data : ToneMatchOverview
            Complete relations and their reference projection.

        Returns
        -------
        dict
            Matrix cells and ragged x/y vectors retaining every assignment.
        """
        return _selection_data(data)

    def __call__(self, data: ToneMatchOverview) -> go.Figure:
        """Build diagnostics without matching or modifying the saved table.

        Parameters
        ----------
        data : ToneMatchOverview
            Complete pair relations and their reference projection.

        Returns
        -------
        plotly.graph_objects.Figure
            Coverage counts, median and 16-84 percentile residuals for unique
            matches, a paged correspondence matrix and selected-row offsets.
            Non-unique assignments use separate markers; every branch appears in
            selected-tone offsets. Matrix frequencies use the best-match projection.

        Raises
        ------
        ValueError
            The matrix start or selected reference row is outside the table.
        """
        table = data.table
        if self.start >= len(table) or self.tone >= len(table):
            msg = "Reference row is outside this lineage table"
            raise ValueError(msg)
        reference = table.meta["reference"]
        ref_uid = reference["uid_raw_obs"]
        labels, ref_frequency, frequency, residual, status = _projection(data)
        colors = ["#dee2e6", "#fab005", "#12b886", "#228be6"]
        names = ["Absent", "Non-unique", "Unique", "Reference"]
        figure = make_subplots(
            3,
            2,
            fig_layout=FIG_LAYOUT,
            specs=[[{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None]],
            row_heights=[0.23, 0.45, 0.32],
            vertical_spacing=0.12,
            horizontal_spacing=0.16,
            subplot_titles=[
                "Correspondence counts",
                "Unique-match residuals",
                "Reference-tone correspondences",
                "Selected reference tone",
            ],
        )
        for state, (name, color) in enumerate(zip(names, colors, strict=True)):
            figure.add_trace(
                go.Bar(
                    x=labels,
                    y=(status == state).sum(axis=1),
                    name=name,
                    legendgroup=name,
                    marker_color=color,
                    hovertemplate="%{x}<br>%{y} reference rows<extra>"
                    + name
                    + "</extra>",
                ),
                row=1,
                col=1,
            )
        quantiles = np.full((len(labels), 3), np.nan)
        for i in range(len(labels)):
            values = residual[i][(status[i] == _UNIQUE) & np.isfinite(residual[i])]
            if len(values):
                quantiles[i] = np.percentile(values, [16, 50, 84])
        figure.add_trace(
            go.Scattergl(
                x=labels,
                y=quantiles[:, 1],
                mode="markers",
                name="Median residual",
                showlegend=False,
                marker_color=colors[2],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": quantiles[:, 2] - quantiles[:, 1],
                    "arrayminus": quantiles[:, 1] - quantiles[:, 0],
                },
                hovertemplate=(
                    "%{x}<br>Median %{y:.2f} kHz<br>Whiskers: 16-84%<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        stop = min(self.start + self.count, len(table))
        rows = np.arange(self.start, stop)
        scale = [
            [edge, color]
            for i, color in enumerate(colors)
            for edge in (i / 4, (i + 1) / 4)
        ]
        selected_status = status[:, self.start : stop].T
        figure.add_trace(
            go.Heatmap(
                meta={"role": "matrix"},
                x=labels,
                y=rows,
                z=selected_status,
                zmin=-0.5,
                zmax=3.5,
                colorscale=scale,
                showscale=False,
                customdata=np.stack(
                    [
                        np.broadcast_to(rows[:, None], selected_status.shape),
                        frequency[:, self.start : stop].T,
                    ],
                    axis=-1,
                ),
                text=np.asarray(names)[selected_status],
                hovertemplate=(
                    "%{x}<br>Reference row %{y}<br>%{text}"
                    "<br>%{customdata[1]:.6f} MHz<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        if self.start <= self.tone < stop:
            figure.add_hrect(
                y0=self.tone - 0.5,
                y1=self.tone + 0.5,
                fillcolor="rgba(0,0,0,0)",
                line_width=2,
                line_color="#343a40",
                row=2,
                col=1,
            )
        traces = _selection_data(data)["traces"]
        figure.add_trace(
            go.Scattergl(
                x=traces[0][self.tone]["x"],
                y=traces[0][self.tone]["y"],
                meta={"role": "tone", "trace": 0},
                mode="markers",
                name="Frequency offset",
                showlegend=False,
                line_color="#228be6",
                connectgaps=False,
                hovertemplate="%{x}<br>Δf %{y:.2f} kHz<extra></extra>",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=traces[1][self.tone]["x"],
                y=traces[1][self.tone]["y"],
                meta={"role": "tone", "trace": 1},
                mode="markers",
                name="Non-unique",
                legendgroup="Non-unique",
                showlegend=False,
                marker={"color": colors[1], "symbol": "diamond", "size": 9},
                hovertemplate="%{x}<br>Non-unique Δf %{y:.2f} kHz<extra></extra>",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=traces[2][self.tone]["x"],
                y=traces[2][self.tone]["y"],
                meta={"role": "tone", "trace": 2},
                mode="markers",
                name="After fitted shift",
                line={"color": "#12b886", "dash": "dot"},
                connectgaps=False,
                hovertemplate="%{x}<br>Residual %{y:.2f} kHz<extra></extra>",
            ),
            row=3,
            col=1,
        )
        figure.update_yaxes(title_text="Reference rows", row=1, col=1)
        figure.update_yaxes(title_text="Residual (kHz)", row=1, col=2)
        figure.update_yaxes(
            title_text="Reference row", autorange="reversed", row=2, col=1
        )
        figure.update_yaxes(title_text="Offset (kHz)", row=3, col=1)
        figure.update_xaxes(
            tickangle=-25, automargin=True, categoryorder="array", categoryarray=labels
        )
        figure.update_xaxes(showticklabels=False, row=2, col=1)
        figure.update_layout(
            height=1000,
            barmode="stack",
            margin={"l": 60, "r": 20, "t": 80, "b": 60},
            title={
                "text": (
                    f"{reference['interface']} · row {self.tone} · "
                    f"{ref_frequency[self.tone]:.6f} MHz"
                )
            },
            legend={"orientation": "h", "y": 1.12, "x": 0},
            uirevision=f"{ref_uid}:{self.start}:{self.count}",
        )
        return figure
