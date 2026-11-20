"""Plotly diagnostics for saved ROACH readout powers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import Field
from tollan.plot.plotly import PlotlyViz

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .sweep_check_view import SweepCheckView


class RoachReadoutPowerViz(PlotlyViz):
    """Predicted/inferred power chain and per-tone drive powers.

    Parameters
    ----------
    height : int, optional
        Figure height in pixels, default 950.
    """

    height: int = Field(default=950, ge=400)

    def __call__(
        self, view: SweepCheckView, *, channels: Sequence[int] | None = None
    ) -> go.Figure:
        """Plot an acquired readout's saved power budget.

        Parameters
        ----------
        view : SweepCheckView
            Saved check with readout-power diagnostics and an open source.
        channels : sequence of int or None, optional
            Positional channel indices for the frequency panel. None shows all.
            The total-power panel always describes the complete checked comb.

        Returns
        -------
        plotly.graph_objects.Figure
            Loaded two-panel figure with calibrated limits and linked-scale
            predicted/inferred drive powers. The source can then be closed.

        Raises
        ------
        ValueError
            The check has no saved readout-power diagnostics.
        """
        power = view.readout_power
        if power is None:
            msg = "The sweep check has no readout-power result"
            raise ValueError(msg)
        selected = (
            list(range(power.frequency.size)) if channels is None else list(channels)
        )
        return _make_figure(power, selected, height=self.height)


def _make_figure(result, selected, *, height):  # noqa: PLR0915 -- v2 multi-panel layout
    """Preserve the production-v2 layout and plotted quantities."""
    snapshot = result.adc_snap_frac
    snapshot_label = "unavailable" if snapshot is None else f"{snapshot.item():.1%}"
    # Create figure with two panels (rows)
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            (f"Total Power Chain (DAC → ADC) | ADC Snap: {snapshot_label}"),
            "Drive Output Power vs Detector Frequency",
        ),
        vertical_spacing=0.20,
        row_heights=[0.5, 0.5],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    # ========== Top Panel: Total Power Chain ==========
    labels = result.element.values.tolist()
    n_elements = len(labels)

    # Collect input/output power and op values for each element
    predicted_inputs = result.predicted_total_input.values.tolist()
    predicted_outputs = result.predicted_total_output.values.tolist()
    inferred_inputs = result.inferred_total_input.values.tolist()
    inferred_outputs = result.inferred_total_output.values.tolist()
    op_values = result.gain.values.tolist()

    # Predicted: Input side (nodes)
    fig.add_trace(
        go.Scatter(
            x=[i - 0.15 for i in range(n_elements)],
            y=predicted_inputs,
            mode="markers",
            name="Pred In",
            marker={"color": "lightblue", "size": 14, "symbol": "circle"},
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Predicted: Output side (nodes) - larger and prominent
    fig.add_trace(
        go.Scatter(
            x=[i + 0.15 for i in range(n_elements)],
            y=predicted_outputs,
            mode="markers",
            name="Pred Out",
            marker={"color": "blue", "size": 18, "symbol": "square"},
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Predicted: Connection lines
    for i in range(n_elements):
        fig.add_trace(
            go.Scatter(
                x=[i - 0.15, i + 0.15],
                y=[predicted_inputs[i], predicted_outputs[i]],
                mode="lines",
                line={"color": "blue", "width": 3},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # Inferred: Input side (nodes)
    fig.add_trace(
        go.Scatter(
            x=[i - 0.15 for i in range(n_elements)],
            y=inferred_inputs,
            mode="markers",
            name="Infer In",
            marker={"color": "lightcoral", "size": 14, "symbol": "circle"},
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Inferred: Output side (nodes) - larger and prominent
    fig.add_trace(
        go.Scatter(
            x=[i + 0.15 for i in range(n_elements)],
            y=inferred_outputs,
            mode="markers",
            name="Infer Out",
            marker={"color": "red", "size": 18, "symbol": "square"},
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Inferred: Connection lines
    for i in range(n_elements):
        fig.add_trace(
            go.Scatter(
                x=[i - 0.15, i + 0.15],
                y=[inferred_inputs[i], inferred_outputs[i]],
                mode="lines",
                line={"color": "red", "width": 3, "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # Add config limits as thick segments at nodes with shading
    limit_configs = [
        ("LNA In Max", "lna", result.config.lna_input_dbm_max, "orange", "input"),
        (
            "LNA Out Max",
            "lna",
            result.config.lna_output_dbm_max,
            "darkorange",
            "output",
        ),
        (
            "IF Brd In Max",
            "if_board",
            result.config.if_board_input_dbm_max,
            "purple",
            "input",
        ),
    ]

    # Color mapping for shading (with transparency)
    color_shading = {
        "orange": "rgba(255, 165, 0, 0.15)",
        "darkorange": "rgba(255, 140, 0, 0.15)",
        "purple": "rgba(128, 0, 128, 0.15)",
    }

    for limit_name, elem_label, limit_value, color, side in limit_configs:
        if limit_value is not None:
            elem_idx = labels.index(elem_label) if elem_label in labels else None
            if elem_idx is not None:
                # Draw shorter thick segment at the node
                x_offset = -0.15 if side == "input" else 0.15
                fig.add_trace(
                    go.Scatter(
                        x=[elem_idx + x_offset - 0.15, elem_idx + x_offset + 0.15],
                        y=[limit_value, limit_value],
                        mode="lines",
                        line={"color": color, "width": 6},
                        name=limit_name,
                        showlegend=True,
                        hovertemplate=(
                            f"{limit_name}: {limit_value:.1f}\n dBm<extra></extra>",
                        ),
                    ),
                    row=1,
                    col=1,
                )

                # Add shaded region above limit (out-of-limit zone)
                y_max = (
                    max(
                        *predicted_inputs,
                        *predicted_outputs,
                        *inferred_inputs,
                        *inferred_outputs,
                    )
                    + 5
                )
                fig.add_trace(
                    go.Scatter(
                        x=[
                            elem_idx + x_offset - 0.15,
                            elem_idx + x_offset + 0.15,
                            elem_idx + x_offset + 0.15,
                            elem_idx + x_offset - 0.15,
                        ],
                        y=[limit_value, limit_value, y_max, y_max],
                        fill="toself",
                        fillcolor=color_shading.get(
                            color,
                            "rgba(128, 128, 128, 0.15)",
                        ),
                        mode="lines",
                        line={"width": 0, "color": "rgba(0,0,0,0)"},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )

    # Add input power values on input side of nodes (predicted) with unit
    for i, (_, power) in enumerate(zip(labels, predicted_inputs, strict=False)):
        fig.add_annotation(
            x=i - 0.15,
            y=power,
            text=f"{power:.1f} dBm",
            showarrow=False,
            font={"size": 9, "color": "blue"},
            xshift=-30,
            bgcolor="rgba(255, 255, 255, 0.7)",
            row=1,
            col=1,
        )

    # Add input power values on input side of nodes (inferred) with unit
    for i, (_, power) in enumerate(zip(labels, inferred_inputs, strict=False)):
        fig.add_annotation(
            x=i - 0.15,
            y=power,
            text=f"{power:.1f} dBm",
            showarrow=False,
            font={"size": 9, "color": "red"},
            xshift=-30,
            bgcolor="rgba(255, 255, 255, 0.7)",
            row=1,
            col=1,
        )

    # ========== Bottom Panel: Drive Output Power vs Frequency ==========
    freq_MHz = (
        result.frequency.isel(chan=selected).u.to("MHz").values
    )  # Convert Hz to MHz

    # Get drive_atten output power for predicted and inferred
    predicted_drive_output = (
        result.predicted_per_tone_output.sel(element="drive_atten")
        .isel(chan=selected)
        .values
    )
    inferred_drive_output = (
        result.inferred_per_tone_output.sel(element="drive_atten")
        .isel(chan=selected)
        .values
    )

    # Calculate offset between predicted and inferred
    offset = np.mean(inferred_drive_output) - np.mean(predicted_drive_output)

    # Primary y-axis: Predicted
    fig.add_trace(
        go.Scattergl(
            x=freq_MHz,
            y=predicted_drive_output,
            mode="markers",
            name="Predicted",
            marker={"color": "blue", "size": 6, "opacity": 0.6},
            showlegend=True,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    # Secondary y-axis: Inferred (keep actual values, adjust axis instead)
    fig.add_trace(
        go.Scattergl(
            x=freq_MHz,
            y=inferred_drive_output,
            mode="markers",
            name="Inferred",
            marker={"color": "red", "size": 6, "opacity": 0.6},
            showlegend=True,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # Update axes labels
    # Create tick labels with op values
    tick_labels = []
    for label, op_db in zip(labels, op_values, strict=False):
        if abs(op_db) > 0.0:
            tick_labels.append(f"{label}<br>({op_db:+.1f} dB)")
        else:
            tick_labels.append(label)

    fig.update_xaxes(
        title_text="Power Chain Element",
        tickmode="array",
        tickvals=list(range(n_elements)),
        ticktext=tick_labels,
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Power (dBm)", row=1, col=1)

    fig.update_xaxes(title_text="Detector Frequency (MHz)", row=2, col=1)
    fig.update_yaxes(
        title_text="Predicted Drive Power (dBm)",
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Inferred Drive Power (dBm)",
        row=2,
        col=1,
        secondary_y=True,
    )

    # Lock the scale (same range) for both y-axes in bottom panel
    pred_range = predicted_drive_output.max() - predicted_drive_output.min()
    pred_center = (predicted_drive_output.max() + predicted_drive_output.min()) / 2
    margin = pred_range * 0.1  # 10% margin

    # Primary axis (predicted)
    fig.update_yaxes(
        range=[
            pred_center - pred_range / 2 - margin,
            pred_center + pred_range / 2 + margin,
        ],
        row=2,
        col=1,
        secondary_y=False,
    )
    # Secondary axis (inferred) - offset to show actual values
    infer_center = pred_center + offset
    fig.update_yaxes(
        range=[
            infer_center - pred_range / 2 - margin,
            infer_center + pred_range / 2 + margin,
        ],
        row=2,
        col=1,
        secondary_y=True,
    )

    # Update layout with horizontal legend between panels
    fig.update_layout(
        title={
            "text": f"ROACH {result.roach} Tone Power Analysis<br>"
            f"<sub>Config: Drive Atten={result.atten_drive:.1f} dB, "
            f"Sense Atten={result.atten_sense:.1f} dB, "
            f"# Tones={len(selected)}, "
            f"ADC Snap Min={result.config.adc_snap_frac_min:.1%}</sub>",
            "x": 0.5,
            "xanchor": "center",
        },
        height=height,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.48,
            "xanchor": "center",
            "x": 0.5,
        },
        hovermode="closest",
    )

    return fig
