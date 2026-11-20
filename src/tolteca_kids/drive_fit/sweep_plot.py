"""Per-channel sweep and phase-fit diagnostics from saved results."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.offline import get_plotlyjs
from tollan.pipeline import ArtifactFile, ArtifactSpec
from tollan.plot.plotly import make_range, make_subplots

from tolteca_kidsproc._phase_fit import removecable

from .._plot import html_document as _document

if TYPE_CHECKING:
    from pathlib import Path

    from .result import DriveFitResult


class DriveFitSweepArtifacts(ArtifactSpec[dict[int, go.Figure]]):
    """Channel browser with collated magnitude, I/Q, phase and residual figures.

    Parameters
    ----------
    channels : list of int or None, optional
        Recorded channel IDs in requested order. None includes all channels.

    Notes
    -----
    Figures overlay all powers for each channel. The attenuation selector or
    grouped legend isolates a power across all four panels. Figures use saved
    sweep, circle and phase-fit arrays. Construction loads the
    selected channels; keep the source product open until the call returns.
    Exported pages share a local Plotly library and work without a server or
    internet access. Keep the browser HTML and its sibling directory together.
    """

    channels: list[int] | None = None

    def __call__(self, result: DriveFitResult) -> dict[int, go.Figure]:
        """Construct a figure for each selected channel without fitting again.

        Parameters
        ----------
        result : DriveFitResult
            Saved or newly fitted scientific arrays.

        Returns
        -------
        dict of int to plotly.graph_objects.Figure
            Figures keyed by recorded channel ID. Each figure contains all powers,
            with measured magnitude, I/Q circles and models, phase and residuals.

        Raises
        ------
        ValueError
            The channel selection is empty or the product lacks circle diagnostics.
            Rerun drive fitting to add diagnostics to an older product.
        KeyError
            A requested channel is absent.
        """
        data = result.data
        required = {
            "I_model",
            "Q_model",
            "I_circle_center",
            "Q_circle_center",
            "circle_radius",
            "phase_full",
        }
        if missing := required.difference(data.data_vars):
            msg = (
                f"Drive-fit product lacks circle diagnostics: {sorted(missing)}. "
                "Rerun drive fitting to record them."
            )
            raise ValueError(msg)
        if self.channels is not None:
            data = data.sel(chan=self.channels)
        if not data.sizes["chan"]:
            msg = "Select at least one fitted channel to plot"
            raise ValueError(msg)
        return {
            int(channel): _sweep_figure(data.sel(chan=channel).load())
            for channel in data.chan.values
        }

    def write(self, data: dict[int, go.Figure], *, path: Path) -> list[ArtifactFile]:
        """Export a channel browser, linked figures and their shared library.

        Parameters
        ----------
        data : dict of int to plotly.graph_objects.Figure
            Previously constructed figures, independent of source storage.
        path : pathlib.Path
            Destination directory for ``drive_fit_sweeps.html`` and the
            ``drive_fit_sweeps`` directory.

        Returns
        -------
        list of ArtifactFile
            Browser, channel HTML files and local JavaScript asset receipts,
            suitable for inclusion in a product index.
        """
        directory = path / "drive_fit_sweeps"
        directory.mkdir(parents=True, exist_ok=True)
        library = directory / "plotly.min.js"
        library.write_text(get_plotlyjs())
        receipts = [ArtifactFile(library, "js")]
        groups = {"Flagged or without a selection": [], "Accepted": []}
        for channel, figure in data.items():
            metadata = figure.layout.meta
            label = f"Channel {channel} · {metadata['status']}"
            group = (
                "Accepted" if metadata["accepted"] else "Flagged or without a selection"
            )
            groups[group].append(
                f'<li><a href="drive_fit_sweeps/channel_{channel}.html">'
                f"{escape(label)}</a></li>"
            )
            errors = "".join(
                f"<li>{-power:g} dB attenuation: {escape(error)}</li>"
                for power, error in metadata["errors"]
            )
            notes = (
                f"<details><summary>Fit failures ({len(metadata['errors'])})"
                f"</summary><ul>{errors}</ul></details>"
                if errors
                else ""
            )
            body = (
                '<nav><a href="../drive_fit_sweeps.html">All channels</a></nav>'
                + figure.to_html(
                    full_html=False, include_plotlyjs=False, auto_play=False
                )
                + notes
            )
            target = directory / f"channel_{channel}.html"
            target.write_text(
                _document(label, body, '<script src="plotly.min.js"></script>')
            )
            receipts.append(ArtifactFile(target, "html"))
        body = (
            "<h1>Drive fit · sweep diagnostics</h1>"
            "<p>Each channel collates all powers: magnitude, I/Q circles and fits, "
            "phase, and residuals. Use the attenuation selector to inspect one power, "
            "or double-click a legend entry to isolate it across every panel. "
            "Failed fits retain the measured magnitude and I/Q sweep.</p>"
        )
        for label, links in groups.items():
            body += (
                f"<h2>{label} ({len(links)})</h2>"
                '<ul class="channels">' + "".join(links) + "</ul>"
            )
        target = path / "drive_fit_sweeps.html"
        target.write_text(_document("Drive-fit sweep diagnostics", body))
        return [ArtifactFile(target, "html"), *receipts]


def _sweep_figure(data):
    figure = make_subplots(
        2,
        2,
        horizontal_spacing=0.14,
        vertical_spacing=0.20,
        subplot_titles=(
            "Magnitude · data and fit",
            "I/Q · data, circle and fit",
            "Phase · full sweep and fit interval",
            "Phase residual · data - fit",
        ),
    )
    frequency_range = make_range((data.frequency.values - data.f_chan.item()) / 1e3)
    for row, col in ((1, 1), (2, 1), (2, 2)):
        figure.update_xaxes(
            title_text="Frequency offset (kHz)",
            range=frequency_range,
            row=row,
            col=col,
        )
    figure.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    figure.update_xaxes(title_text="I (reference amplitude)", row=1, col=2)
    figure.update_yaxes(
        title_text="Q (reference amplitude)",
        row=1,
        col=2,
        scaleanchor="x2",
        scaleratio=1,
        constrain="domain",
    )
    figure.update_yaxes(title_text="Phase (rad)", row=2, col=1)
    figure.update_yaxes(title_text="Residual (rad)", row=2, col=2)
    figure.update_xaxes(matches="x", row=2, col=1)
    figure.update_xaxes(matches="x", row=2, col=2)
    colors = sample_colorscale("Turbo", np.linspace(0, 1, data.sizes["power"]))
    signal = data.I.values + 1j * data.Q.values
    for index, color in enumerate(colors):
        sample = data.isel(power=index)
        # Same amplitude/phase normalization and cable correction as kid_phase_fit.
        # These display transforms do not estimate or refit any parameters.
        with np.errstate(divide="ignore", invalid="ignore"):
            corrected = removecable(
                sample.frequency.values,
                signal[index] * (signal[0, -1] / signal[index, -1]),
                data.attrs["cable_delay_ns"],
            )
        _add_power(figure, sample, corrected, color)
    selected = np.isfinite(data.p_best.item())
    accepted = selected and data.p_flag.item() == 0
    status = "accepted" if accepted else "flagged" if selected else "no selection"
    selection = (
        f"selected attenuation {-data.p_best.item():.3f} dB · "
        f"a={data.a_best.item():.3g}"
        if selected
        else "no selected drive"
    )
    failures = int((data.fit_flag.values != 0).sum())
    phase_flags = int(
        ((data.fit_flag.values == 0) & (data.phase_flag.values != 0)).sum()
    )
    # Each choice shows the same power in all panels and resets ranges after isolation.
    # Keep full frequency coverage when isolating powers. Updating autorange on
    # matched frequency axes can leave stale tick positions in Plotly.
    ranges: dict[str, bool | tuple[float, float]] = {
        f"yaxis{suffix}.autorange": True for suffix in ("", "2", "3", "4")
    }
    ranges["xaxis2.autorange"] = True
    for suffix in ("", "3", "4"):
        ranges[f"xaxis{suffix}.range"] = frequency_range
        ranges[f"xaxis{suffix}.autorange"] = False
    buttons = [
        {
            "label": "All powers",
            "method": "update",
            "args": [{"visible": [True] * len(figure.data)}, ranges],
        }
    ]
    for power in data.power.values:
        group = str(float(power))
        buttons.append(
            {
                "label": f"{-power:g} dB attenuation",
                "method": "update",
                "args": [
                    {"visible": [trace.legendgroup == group for trace in figure.data]},
                    ranges,
                ],
            }
        )
    figure.update_layout(
        title={
            "text": f"{escape(data.attrs['interface'])} · "
            f"Channel {data.chan.item()} · {status}"
            f"<br><sup>{data.f_chan.item() / 1e6:.4f} MHz · {selection}"
            f" · {failures} failed / {phase_flags} phase flagged"
            "<br>Dots: data · solid: fitted model · dotted: circle · faint: full phase"
            "<br>I/Q and magnitude: reference-normalized, cable-delay corrected</sup>",
            "y": 0.98,
            "yanchor": "top",
            "x": 0.055,
        },
        height=1000,
        template="plotly_white",
        legend={
            "title": {"text": "Attenuation"},
            "groupclick": "togglegroup",
            "x": 1.03,
            "y": 1,
            "font": {"size": 11},
        },
        updatemenus=[
            {
                "type": "dropdown",
                "buttons": buttons,
                "x": 1.03,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        margin={"t": 205, "b": 65, "l": 80, "r": 205},
        meta={
            "channel": data.chan.item(),
            "accepted": bool(accepted),
            "status": status,
            "errors": [
                (float(p), str(e))
                for p, e in zip(data.power.values, data.fit_error.values, strict=True)
                if e
            ],
        },
    )
    return figure


def _magnitude(z):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20 * np.log10(np.abs(z))


def _add_power(figure, sample, corrected, color):
    power = float(sample.power.item())
    completed = sample.fit_flag.item() == 0
    flag = int(sample.phase_flag.item())
    label = f"{-power:g} dB"
    if not completed:
        label += " · failed"
    elif flag:
        label += f" · phase {flag}"
    detail = (
        f"{label}<br>a={sample.a.item():.4g} · Qr={sample.Qr.item():.5g}"
        f"<br>fit_flag={sample.fit_flag.item()} · phase_flag={flag}"
    )
    frequency = sample.f_chan.item()
    offset = (sample.frequency.values - frequency) / 1e3
    valid = np.isfinite(sample.frequency_fit.values) & completed
    fit_offset = (sample.frequency_fit.values[valid] - frequency) / 1e3
    model = sample.I_model.values[valid] + 1j * sample.Q_model.values[valid]

    def add(x, y, role, row, col, *, mode="lines", showlegend=False, **style):  # noqa: PLR0913 -- trace placement and styling.
        trace = go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=label if showlegend else role,
            legendgroup=str(power),
            showlegend=showlegend,
            line={"color": color, "width": 1.6},
            marker={"color": color, "size": 4},
            meta={"power": power, "role": role},
            hovertemplate=(
                f"{detail}<br>{role}<br>x=%{{x:.6g}} · y=%{{y:.6g}}<extra></extra>"
            ),
        )
        trace.update(**style)
        figure.add_trace(trace, row=row, col=col)

    add(
        offset,
        _magnitude(corrected),
        "Magnitude data",
        1,
        1,
        mode="markers",
        showlegend=True,
    )
    add(fit_offset, _magnitude(model), "Magnitude model", 1, 1)
    add(corrected.real, corrected.imag, "I/Q data", 1, 2, mode="markers")
    circle = (
        sample.I_circle_center.item()
        + 1j * sample.Q_circle_center.item()
        + sample.circle_radius.item() * np.exp(1j * np.linspace(0, 2 * np.pi, 100))
    )
    add(
        circle.real if completed else [],
        circle.imag if completed else [],
        "Fitted circle",
        1,
        2,
        line={"dash": "dot", "width": 1},
        opacity=0.45,
    )
    add(model.real, model.imag, "I/Q model", 1, 2, line={"width": 2})
    add(
        offset if completed else [],
        sample.phase_full.values if completed else [],
        "Full phase data",
        2,
        1,
        mode="markers",
        opacity=0.18,
        marker={"size": 3},
    )
    add(fit_offset, sample.phase_data.values[valid], "Phase data", 2, 1, mode="markers")
    add(fit_offset, sample.phase_model.values[valid], "Phase model", 2, 1)
    add(
        fit_offset,
        sample.phase_residual.values[valid],
        "Phase residual",
        2,
        2,
        mode="lines+markers",
    )
