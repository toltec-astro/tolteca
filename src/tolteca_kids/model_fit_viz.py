"""Summary, group and channel-resonator diagnostics from saved ModelFit arrays."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from pydantic import Field
from tollan.pipeline import ArtifactContext, ArtifactFile, ArtifactSpec
from tollan.plot.plotly import PlotlyViz, make_subplots

from tolteca_kidsproc.models import KidsSweepWithAffineGain, KidsSweepWithDelayGain

from ._model_fit_html import write_model_fit_browser
from ._plot import FIG_LAYOUT
from .model_fit_quality import ModelFitBitMask, flag_names

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from tolteca_kidsproc.analysis import KidsModelView

_DATA = "#3467a1"
_MODEL = "#c25b17"
_GUIDE = "#52606d"


def _style(figure, title, height=850):
    figure.update_layout(**FIG_LAYOUT)
    figure.update_xaxes(FIG_LAYOUT["xaxis"])
    figure.update_yaxes(FIG_LAYOUT["yaxis"])
    figure.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.05, "y": 0.98, "yanchor": "top"},
        height=height,
        margin={"t": 150, "l": 80, "r": 60, "b": 65},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return figure


class ModelFitSummaryViz(PlotlyViz):
    """Overview of every joint fit and every acquired channel's warning bits."""

    def __call__(self, view, quality):
        """Return residual, conditioning, nonlinearity and channel-flag panels.

        Parameters
        ----------
        view : tolteca_kidsproc.analysis.model_fit.KidsModelView
            Saved Python fit; keep backing storage open during construction.
        quality : xarray.Dataset
            ModelFit quality table with the same group and channel order.

        Returns
        -------
        plotly.graph_objects.Figure
            Four-panel overview of every group and acquired channel.
        """
        fig = make_subplots(
            2,
            2,
            horizontal_spacing=0.12,
            vertical_spacing=0.20,
            subplot_titles=(
                "Residual / measured IQ norm",
                "Scaled Jacobian condition",
                "Fitted nonlinearity",
                "Channel warning counts (overlap allowed)",
            ),
        )
        group_flags = quality.group_flag.values
        freq = quality.frequency.values / 1e6
        a_nl = [
            float(view.fit.a_nl.where(view.fit.group_id == g).max())
            for g in quality.group_id.values
        ]
        custom = np.array(
            [
                [str(g), flag_names(f)]
                for g, f in zip(quality.group_id.values, group_flags, strict=True)
            ]
        )
        for mask, label, color in (
            (group_flags == 0, "No group warnings", _DATA),
            (group_flags != 0, "Group warnings", _MODEL),
        ):
            for row, col, values in (
                (1, 1, quality.residual_fraction.values),
                (1, 2, view.joint.condition.values),
                (2, 1, np.asarray(a_nl)),
            ):
                fig.add_trace(
                    go.Scattergl(
                        x=freq[mask],
                        y=values[mask],
                        mode="markers",
                        marker={"color": color, "size": 5},
                        name=label,
                        legendgroup=label,
                        showlegend=(row, col) == (1, 1),
                        customdata=custom[mask],
                        hovertemplate=(
                            "Group %{customdata[0]}<br>%{x:.6f} MHz · %{y:.4g}"
                            "<br>%{customdata[1]}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                fig.update_xaxes(
                    title_text="Resonance frequency (MHz)", row=row, col=col
                )
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=2)
        bits = list(ModelFitBitMask)
        fig.add_bar(
            x=[f.name for f in bits],
            y=[int(np.count_nonzero(quality.channel_flag.values & f)) for f in bits],
            marker_color=_MODEL,
            showlegend=False,
            row=2,
            col=2,
        )
        fig.update_xaxes(tickangle=-45, row=2, col=2)
        fig.update_yaxes(title_text="Channels", row=2, col=2)
        return _style(
            fig,
            f"Model fit · {len(freq)} groups · "
            f"{quality.sizes['chan']} acquired channels"
            "<br><sup>Warnings are diagnostics; fixed-probe recovery "
            "remains separately assessed.</sup>",
            920,
        )


def _pair(row):
    # The caller validates the complete calibration once before constructing the
    # atlas. These are public physical views, not a second derotation formula.
    values = {
        "fr": float(row.fr),
        "r": float(row.r),
        "c": complex(row.c_re, row.c_im),
        "f_g": float(row.f_g),
        "H0": complex(row.H0_re, row.H0_im),
        "a_nl": float(row.a_nl),
        "nonlinear_branch": row.attrs.get("nonlinear_branch", "unique"),
    }
    if row.gain_family.item() == "affine":
        return KidsSweepWithAffineGain(**values, H1=complex(row.H1_re, row.H1_im))
    return KidsSweepWithDelayGain(**values, a=float(row.gain_a), tau=float(row.tau))


def _group_figure(rows, samples, quality, joint):
    group = str(quality.group_id.item())
    fig = make_subplots(
        2,
        2,
        horizontal_spacing=0.14,
        vertical_spacing=0.20,
        subplot_titles=(
            "Magnitude · measured and joint model",
            "Measured I/Q · data and joint model",
            "I/Q residual · data minus joint model",
            "Derotated about selected resonator / Qr",
        ),
    )
    seen, selections, trace_ranges = set(), [], []
    theta = np.linspace(0, 2 * np.pi, 181)
    circle = 0.5 + 0.5 * np.exp(1j * theta)
    for row_index in range(rows.sizes["row"]):
        row = rows.isel(row=row_index)
        chan, res = row.chan.item(), row.resonator_id.item()
        if (chan, res) in seen:
            continue
        seen.add((chan, res))
        sample = samples.isel(
            sample=np.flatnonzero(samples.chan.values == chan)
        ).sortby("frequency")
        freq = sample.frequency.values
        z = sample.I.values + 1j * sample.Q.values
        prediction = sample.I_model.values + 1j * sample.Q_model.values
        residual = z - prediction
        offset = (freq - float(row.f_g)) / 1e3
        pair = _pair(row)
        transformed = pair.derotate(z, freq) / float(row.Qr)
        transformed_model = pair.derotate(prediction, freq) / float(row.Qr)
        first = len(fig.data)
        label = f"Channel {chan} · resonator {res}"
        title = f"Group {escape(group)} · {escape(label)} · {row.gain_family.item()}"
        title += (
            f"<br><sup>Qr={float(row.Qr):.5g} · Qc={float(row.Qc):.5g} · "
            f"A={float(row.A):.4g} · a_nl={float(row.a_nl):.4g}"
            f" · residual={float(quality.residual_fraction):.3g} · "
            f"condition={float(joint.condition):.3g}"
            "<br>Derotation retains neighbours. Dashed circle describes "
            "the selected isolated resonance.</sup>"
        )

        def add(x, y, name, r, c, *, color=_DATA, mode="lines", dash=None, show=False):  # noqa: PLR0913 -- Trace coordinates and style.
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=name,
                    mode=mode,
                    line={"color": color, "width": 1.6, "dash": dash},
                    marker={"color": color, "size": 4},
                    showlegend=show,
                    visible=not selections,
                    hovertemplate=(
                        f"{escape(name)}<br>x=%{{x:.6g}} · y=%{{y:.6g}}<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )

        included = sample.fit_selected.values
        for selected, name, color in (
            (included, "Fitted samples", _DATA),
            (~included, "Excluded samples", _GUIDE),
        ):
            add(
                offset[selected],
                abs(z[selected]),
                name,
                1,
                1,
                mode="markers",
                color=color,
                show=True,
            )
            add(
                z[selected].real,
                z[selected].imag,
                name,
                1,
                2,
                mode="markers",
                color=color,
            )
        add(offset, abs(prediction), "Joint model", 1, 1, color=_MODEL, show=True)
        add(prediction.real, prediction.imag, "Joint model", 1, 2, color=_MODEL)
        add(
            offset[included],
            residual.real[included],
            "I residual",
            2,
            1,
            mode="lines+markers",
            show=True,
        )
        add(
            offset[included],
            residual.imag[included],
            "Q residual",
            2,
            1,
            mode="lines+markers",
            color=_MODEL,
            show=True,
        )
        add(transformed.real, transformed.imag, "Derotated data", 2, 2, mode="markers")
        add(
            transformed_model.real,
            transformed_model.imag,
            "Derotated joint model",
            2,
            2,
            color=_MODEL,
        )
        add(
            circle.real,
            circle.imag,
            "Isolated canonical circle",
            2,
            2,
            color=_GUIDE,
            dash="dash",
            show=True,
        )
        trace_ranges.append((first, len(fig.data)))
        selections.append((label, title))
    ranges = {
        f"{axis}axis{suffix}.autorange": True
        for axis in ("x", "y")
        for suffix in ("", "2", "3", "4")
    }
    buttons = [
        {
            "label": label,
            "method": "update",
            "args": [
                {"visible": [start <= i < stop for i in range(len(fig.data))]},
                {**ranges, "title.text": title},
            ],
        }
        for (label, title), (start, stop) in zip(selections, trace_ranges, strict=True)
    ]
    fig.update_xaxes(title_text="f - gain origin (kHz)", row=1, col=1)
    fig.update_xaxes(title_text="f - gain origin (kHz)", row=2, col=1, matches="x")
    units = rows.signal_units.values[0]
    fig.update_yaxes(title_text=f"|S21| ({units})", row=1, col=1)
    fig.update_yaxes(title_text=f"Residual ({units})", row=2, col=1)
    for r, c, suffix in ((1, 2, "2"), (2, 2, "4")):
        fig.update_xaxes(
            title_text="I" if r == 1 else "Re(canonical / Qr)", row=r, col=c
        )
        fig.update_yaxes(
            title_text="Q" if r == 1 else "Im(canonical / Qr)",
            scaleanchor=f"x{suffix}",
            scaleratio=1,
            constrain="domain",
            row=r,
            col=c,
        )
    _style(fig, selections[0][1], 1000)
    fig.update_layout(
        margin={"t": 210, "l": 90, "r": 60, "b": 65},
        legend={"y": 1.06, "font": {"size": 11}},
        updatemenus=[
            {"buttons": buttons, "x": 0, "y": 1.17, "xanchor": "left", "yanchor": "top"}
        ],
    )
    return fig


def _correlation(joint, index):
    row = joint.isel(group=index)
    n, start, offset = (
        int(row.parameter_count),
        int(row.parameter_start),
        int(row.covariance_start),
    )
    covariance = joint.covariance_value.values[offset : offset + n * n].reshape(n, n)
    sd = np.sqrt(np.maximum(0, np.diag(covariance)))
    denominator = np.outer(sd, sd)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0,
    )
    labels = joint.parameter_label.values[start : start + n]
    fig = go.Figure(
        go.Heatmap(
            z=correlation,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            colorbar={"title": "Correlation"},
            hovertemplate="%{x}<br>%{y}<br>Correlation=%{z:.4f}<extra></extra>",
        )
    )
    _style(
        fig,
        "Local parameter correlation<br><sup>Includes within-group gain "
        "uncertainty; excludes cross-group systematics."
        f" · {escape(row.uncertainty_status.item())}</sup>",
        max(650, min(1200, 30 * n)),
    )
    fig.update_layout(margin={"l": 240, "b": 210, "t": 100})
    return fig


class ModelFitGroupViz(PlotlyViz):
    """Inspect one saved joint fit without loading other groups' samples.

    The group selector is explicit; every channel/resonator association in
    that group remains available in the figure's pair selector. This is the
    same diagnostic used in the offline ModelFitPlot browser.
    """

    def __call__(
        self,
        view: KidsModelView,
        quality: xr.Dataset,
        *,
        group: str,
        resonator: str | None = None,
    ) -> go.Figure:
        """Read one group's saved samples and construct its four-panel figure.

        Parameters
        ----------
        view : tolteca_kidsproc.analysis.model_fit.KidsModelView
            Open saved model result.
        quality : xarray.Dataset
            Saved group/channel diagnostics.
        group : str
            Exact group label, local to this block and result.
        resonator : str or None, optional
            Initially displayed component, selected by the caller. None starts
            with the first saved pair. All channel/resonator pairs remain in the
            selector. This choice does not select a primary detector or subtract
            other resonances.

        Returns
        -------
        plotly.graph_objects.Figure
            Measured IQ, joint prediction, residual and pair derotation.
        """
        fit, samples, joint = view.fit, view.samples, view.joint
        positions = np.flatnonzero(joint.group_id.values == group)
        if len(positions) != 1:
            msg = f"Unknown or duplicate model group: {group}"
            raise ValueError(msg)
        index = int(positions[0])
        rows = fit.isel(row=np.flatnonzero(fit.group_id.values == group)).load()
        if resonator is not None:
            selected_rows = rows.resonator_id.values.astype(str) == resonator
            if not np.any(selected_rows):
                msg = f"Resonator {resonator} is absent from model group {group}"
                raise ValueError(msg)
            rows = rows.isel(
                row=np.r_[np.flatnonzero(selected_rows), np.flatnonzero(~selected_rows)]
            )
        selected = samples.isel(
            sample=np.flatnonzero(samples.group_id.values == group)
        ).load()
        fig = _group_figure(
            rows, selected, quality.isel(group=index).load(), joint.isel(group=index)
        )
        # WebGL is supported for all line/marker traces in this bounded view.
        return go.Figure(
            data=[
                go.Scattergl(
                    **{k: v for k, v in t.to_plotly_json().items() if k != "type"}
                )
                for t in fig.data
            ],
            layout=fig.layout,
        )


def _power_figure(power):
    fig = make_subplots(
        1,
        2,
        subplot_titles=(
            "Current - reference power",
            "Amplitude / LUT after common scale removal",
        ),
    )
    for col, name in ((1, "power_delta_db"), (2, "lut_residual_db")):
        if name in power:
            fig.add_scattergl(
                x=power.frequency.values / 1e6,
                y=power[name].values,
                mode="markers",
                showlegend=False,
                customdata=power.chan.values,
                marker_color=_DATA,
                hovertemplate=(
                    "Channel %{customdata}<br>%{x:.6f} MHz · %{y:.3f} dB<extra></extra>"
                ),
                row=1,
                col=col,
            )
        else:
            domain = fig.layout[f"xaxis{col if col > 1 else ''}"].domain
            fig.add_annotation(
                text="Reference not configured",
                showarrow=False,
                x=float(np.mean(domain)),
                y=0.5,
                xref="paper",
                yref="paper",
            )
            fig.update_xaxes(visible=False, row=1, col=col)
            fig.update_yaxes(visible=False, row=1, col=col)
        fig.update_xaxes(title_text="Probe frequency (MHz)", row=1, col=col)
        fig.update_yaxes(title_text="Difference (dB)", row=1, col=col)
    return _style(
        fig,
        "Power diagnostics · acquired comb<br><sup>Drive-fit comparison uses "
        "full-comb peak-normalized power fractions.</sup>",
        550,
    )


class ModelFitPlotArtifacts(ArtifactSpec[ArtifactContext]):
    """Offline HTML browser, summary, group fits and local correlations.

    Parameters
    ----------
    figure : PlotlyViz, optional
        Export settings for individual figures; HTML by default. The HTML
        browser shares a local Plotly library, following the drive-fit browser.
    groups : list of str or None, optional
        Group details to construct. None includes all groups. The summary and
        channel warnings always cover the complete acquired block.
    """

    figure: PlotlyViz = Field(default_factory=PlotlyViz)
    groups: list[str] | None = None

    def __call__(self, view, quality, power=None):
        """Load saved results once and retain figures independent of source files.

        Parameters
        ----------
        view : tolteca_kidsproc.analysis.model_fit.KidsModelView
            Saved Python fit; keep storage open until construction returns.
        quality : xarray.Dataset
            Saved quality metrics and warning masks in the same group order.
        power : xarray.Dataset or None, optional
            Acquired-comb diagnostics. None omits that figure.

        Returns
        -------
        tollan.pipeline.ArtifactContext
            Named Plotly figures with summary navigation metadata.

        Raises
        ------
        ValueError
            The calibration or requested group selection is inconsistent.
        """
        view.data_source.load()
        quality = quality.load()
        if power is not None:
            power = power.load()
        view.validate()
        fit, samples, joint = view.fit.load(), view.samples.load(), view.joint.load()
        result = ArtifactContext()
        summary = ModelFitSummaryViz()(view, quality)
        result.add(self.figure.model_copy(update={"alias": "summary"}), summary)
        if power is not None:
            result.add(
                self.figure.model_copy(update={"alias": "power"}), _power_figure(power)
            )
        groups = joint.group_id.values.tolist() if self.groups is None else self.groups
        if len(set(groups)) != len(groups) or not set(groups).issubset(
            joint.group_id.values
        ):
            msg = "Plot groups must be unique existing group IDs"
            raise ValueError(msg)
        records = []
        for index, group in enumerate(joint.group_id.values):
            rows = fit.isel(row=np.flatnonzero(fit.group_id.values == group))
            q = quality.isel(group=index)
            channels = list(dict.fromkeys(rows.chan.values.tolist()))
            flag = int(q.group_flag)
            record = {
                "group": str(group),
                "channels": channels,
                "flag": flag,
                "flags": flag_names(flag),
                "residual": float(q.residual_fraction),
                "condition": float(joint.condition[index]),
                "stem": f"group_{index:04d}" if group in groups else "",
            }
            records.append(record)
            if group not in groups:
                continue
            ss = samples.isel(sample=np.flatnonzero(samples.group_id.values == group))
            for suffix, fig in (
                ("", _group_figure(rows, ss, q, joint.isel(group=index))),
                ("_correlation", _correlation(joint, index)),
            ):
                result.add(
                    self.figure.model_copy(update={"alias": record["stem"] + suffix}),
                    fig,
                )
        summary.update_layout(
            meta={
                "groups": records,
                "channels": [
                    {
                        "chan": c.item() if hasattr(c, "item") else c,
                        "flag": int(f),
                        "flags": flag_names(f),
                    }
                    for c, f in zip(
                        quality.chan.values, quality.channel_flag.values, strict=True
                    )
                ],
            }
        )
        return result

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """Write HTML navigation and typed receipts for every retained figure.

        Parameters
        ----------
        data : tollan.pipeline.ArtifactContext
            Figures constructed by this specification.
        path : pathlib.Path
            Artifact directory. Keep the HTML browser and its sibling directory
            together. Existing files with the same names are overwritten.

        Returns
        -------
        list of tollan.pipeline.ArtifactFile
            Browser, figures and local JavaScript asset receipts. Non-HTML
            formats use the ordinary Plotly writer without an HTML browser.
        """
        return write_model_fit_browser(data, path=path)

    def show(self, data: ArtifactContext) -> None:
        """Display retained figures using the common tabbed Plotly viewer.

        Parameters
        ----------
        data : tollan.pipeline.ArtifactContext
            Previously constructed figures.

        Returns
        -------
        None
            Returns when the local viewer stops.
        """
        self.figure.show_plotly(
            {item.spec.alias or str(key): item.data for key, item in data.items()}
        )
