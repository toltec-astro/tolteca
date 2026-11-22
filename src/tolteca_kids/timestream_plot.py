"""Retained Plotly figures from saved post-solving diagnostics."""

from html import escape

import numpy as np
from plotly.offline import get_plotlyjs
from pydantic import ConfigDict, Field
from tollan.pipeline import (
    ArtifactContext,
    ArtifactFile,
    ArtifactSpec,
    Step,
    StepConfig,
    StepContext,
)
from tollan.plot.plotly import PlotlyViz, make_subplots

from ._plot import html_document


class TimestreamSummaryViz(PlotlyViz):
    """Full-network accounting, separate from response-qualified subsets."""

    def __call__(self, ledger):
        """Return status counts and measured coordinate-noise comparisons."""
        fig = make_subplots(
            1,
            2,
            subplot_titles=("All fitted components", "5-20 Hz coordinate ASD ratio"),
        )
        states, counts = np.unique(ledger.spectral_status.values, return_counts=True)
        fig.add_bar(x=states, y=counts, name="Components", row=1, col=1)
        for multiplicity, label in ((1, "Singlet"), (2, "Multiplet")):
            mask = (
                ledger.multiplicity == 1
                if multiplicity == 1
                else ledger.multiplicity > 1
            )
            rows = ledger.isel(component=np.flatnonzero(mask.values))
            fig.add_scatter(
                x=rows.a_fit.values,
                y=rows.coordinate_asd_ratio.values,
                mode="markers",
                name=label,
                row=1,
                col=2,
                customdata=np.column_stack(
                    [rows.group_id, rows.resonator_id, rows.cohort]
                ),
                hovertemplate=(
                    "Group %{customdata[0]} · detector %{customdata[1]}<br>"
                    "%{customdata[2]}<br>a=%{x:.3g}, ratio=%{y:.3g}<extra></extra>"
                ),
            )
        fig.update_xaxes(title_text="Fitted a", row=1, col=2)
        fig.update_yaxes(title_text="ASD(x_opt) / ASD(x')", row=1, col=2)
        fig.update_layout(
            template="plotly_white",
            height=600,
            title="Post-solving network diagnostics · all fitted components",
            font={"size": 16},
        )
        return fig


class TimestreamGroupViz(PlotlyViz):
    """Detector trajectories and signed spectral terms at declared probe frequencies."""

    def __call__(self, node):
        """Return one group figure using evaluated coordinates and measured spectra."""
        model = node.attrs["model"]
        fig = make_subplots(
            3,
            2,
            subplot_titles=(
                "r versus x_opt",
                "x' and x_opt versus time",
                "Coordinate PSD",
                "Signed local PSD terms",
                "x' / r cross-spectrum",
                "x' / r coherence",
            ),
        )
        coordinates = node["coordinates"].to_dataset(inherit=False)
        stride = max(1, int(np.ceil(coordinates.sizes["sample"] / 1000)))
        for j, rid in enumerate(model["resonator_ids"]):
            k = int(np.argmin(abs(np.asarray(model["probes"]) - model["fr"][j])))
            pair = coordinates.isel(
                chan=k, resonator=j, sample=slice(None, None, stride)
            )
            label = f"Detector {rid} · probe {model['channels'][k]}"
            fig.add_scattergl(
                x=pair.x_opt.values,
                y=pair.r.values,
                mode="markers",
                name=label,
                row=1,
                col=1,
            )
            for name, symbol in (("x_prime", "x'"), ("x_opt", "x_opt")):
                fig.add_scattergl(
                    x=pair.time.values,
                    y=pair[name].values,
                    mode="lines",
                    name=f"{rid} {symbol}",
                    row=1,
                    col=2,
                )
            if "spectra" not in node.children or str(j) not in node["spectra"].children:
                continue
            ds = node[f"spectra/{j}"].to_dataset(inherit=False)
            for name in ("x_prime", "x_opt", "r"):
                fig.add_scatter(
                    x=ds.f_psd.values,
                    y=ds.cross_real.sel(coordinate=name, coordinate2=name).values,
                    name=f"{rid} {name}",
                    row=2,
                    col=1,
                )
            for name in ds.term.values:
                fig.add_scatter(
                    x=ds.f_psd.values,
                    y=ds.linear_terms.sel(term=name).values,
                    name=f"{rid} {name} term",
                    row=2,
                    col=2,
                )
            cross = ds.sel(coordinate="x_prime", coordinate2="r")
            for name in ("cross_real", "cross_imag"):
                fig.add_scatter(
                    x=ds.f_psd.values,
                    y=cross[name].values,
                    name=f"{rid} {name}",
                    row=3,
                    col=1,
                )
            fig.add_scatter(
                x=ds.f_psd.values,
                y=cross.coherence.values,
                name=f"{rid} coherence",
                row=3,
                col=2,
            )
        for row, col, x, y in (
            (1, 1, "x_opt (dimensionless)", "r (dimensionless)"),
            (1, 2, "Time (s)", "Detuning (dimensionless)"),
            (2, 1, "Frequency (Hz)", "PSD (Hz⁻¹)"),
            (2, 2, "Frequency (Hz)", "Signed PSD contribution (Hz⁻¹)"),
            (3, 1, "Frequency (Hz)", "Cross-spectrum (Hz⁻¹)"),
            (3, 2, "Frequency (Hz)", "Magnitude-squared coherence"),
        ):
            fig.update_xaxes(title_text=x, row=row, col=col)
            fig.update_yaxes(title_text=y, row=row, col=col)
        fig.update_xaxes(type="log", row=2)
        fig.update_xaxes(type="log", row=3)
        fig.update_yaxes(range=[0, 1], row=3, col=2)
        fig.update_yaxes(type="log", row=2, col=1)
        fig.update_layout(
            template="plotly_white",
            height=1300,
            font={"size": 16},
            title=(
                f"Group {model['group_id']} · same fitted model · fixed b · "
                f"{model['nonlinear_branch']} branch"
            ),
        )
        return fig


class TimestreamPlotArtifacts(ArtifactSpec[ArtifactContext]):
    """Offline figures sharing one Plotly library, written by product output."""

    def __call__(self, diagnostics):
        """Prepare figures from saved diagnostics without repeating solving."""
        artifacts = ArtifactContext()
        summary = TimestreamSummaryViz(alias="summary")
        artifacts.add(summary, summary(diagnostics.to_dataset(inherit=False)))
        for name, node in diagnostics["groups"].children.items():
            if "coordinates" in node.children:
                spec = TimestreamGroupViz(alias=f"group_{name}")
                artifacts.add(spec, spec(node))
        return artifacts

    def write(self, data, *, path):
        """Materialize retained figures, navigation and a shared offline library."""
        path.mkdir(parents=True, exist_ok=True)
        library = path / "plotly.min.js"
        library.write_text(get_plotlyjs())
        receipts = [ArtifactFile(library, "js", name="plotly")]
        links = []
        for artifact in data.values():
            name, figure = artifact.spec.alias, artifact.data
            target = path / f"{name}.html"
            body = (
                '<nav><a href="timestream.html">All figures</a></nav>'
                + figure.to_html(full_html=False, include_plotlyjs=False)
            )
            target.write_text(
                html_document(name, body, '<script src="plotly.min.js"></script>')
            )
            receipts.append(ArtifactFile(target, "html", name=name))
            links.append(
                f'<li><a href="{name}.html">'
                f"{escape(str(figure.layout.title.text))}</a></li>"
            )
        target = path / "timestream.html"
        target.write_text(
            html_document(
                "Post-solving diagnostics",
                (
                    "<h1>Post-solving diagnostics</h1><p>All fitted components remain "
                    "in detectors.ecsv and array.zarr. Figures use the same fitted "
                    "group model. Missing evidence is not a fit failure. Coordinate "
                    "PSD is not absolute optical sensitivity.</p><ul>"
                )
                + "".join(links)
                + "</ul>",
            )
        )
        return [ArtifactFile(target, "html", name="timestream_browser"), *receipts]


class TimestreamPlotConfig(StepConfig):
    """Retain post-solving figures for DataProdOutput."""

    model_config = ConfigDict(extra="forbid")
    artifacts: TimestreamPlotArtifacts = Field(default_factory=TimestreamPlotArtifacts)


class TimestreamPlotContext(StepContext["TimestreamPlot", TimestreamPlotConfig]):
    """Figures retained independently of their eventual output destination."""


class TimestreamPlot(Step[TimestreamPlotConfig, TimestreamPlotContext]):
    """Plot saved group diagnostics with the common VizBase artifact lifecycle."""

    @classmethod
    def run(cls, data, context):
        """Construct figures without solving, fitting or selecting detectors."""
        context.artifacts = context.config.artifacts(data["timestream_diagnostics"])
