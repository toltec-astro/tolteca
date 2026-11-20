"""Shared standalone and web figures for saved raw noise diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field
from tollan.pipeline import ArtifactContext, ArtifactFile, ArtifactSpec
from tollan.plot.plotly import (
    PlotlyFigExport,
    PlotlyFigExportFormat,
    PlotlyFigShowInDash,
    PlotlyViz,
    SubplotGrid,
    make_subplots,
)

from ._plot import FIG_LAYOUT

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import plotly.graph_objects as go

    from .timestream_check import TimestreamCheckView


class TimestreamNoisePSDViz(PlotlyViz):
    """I/Q/Sphi waterfalls and band statistics with a channel-frequency navigator.

    Parameters
    ----------
    height : int, default 1800
        Figure height in pixels.
    """

    height: int = Field(default=1800, ge=400)

    def __call__(
        self, view: TimestreamCheckView, *, channels: Sequence[int] | None = None
    ) -> go.Figure:
        """Plot saved spectra without opening raw inputs or rerunning analysis.

        Parameters
        ----------
        view : TimestreamCheckView
            Saved check; keep its source open until this call returns.
        channels : sequence of int or None, optional
            Positional channel selection. None plots every channel. Stored
            channel IDs label the plots and frequency-range slider.

        Returns
        -------
        plotly.graph_objects.Figure
            Loaded seven-panel figure usable in Dash or standalone HTML.
        """
        schema = view.mapper.schema
        dim = view.mapper.get_name(schema.chan_dim)
        selected = {} if channels is None else {dim: list(channels)}
        psd = view.psd
        spectra = psd.psd.isel(selected).compute()
        medians = psd.psd_median.isel(selected).compute()
        mads = psd.psd_mad_std.isel(selected).compute()
        normalized = view.normalized.isel(selected).compute()
        ids = view.mapper.get_arr(view.data_source, schema.chan).isel(selected).values
        frequencies = psd.f_psd.values
        grid = SubplotGrid(fig_layout=FIG_LAYOUT)
        for name, values, median, mad, color in (
            ("I PSD", spectra.I, medians.I, mads.I, "blue"),
            ("Q PSD", spectra.Q, medians.Q, mads.Q, "green"),
            (
                "Sphi",
                normalized.Sphi,
                normalized.Sphi_median,
                normalized.Sphi_mad_std,
                "red",
            ),
        ):
            fig = make_subplots(1, 1, fig_layout=FIG_LAYOUT)
            fig.add_heatmap(
                x=ids,
                y=frequencies,
                z=np.log10(values.values.T),
                colorscale="Viridis",
                name=name,
                colorbar={"title": f"log10({name})"},
                row=1,
                col=1,
            )
            fig.update_yaxes(title={"text": "Frequency (Hz)"}, row=1, col=1)
            fig.update_xaxes(title={"text": "Channel ID"}, row=1, col=1)
            grid.add_subplot(grid.shape[0] + 1, 1, fig=fig, row_height=1)
            fig = make_subplots(1, 1, fig_layout=FIG_LAYOUT)
            fig.add_scattergl(
                x=ids,
                y=median.values,
                error_y={
                    "type": "data",
                    "array": mad.values,
                    "width": 0,
                    "color": "gray",
                },
                mode="markers",
                marker={"size": 4, "color": color},
                name=f"{name} Med",
                row=1,
                col=1,
            )
            # Native ADC counts have no unit attribute. Keep the established v2
            # label for these raw readout counts.
            label = "1/Hz" if name == "Sphi" else "adu^2/Hz"
            if view.config.psd.scaling != "density":
                label = "1" if name == "Sphi" else "adu^2"
            fig.update_yaxes(
                title={"text": f"{name} ({label})"}, type="log", row=1, col=1
            )
            fig.update_xaxes(title={"text": "Channel ID"}, row=1, col=1)
            grid.add_subplot(grid.shape[0] + 1, 1, fig=fig, row_height=0.5)
        fig = make_subplots(1, 1, fig_layout=FIG_LAYOUT)
        fig.add_scatter(
            x=ids,
            y=normalized.f_chan.u.to("MHz").values,
            mode="markers",
            marker={"size": 4, "color": "purple"},
            name="Chan Freq",
            row=1,
            col=1,
        )
        fig.update_yaxes(title={"text": "Chan Frequency (MHz)"}, row=1, col=1)
        fig.update_xaxes(title={"text": "Channel ID"}, row=1, col=1)
        grid.add_subplot(7, 1, fig=fig, row_height=0.5)
        figure = grid.make_figure(
            shared_xaxes="all",
            vertical_spacing=40 / self.height,
            fig_layout={"height": self.height, "showlegend": False},
        )
        # Plotly's native slider requires a scatter (SVG) overview trace. Large
        # channel scatter panels above use WebGL; only this navigator uses SVG.
        figure.update_xaxes(
            rangeslider={
                "autorange": True,
                "range": [int(ids.min()), int(ids.max()) + 1],
                "thickness": 0.05,
            },
            row=7,
            col=1,
        )
        return figure


class TimestreamCheckArtifacts(ArtifactSpec[ArtifactContext]):
    """Noise figure construction with shared export and local display settings."""

    noise_psd: TimestreamNoisePSDViz = Field(default_factory=TimestreamNoisePSDViz)
    save: bool = True
    save_formats: list[PlotlyFigExportFormat] = Field(default=["html"])
    save_plotly: PlotlyFigExport = Field(default_factory=PlotlyFigExport)
    show_plotly: PlotlyFigShowInDash = Field(default_factory=PlotlyFigShowInDash)

    def __call__(self, view: TimestreamCheckView) -> ArtifactContext:
        """Construct the noise figure and retain its effective export settings.

        Parameters
        ----------
        view : TimestreamCheckView
            Saved result with an open source.

        Returns
        -------
        ArtifactContext
            Loaded noise figure, retrievable by TimestreamNoisePSDViz.
        """
        spec = self.noise_psd.model_copy(
            update={
                "save": self.save and self.noise_psd.save,
                "save_formats": self.save_formats,
                "save_plotly": self.save_plotly,
                "show_plotly": self.show_plotly,
            }
        )
        result = ArtifactContext()
        result.add(spec, spec(view))
        return result

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """Export retained figures under a supplied directory.

        Parameters
        ----------
        data : ArtifactContext
            Figures returned by this spec.
        path : pathlib.Path
            Destination directory; the noise plot uses the stem ``noise_psd``.

        Returns
        -------
        list of ArtifactFile
            Exported-file receipts.
        """
        artifact = data[self.noise_psd]
        return artifact.spec.write(artifact.data, path=path / "noise_psd")

    def show(self, data: ArtifactContext) -> None:
        """Display the noise figure in a local Dash app.

        Parameters
        ----------
        data : ArtifactContext
            Figures returned by this spec.

        Returns
        -------
        None
            Returns when the display server stops.
        """
        self.show_plotly(
            {"noise_psd": data[self.noise_psd].data}, title="Timestream check"
        )
