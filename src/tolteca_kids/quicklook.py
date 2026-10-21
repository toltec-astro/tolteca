"""Acquisition feedback from saved sweep checks.

``KidsQuicklook`` prepares a summary of the selected interfaces. Its artifact
contains a portable JSON descriptor and a Plotly figure; data-product output
writes both in the same publication as the reduced arrays. No detector finding
is performed. Channel counts describe sweep-quality flags, not detector yield.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import Field
from tollan.config import FrozenBaseModel
from tollan.pipeline import ArtifactFile, ArtifactSpec
from tollan.plot.plotly import PlotlyFigExportFormat, PlotlyViz

from tolteca_datamodels.toltec.types import ToltecDataKind

from .sweep_check import SweepCheck
from .sweep_check_view import SweepCheckView

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import xarray as xr


class QuicklookInterface(FrozenBaseModel):
    """Sweep-check coverage for one selected readout interface."""

    interface: str
    blocks: list[int]
    block: int | None
    n_channels: int | None = None
    n_flagged: int | None = None


class QuicklookDocument(FrozenBaseModel):
    """Portable feedback with plot paths relative to this document."""

    format: Literal["tolteca.kids.quicklook.v1"] = "tolteca.kids.quicklook.v1"
    title: str
    acquisitions: list[str]
    interfaces: list[QuicklookInterface]
    plots: list[str]


@dataclass
class QuicklookResult:
    """Loaded acquisition feedback and its summary figure."""

    document: QuicklookDocument
    figure: go.Figure


class SweepCheckSummaryViz(PlotlyViz):
    """Per-interface channel level, uncertainty and D21 summary.

    Parameters
    ----------
    row_height : int, optional
        Height in pixels per interface, default 250.
    """

    save_formats: list[PlotlyFigExportFormat] = Field(default=["png"])
    row_height: int = Field(default=250, ge=150)

    def __call__(
        self, checks: Mapping[str, SweepCheckView | None], *, title: str
    ) -> go.Figure:
        """Build a figure from selected checks without changing their data.

        Parameters
        ----------
        checks : mapping of str to SweepCheckView or None
            Readout interfaces in display order. None shows missing coverage.
        title : str
            Acquisition identity and feedback heading.

        Returns
        -------
        plotly.graph_objects.Figure
            Loaded figure that can outlive the input readers.
        """
        figure = make_subplots(
            rows=len(checks),
            cols=3,
            subplot_titles=[
                f"{name} · {label}"
                for name in checks
                for label in (
                    "level (dB)",
                    "RMS (dB)",
                    "D21 (1/Hz) · baseline in orange",
                )
            ],
            vertical_spacing=min(0.3, 80 / (self.row_height * len(checks))),
        )
        for row, (name, check) in enumerate(checks.items(), 1):
            if check is None:
                figure.add_annotation(
                    text="No sweep check",
                    showarrow=False,
                    row=row,
                    col=1,
                )
                continue
            sweep = check.sweep
            channels = sweep.mapper.get_arr(
                sweep.data_source, sweep.mapper.schema.chan
            ).values
            colors = np.where(check.mask_chan_bad, "#e03131", "#1971c2")
            for col, values in enumerate((check.chan_level, check.chan_rms_mean), 1):
                figure.add_trace(
                    go.Scatter(
                        x=channels,
                        y=values,
                        mode="markers",
                        marker={"size": 3, "color": colors},
                        name=name,
                    ),
                    row=row,
                    col=col,
                )
                figure.update_xaxes(title_text="Channel ID", row=row, col=col)
            frequency = check.d21_frequency.to_value("MHz")
            for values, label, color in (
                (check.d21, "D21", "#343a40"),
                (check.d21_baseline, "Baseline", "#e67700"),
            ):
                figure.add_trace(
                    go.Scatter(
                        x=frequency,
                        y=values.to_value("1/Hz"),
                        mode="lines",
                        name=label,
                        line={"color": color},
                    ),
                    row=row,
                    col=3,
                )
            figure.update_xaxes(title_text="Frequency (MHz)", row=row, col=3)
        figure.update_layout(
            title=(
                f"{title}<br><sup>Red: flagged channel · blue: unflagged channel</sup>"
            ),
            height=120 + self.row_height * len(checks),
            width=1600,
            showlegend=False,
            template="plotly_white",
        )
        return figure


class KidsQuicklook(ArtifactSpec[QuicklookResult]):
    """Prepare acquisition feedback from existing block checks.

    Parameters
    ----------
    enabled : bool, optional
        Enable feedback in the KIDs runner, default False.
    block : int, optional
        Checked block to display on each interface. -1 selects the last sweep
        block (default). The descriptor also lists all checked blocks.
    summary : SweepCheckSummaryViz, optional
        Figure construction and export settings.
    """

    enabled: bool = False
    block: int = Field(default=-1, ge=-1)
    summary: SweepCheckSummaryViz = Field(default_factory=SweepCheckSummaryViz)

    def __call__(self, trees: Mapping[str, xr.DataTree]) -> QuicklookResult:
        """Prepare feedback for selected interfaces of one acquisition.

        Parameters
        ----------
        trees : mapping of str to xarray.DataTree
            Acquisition trees keyed by interface. Keep readers open until this
            call returns. Missing checks remain explicit in the result.

        Returns
        -------
        QuicklookResult
            Loaded summary and descriptor, ready for publication.
        """
        checks = {}
        interfaces = []
        for name, tree in trees.items():
            blocks = [
                block
                for block in tree.toltec_kids.blocks
                if block.data_kind & ToltecDataKind.sweep
            ]
            available = {
                block.meta.block: SweepCheckView(block.tree[SweepCheck.result_path()])
                for block in blocks
                if SweepCheck.result_path() in block.tree.children
            }
            selected = (
                max((block.meta.block for block in blocks), default=None)
                if self.block == -1
                else self.block
            )
            check = available.get(selected)
            checks[name] = check
            interfaces.append(
                QuicklookInterface(
                    interface=name,
                    blocks=list(available),
                    block=selected if check is not None else None,
                    n_channels=None if check is None else len(check.mask_chan_bad),
                    n_flagged=None
                    if check is None
                    else int(np.count_nonzero(check.mask_chan_bad)),
                )
            )
        meta = next(iter(trees.values())).toltec_kids.meta
        title = f"{meta.uid_raw_obs} · Sweep check"
        document = QuicklookDocument(
            title=title,
            acquisitions=[meta.uid_raw_obs],
            interfaces=interfaces,
            plots=[f"summary.{fmt}" for fmt in self.summary.save_formats]
            if self.summary.save
            else [],
        )
        return QuicklookResult(
            document=document, figure=self.summary(checks, title=title)
        )

    def write(self, data: QuicklookResult, *, path: Path) -> list[ArtifactFile]:
        """Write prepared PNG and JSON feedback beneath a publication directory.

        Parameters
        ----------
        data : QuicklookResult
            Previously prepared feedback.
        path : pathlib.Path
            Directory owned by the caller's staging publication.

        Returns
        -------
        list of ArtifactFile
            PNG and JSON receipts. The JSON has format ``kids_quicklook``.
        """
        images = self.summary.write(data.figure, path=path / "summary")
        path.mkdir(parents=True, exist_ok=True)
        document = path / "feedback.json"
        document.write_text(data.document.model_dump_json(indent=2))
        return [*images, ArtifactFile(path=document, format="kids_quicklook")]
