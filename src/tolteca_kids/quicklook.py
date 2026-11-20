"""KIDs feedback artifacts from explicitly selected sweep block results.

Every selected acquisition/interface/block retains its own diagnostics. Panels
collate interfaces within one acquisition and block number; they do not assume
that channel IDs or tone settings match between blocks. Processing and product
discovery belong to the quicklook workflow, not this artifact specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import Field
from tollan.accessor.pandas.model_table import ModelTableAccessor
from tollan.config import FrozenBaseModel
from tollan.pipeline import ArtifactFile, ArtifactSpec
from tollan.plot.plotly import PlotlyFigExportFormat, PlotlyViz

from tolteca_datamodels.toltec.types import ToltecDataKind

from .kids_find import KidsFind
from .quicklook_finding import FindingArray, FindingCounts, FindingSummaryViz
from .sweep_check import SweepCheck

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    import xarray as xr
    from matplotlib.figure import Figure

    from .sweep_check_view import SweepCheckView


class QuicklookInterface(FrozenBaseModel):
    """Sweep check and detector finding on one readout interface."""

    interface: str
    check_status: Literal["available", "missing", "failed"]
    finding_status: Literal["available", "missing", "failed", "not_requested"]
    n_channels: int | None = None
    n_flagged: int | None = None
    finding: FindingCounts | None = None


class QuicklookBlock(QuicklookInterface):
    """Interface measurements scoped to an acquired sweep block."""

    uid_raw_obs: str
    block: int


class QuicklookArray(FindingArray):
    """Array aggregation within one acquisition and acquired block number."""

    uid_raw_obs: str
    block: int


def _panel_stem(uid, block):
    return f"{uid}/blocks/{block}"


class QuicklookDocument(FrozenBaseModel):
    """Portable multi-block feedback with relative plot paths."""

    format: Literal["tolteca.kids.quicklook.v1"] = "tolteca.kids.quicklook.v1"
    title: str
    acquisitions: list[str]
    blocks: list[QuicklookBlock]
    plots: list[str]
    arrays: list[QuicklookArray] = Field(default_factory=list)

    def block_plots(self, uid_raw_obs: str, block: int) -> dict[str, list[str]]:
        """Return this sweep block's sweep-check and detector-finding plot paths.

        Keys are the two panel names, ``summary`` and ``finding``; lists include
        each saved format. Paths are relative to the descriptor.
        """
        parent = PurePosixPath(_panel_stem(uid_raw_obs, block))
        result = {"summary": [], "finding": []}
        for name in self.plots:
            path = PurePosixPath(name)
            if path.parent == parent:
                result[path.stem].append(name)
        return result


@dataclass
class QuicklookInput:
    """Borrowed acquisition tree and one acquired sweep block.

    Keep the tree open until the artifact specification returns. The same input
    contract accepts freshly processed and reopened scientific trees.
    """

    tree: xr.DataTree
    block: int

    @property
    def identity(self) -> tuple[str, str, int]:
        """Acquisition, interface and acquired block identifier."""
        meta = self.tree.toltec_kids.meta
        return meta.uid_raw_obs, meta.interface, self.block

    @property
    def data(self) -> xr.DataTree:
        """Selected block's scientific tree."""
        return self.tree.toltec_kids.blocks[self.block].tree


def select_quicklook_inputs(
    trees: Iterable[xr.DataTree], *, select: str | None = None
) -> list[QuicklookInput]:
    """Select sweep blocks from borrowed acquisition trees.

    Parameters
    ----------
    trees : iterable of xarray.DataTree
        Open acquisition trees, potentially spanning acquisitions and interfaces.
    select : str or None, optional
        Pandas expression on uid_raw_obs, interface, block and obsnum. None
        selects all acquired sweep blocks, including both Tune sweeps.

    Returns
    -------
    list of QuicklookInput
        Selected blocks in supplied acquisition and acquired block order.

    Raises
    ------
    ValueError
        Selection is empty or two inputs identify the same acquisition block.
    """
    rows = []
    for tree in trees:
        meta = tree.toltec_kids.meta
        rows.extend(
            {
                "uid_raw_obs": meta.uid_raw_obs,
                "interface": meta.interface,
                "obsnum": meta.obsnum,
                "block": block.meta.block,
                "input": QuicklookInput(tree, block.meta.block),
            }
            for block in tree.toltec_kids.blocks
            if block.data_kind & ToltecDataKind.sweep
        )
    table = pd.DataFrame(rows)
    if not table.empty and select is not None:
        table = ModelTableAccessor(table).query(select)
    if table.empty:
        msg = "Quicklook requires at least one selected sweep block"
        raise ValueError(msg)
    if table.duplicated(["uid_raw_obs", "interface", "block"]).any():
        msg = "Select one result per acquisition/interface/block"
        raise ValueError(msg)
    return table["input"].tolist()


@dataclass
class QuicklookPanel:
    """Loaded figures for one acquisition and acquired block number."""

    summary: go.Figure | None
    finding: Figure | None


@dataclass
class QuicklookResult:
    """Loaded feedback and panels, independent of input reader lifetimes."""

    document: QuicklookDocument
    panels: dict[str, QuicklookPanel]


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
            ).to_numpy()
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


def _status(step, data):
    if step.has_context(data) and not step.get_context(data).completed:
        return "failed"
    return "available" if step.result_path() in data.children else "missing"


@dataclass
class QuicklookBlockResult:
    """Loaded high-level inputs shared by scientific-tree and inline quicklooks."""

    record: QuicklookBlock
    finding_tables: dict | None
    check: SweepCheckView | None = None

    @property
    def identity(self) -> tuple[str, str, int]:
        """Acquisition, interface and acquired block."""
        return self.record.uid_raw_obs, self.record.interface, self.record.block

    @property
    def block(self) -> int:
        return self.record.block


class KidsQuicklook(ArtifactSpec[QuicklookResult]):
    """Build multi-block feedback from existing sweep checks and findings.

    Parameters
    ----------
    enabled : bool, optional
        Include this artifact in ordinary KIDs product publication, default False.
        The independent quicklook workflow invokes the specification explicitly.
    summary : SweepCheckSummaryViz or None, optional
        Channel-level/RMS/D21 panels. None omits their construction and export.
    finding : FindingSummaryViz or None, optional
        V2 yield, tuning histograms and array Qr panels. Requires a design APT.
        None omits finding summaries. No scientific processing is run here.
    """

    enabled: bool = False
    summary: SweepCheckSummaryViz | None = Field(default_factory=SweepCheckSummaryViz)
    finding: FindingSummaryViz | None = None

    def __call__(
        self, inputs: Iterable[QuicklookInput | QuicklookBlockResult]
    ) -> QuicklookResult:
        """Summarize every selected block and render its configured panels.

        Parameters
        ----------
        inputs : iterable of QuicklookInput
            Explicit block selections. Keep borrowed readers open until return.
            Missing results remain explicit and never select an earlier block.

        Returns
        -------
        QuicklookResult
            Loaded descriptor and figures, safe to use after readers close.

        Raises
        ------
        ValueError
            Selection is empty, duplicated, or contains a non-sweep block.
        """
        groups = {}
        seen = set()
        for item in inputs:
            uid, interface, block = item.identity
            if item.identity in seen:
                msg = "Select one result per acquisition/interface/block"
                raise ValueError(msg)
            seen.add(item.identity)
            result = self._block_result(item)
            groups.setdefault((uid, block), {})[interface] = result
        if not groups:
            msg_0 = "Quicklook requires at least one selected sweep block"
            raise ValueError(msg_0)
        records, arrays, plots, panels = [], [], [], {}
        for (uid, block), data in groups.items():
            checks = {name: item.check for name, item in data.items()}
            values = (
                self.finding.summarize_tables(
                    {name: item.finding_tables for name, item in data.items()}
                )
                if self.finding
                else None
            )
            for name, item in data.items():
                records.append(
                    item.record.model_copy(
                        update={
                            "finding": values.interfaces[name].counts
                            if values and name in values.interfaces
                            else None
                        }
                    )
                )
            if values is not None:
                arrays.extend(
                    QuicklookArray(**a.model_dump(), uid_raw_obs=uid, block=block)
                    for a in values.arrays
                )
            stem = _panel_stem(uid, block)
            panels[stem] = QuicklookPanel(
                summary=self.summary(
                    checks, title=f"{uid} · block {block} · Sweep check"
                )
                if self.summary is not None
                else None,
                finding=self.finding(
                    values, title=f"{uid} · block {block} · KIDs finding"
                )
                if self.finding is not None and values is not None
                else None,
            )
            if self.summary is not None and self.summary.save:
                plots.extend(
                    f"{stem}/summary.{fmt}" for fmt in self.summary.save_formats
                )
            if self.finding is not None and self.finding.save:
                plots.append(f"{stem}/finding.png")
        acquisitions = list(dict.fromkeys(uid for uid, _ in groups))
        return QuicklookResult(
            document=QuicklookDocument(
                title=" · ".join(acquisitions) + " · KIDs feedback",
                acquisitions=acquisitions,
                blocks=records,
                arrays=arrays,
                plots=plots,
            ),
            panels=panels,
        )

    def _block_result(
        self, item: QuicklookInput | QuicklookBlockResult
    ) -> QuicklookBlockResult:
        uid, interface, block = item.identity
        if isinstance(item, QuicklookInput):
            if not item.tree.toltec_kids.blocks[block].data_kind & ToltecDataKind.sweep:
                msg = "KIDs quicklook requires sweep blocks"
                raise ValueError(msg)
            tree = item.data
            check_status = _status(SweepCheck, tree)
            check = SweepCheck.view(tree) if check_status == "available" else None
            status = _status(KidsFind, tree) if self.finding else "not_requested"
            return QuicklookBlockResult(
                QuicklookBlock(
                    uid_raw_obs=uid,
                    interface=interface,
                    block=block,
                    check_status=check_status,
                    finding_status=status,
                    n_channels=len(check.mask_chan_bad) if check is not None else None,
                    n_flagged=int(np.count_nonzero(check.mask_chan_bad))
                    if check is not None
                    else None,
                ),
                KidsFind.view(tree).export_tables() if status == "available" else None,
                check,
            )
        if (
            self.summary is not None
            and item.record.check_status == "available"
            and item.check is None
        ):
            msg = (
                "Channel-level plots need saved sweep arrays. "
                "Select summary: null for inline tables."
            )
            raise ValueError(msg)
        if self.finding is None:
            return QuicklookBlockResult(
                item.record.model_copy(update={"finding_status": "not_requested"}),
                None,
                item.check,
            )
        return item

    def write(self, data: QuicklookResult, *, path: Path) -> list[ArtifactFile]:
        """Write loaded panels and feedback.json into a caller-owned directory.

        Parameters
        ----------
        data : QuicklookResult
            Previously prepared feedback; input readers may be closed.
        path : pathlib.Path
            Destination directory, normally a staged product publication.

        Returns
        -------
        list of ArtifactFile
            Image receipts and the kids_quicklook JSON descriptor.
        """
        images = []
        for stem, panel in data.panels.items():
            if self.summary is not None and panel.summary is not None:
                images.extend(
                    self.summary.write(panel.summary, path=path / stem / "summary")
                )
            if self.finding is not None and panel.finding is not None:
                images.extend(
                    self.finding.write(panel.finding, path=path / stem / "finding")
                )
        path.mkdir(parents=True, exist_ok=True)
        document = path / "feedback.json"
        document.write_text(data.document.model_dump_json(indent=2))
        return [*images, ArtifactFile(path=document, format="kids_quicklook")]
