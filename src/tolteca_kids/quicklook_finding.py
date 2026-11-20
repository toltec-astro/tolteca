"""Acquisition detector yield and tuning diagnostics from saved finding results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.table import QTable
from pydantic import Field
from tollan.config import FrozenBaseModel
from tollan.config.types import AbsFilePath  # noqa: TC002 -- Pydantic runtime schema
from tollan.pipeline import ArtifactFile
from tollan.plot import VizBase

from tolteca_datamodels.toltec.types import ToltecInfo

from .kids_find import SegmentBitMask

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from matplotlib.figure import Figure

    from .kids_find_view import KidsFindView


class FindingCounts(FrozenBaseModel):
    """V2 detection counts and yield relative to the participating design APT."""

    n_design: int
    n_found: int
    n_good: int
    n_ok: int
    n_single: int
    n_multiple: int
    n_missed: int
    frac_found: float
    frac_in_tune: float
    frac_ok_tune: float
    qr_median: float | None
    qr_q25: float | None
    qr_q75: float | None


class FindingArray(FrozenBaseModel):
    """Array summary of interfaces with finding results for the selected block."""

    array: str
    interfaces: list[str]
    counts: FindingCounts


@dataclass
class _InterfaceFinding:
    counts: FindingCounts
    qr: np.ndarray
    phi: np.ndarray
    single: np.ndarray
    good: np.ndarray
    footprint: np.ndarray


def _counts(n_design, n_found, n_good, n_ok, n_single, n_multiple, n_missed, qr):  # noqa: PLR0913, PLR0917
    quartiles = np.quantile(qr, [0.25, 0.5, 0.75]) if len(qr) else [None] * 3
    return FindingCounts(
        n_design=n_design,
        n_found=n_found,
        n_good=n_good,
        n_ok=n_ok,
        n_single=n_single,
        n_multiple=n_multiple,
        n_missed=n_missed,
        frac_found=n_found / n_design,
        frac_in_tune=n_good / n_design,
        frac_ok_tune=n_ok / n_design,
        qr_q25=quartiles[0],
        qr_median=quartiles[1],
        qr_q75=quartiles[2],
    )


def _footprint(apt):
    # Preserve the v2 row traversal of orientation-zero detectors.
    ui = np.flatnonzero(apt["ori"] == 0)
    edge = apt[ui]
    vertices = []
    for row in np.sort(np.unique(apt["i"])):
        indices = sorted(np.flatnonzero(edge["i"] == row), key=lambda k: edge[k]["j"])
        if vertices:
            vertices = indices[:1] + vertices + indices[-1:]
        else:
            vertices = indices[:1] + indices[-1:] if len(indices) > 1 else indices
    return np.column_stack(
        [apt[name][ui[vertices]].to_value(u.arcsec) for name in ("x_t", "y_t")]
    )


def _summarize(tables, apt):
    found, probes = tables["kids_find"], tables["chan_prop"]
    qr = np.asarray(found["Qr"])
    if not len(apt):
        msg = "Design APT has no detectors for the selected interface"
        raise ValueError(msg)
    x_off = (found["dist"] / found["f_chan"]).to_value(u.dimensionless_unscaled)
    # V2 identifies VNA by 1000 probe rows, not by the filename or kind flag.
    # Retain that rule and the independent good/ok versus missed masks.
    if len(probes) == 1000:  # noqa: PLR2004
        phi = np.zeros(len(found))
        missed = np.zeros(len(found), dtype=bool)
    else:
        r_med = np.median(0.5 / qr) if len(qr) else np.nan
        phi = np.rad2deg(np.arctan2(x_off, r_med))
        missed = np.abs(found["dist"]) > (88 << u.kHz)
    good = np.abs(phi) < 5  # noqa: PLR2004
    ok = np.abs(phi) < 15  # noqa: PLR2004
    single = (found["bitmask_det"] & SegmentBitMask.blended) == 0
    return _InterfaceFinding(
        counts=_counts(
            len(apt),
            len(found),
            int(good.sum()),
            int(ok.sum()),
            int(np.sum(single & ~missed)),
            int(np.sum(~single & ~missed)),
            int(missed.sum()),
            qr,
        ),
        qr=qr,
        phi=phi,
        single=np.asarray(single),
        good=good,
        footprint=_footprint(apt),
    )


def _aggregate(names, findings):
    selected = [findings[name] for name in names]
    qr = np.concatenate([part.qr for part in selected])
    fields = (
        "n_design",
        "n_found",
        "n_good",
        "n_ok",
        "n_single",
        "n_multiple",
        "n_missed",
    )
    counts = _counts(
        **{key: sum(getattr(part.counts, key) for part in selected) for key in fields},
        qr=qr,
    )
    return counts, qr


@dataclass
class FindingSummaryData:
    """Loaded per-interface diagnostics and their array aggregations."""

    interfaces: dict[str, _InterfaceFinding]
    arrays: list[FindingArray]


class FindingSummaryViz(VizBase["Figure"]):
    """V2 acquisition yield footprints, tuning histograms and Qr distributions.

    Parameters
    ----------
    apt_design : path-like
        Design ECSV with ``nw``, ``i``, ``j``, ``ori``, ``x_t`` and ``y_t``.
        Coordinates must have angular units. All rows contribute to the design
        count, as in v2. Provide the same APT for a numerical yield comparison.
    save : bool, optional
        Write the prepared PNG, default True.
    dpi : int, optional
        PNG resolution, default 100 dots per inch (1600 by 1100 pixels).

    Notes
    -----
    In-tune and OK-tune use strict 5 and 15 degree limits. Missed detections use
    an 88 kHz distance limit. Qr comes from finding; native fitted Qr is not an
    input to this summary. A 1000-row probe table uses the v2 VNA convention of
    zero phase offsets and no missed detections.
    """

    apt_design: AbsFilePath
    save: bool = True
    dpi: int = Field(default=100, gt=0)

    def summarize(
        self, findings: Mapping[str, KidsFindView | None]
    ) -> FindingSummaryData:
        """Collate loaded diagnostics without rerunning finding or matching.

        Parameters
        ----------
        findings : mapping of str to KidsFindView or None
            Selected interface/block results. None denotes unavailable results.
            Keep readers open until this call returns.

        Returns
        -------
        FindingSummaryData
            Loaded values; array denominators include only contributing interfaces.
        """
        return self.summarize_tables(
            {
                name: view.export_tables() if view is not None else None
                for name, view in findings.items()
            }
        )

    def summarize_tables(
        self, tables: Mapping[str, dict[str, QTable] | None]
    ) -> FindingSummaryData:
        """Collate finding and channel tables with the design APT."""
        apt = QTable.read(self.apt_design, format="ascii.ecsv")
        parts = {
            name: _summarize(view, apt[apt["nw"] == ToltecInfo.interface_roach[name]])
            for name, view in tables.items()
            if view is not None
        }
        arrays = []
        for array in ToltecInfo.array_names:
            names = [
                name for name in parts if ToltecInfo.interface_array_name[name] == array
            ]
            if names:
                counts, _ = _aggregate(names, parts)
                arrays.append(
                    FindingArray(array=array, interfaces=names, counts=counts)
                )
        return FindingSummaryData(parts, arrays)

    def __call__(self, data: FindingSummaryData, *, title: str) -> Figure:
        """Render loaded summaries in the v2 three-array/thirteen-readout layout.

        Parameters
        ----------
        data : FindingSummaryData
            Values returned by ``summarize``; input readers may now be closed.
        title : str
            Acquisition identity displayed above the figure.

        Returns
        -------
        matplotlib.figure.Figure
            Standalone figure; no window or pyplot-global figure is created.
        """
        # Configuring a reduction or collating counts does not render figures.
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
        from matplotlib.figure import Figure  # noqa: PLC0415

        figure = Figure(figsize=(16, 11))
        # Lay out with the same renderer used for the published PNG.
        FigureCanvasAgg(figure)
        outer = figure.add_gridspec(2, 1)
        grid_array = outer[0, 0].subgridspec(1, 3, wspace=0.01, hspace=0.01)
        grid_nw = outer[1, 0].subgridspec(2, 7, wspace=0.01, hspace=0.01)
        arrays = _array_axes(figure, grid_array)
        networks = _network_axes(figure, grid_nw)
        for name, ax in networks.items():
            part = data.interfaces.get(name)
            if part is None:
                _no_data(ax)
            else:
                _plot_network(ax, part)
                arrays[ToltecInfo.interface_array_name[name]].fill(
                    *part.footprint.T, color=_cmap(part.counts.frac_found)
                )
            ax.text(1, 1, name, ha="right", va="top", transform=ax.transAxes)
        limits = {"a1100": (8000, 12000), "a1400": (4500, 8000), "a2000": (4000, 6000)}
        by_array = {item.array: item for item in data.arrays}
        for name, ax in arrays.items():
            ax.text(1, 1, name, ha="right", va="top", transform=ax.transAxes)
            if name not in by_array:
                _no_data(ax)
            else:
                item = by_array[name]
                _, qr = _aggregate(item.interfaces, data.interfaces)
                _plot_array(ax, item.counts, qr, limits[name], name == "a1100")
        figure.suptitle(title)
        figure.tight_layout()
        return figure

    def write(self, data: Figure, *, path: Path) -> list[ArtifactFile]:
        """Write the prepared PNG using a filename stem.

        Parameters
        ----------
        data : matplotlib.figure.Figure
            Previously rendered figure.
        path : pathlib.Path
            Destination stem in the caller's publication directory.

        Returns
        -------
        list of ArtifactFile
            PNG receipt, or an empty list when saving is disabled.
        """
        if not self.save:
            return []
        target = path.with_name(f"{path.name}.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        data.savefig(target, dpi=self.dpi)
        return [
            ArtifactFile(
                path=target, format="png", name=self.alias or type(self).__name__
            )
        ]


def _cmap(value):
    import matplotlib as mpl  # noqa: PLC0415 -- rendering-only dependency

    return mpl.colormaps["RdYlGn"](value)


def _no_data(ax):
    ax.text(
        0.5,
        0.5,
        "NO DATA",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=_cmap(0.0),
    )


def _array_axes(figure, grid):
    axes = {}
    for index, name in enumerate(ToltecInfo.array_names):
        first = next(iter(axes.values()), None)
        ax = figure.add_subplot(grid[0, index], sharex=first, sharey=first)
        ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=index == 0,
            labelbottom=False,
            labeltop=index == 0,
        )
        ax.tick_params(
            axis="y",
            which="both",
            left=index == 0,
            right=False,
            labelleft=index == 0,
            labelright=False,
        )
        ax.set_aspect("equal")
        if index == 0:
            ax.set_xlabel(r"$\Delta lon$ (arcsec)")
            ax.set_ylabel(r"$\Delta lat$ (arcsec)")
            half = (ToltecInfo.fov_diameter / 2).to_value(u.arcsec)
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
        axes[name] = ax
    return axes


def _network_axes(figure, grid):
    axes = {}
    for nw, name in enumerate(ToltecInfo.roach_interfaces):
        first = next(iter(axes.values()), None)
        row, col = (0, nw) if nw < 7 else (1, nw - 7)  # noqa: PLR2004
        ax = figure.add_subplot(grid[row, col], sharex=first, sharey=first)
        if row == 1 and col == 0:
            ax.set_xlabel(r"Phase Offset $atan^{-1}(x / r)$ (deg)")
            ax.set_ylabel("Num. Dets")
            ax.set_xlim(-90, 90)
            ax.set_ylim(0, 350)
        else:
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
                labeltop=False,
            )
            if col != 0:
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                    labelright=False,
                )
        axes[name] = ax
    return axes


def _plot_network(ax, part):
    ax.axvspan(-90, -15, color="gray", alpha=0.2)
    ax.axvspan(15, 90, color="gray", alpha=0.2)
    ax.axvspan(-15, -5, color="#cdcdcd", alpha=0.2)
    ax.axvspan(5, 15, color="#cdcdcd", alpha=0.2)
    ax.axvline(0.0, linestyle="-", color="black", linewidth=1)
    single, good = part.single, part.good
    ax.hist(
        [
            part.phi[mask]
            for mask in (single & good, ~single & good, single & ~good, ~single & ~good)
        ],
        bins=np.arange(-92.5, 92.6, 5),
        color=[_cmap(1.0), _cmap(0), _cmap(0.75), _cmap(0.25)],
        stacked=True,
    )
    c = part.counts
    text = (
        f"  Design: {c.n_design:3d}\n   Found: {c.n_found:3d}\n"
        f" In-tune: {c.n_good:3d}\n OK-tune: {c.n_ok:3d}\n"
        f"  Single: {c.n_single:3d}\nMultiple: {c.n_multiple:3d}\n"
        f"  Missed: {c.n_missed:3d}\n"
    )
    ax.text(
        0,
        1,
        text.strip("\n"),
        ha="left",
        va="top",
        fontsize=8,
        fontfamily="monospace",
        transform=ax.transAxes,
    )


def _plot_array(ax, counts, qr, limits, label):
    from matplotlib.transforms import blended_transform_factory  # noqa: PLC0415
    from mpl_toolkits.axes_grid1.axes_divider import (  # noqa: PLC0415
        make_axes_locatable,
    )

    c = counts
    text = (
        f"Found: {c.n_found:3d} / {c.n_design:3d} {c.frac_found:4.1%}\n"
        f"In-tune: {c.n_good:3d}/ {c.n_design:3d} {c.frac_in_tune:4.1%}\n"
        f"OK-tune: {c.n_ok:3d}/ {c.n_design:3d} {c.frac_ok_tune:4.1%}"
    )
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontfamily="monospace",
        transform=ax.transAxes,
    )
    ax = make_axes_locatable(ax).append_axes("bottom", size="10%", pad="10%")
    left, right = limits
    ax.set_xlim(left - 3000, right + 3000)
    ax.set_ylim(-0.5, 0.75)
    if label:
        ax.set_ylabel("Qr")
        ax.yaxis.label.set(rotation="horizontal", ha="right", va="center")
    ax.get_yaxis().set_ticks([])
    ax.axvspan(ax.get_xlim()[0], left, color="#cdcdcd")
    ax.axvspan(right, ax.get_xlim()[1], color="#cdcdcd")
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for x, text, ha in (
        (left - 1000, "ROOM", "right"),
        (right + 1000, "DARK", "left"),
        ((left + right) * 0.5, "SKY", "center"),
    ):
        ax.text(x, 0.95, text, ha=ha, va="top", transform=transform)
    if not len(qr):
        return
    parts = ax.violinplot(
        [qr],
        positions=[0],
        orientation="horizontal",
        showextrema=False,
        showmedians=False,
    )
    color = _cmap(float(c.qr_median > left))
    ax.errorbar(
        [c.qr_median],
        [0],
        xerr=[[c.qr_median - c.qr_q25], [c.qr_q75 - c.qr_median]],
        marker="*",
        color=color,
        ms=20,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
