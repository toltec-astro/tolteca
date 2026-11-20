"""Diagnostic plots for KIDs reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import plotly.colors
import plotly.graph_objects as go
from astropy.table import unique
from tollan.plot.plotly import SubplotGrid, make_subplots

from ._plot import COLOR_PALETTE, FIG_LAYOUT
from .kids_find import SegmentBitMask
from .match1d import _plot_match_density, _plot_match_pairs

if TYPE_CHECKING:
    import xarray as xr

    from .kids_find_view import KidsFindPeakView, KidsFindView
    from .match1d_result import Match1DView
from .sweep_check import _extract_sweep_arrays

__all__ = ["make_kids_find_figs"]


# ── Unit-safe helpers ─────────────────────────────────────────────────────────


def _to_value(arr, unit=None):
    """Strip units from arr, converting to ``unit`` if Quantity, else return as-is."""
    if hasattr(arr, "to_value"):
        if unit is not None:
            return arr.to_value(unit)
        return arr.value
    return np.asarray(arr)


# ── Data extraction helpers ───────────────────────────────────────────────────


def _get_raw_arrays(
    dt: xr.Dataset | xr.DataTree,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequency (MHz) and |S21| (dB) arrays from a DataTree.

    Returns
    -------
    fs : ndarray shape (n_chan, n_steps)  — frequency in MHz
    as21_db : ndarray shape (n_chan, n_steps)  — 20*log10(|S21|)
    as21_unc_db : ndarray shape (n_chan, n_steps)  — uncertainty in dB
    """
    arrays = _extract_sweep_arrays(dt)
    fs = arrays.frequency.to_value(u.MHz)  # (n_chan, n_steps)
    return fs, arrays.aS21_db, arrays.aS21_unc_db


# ── Bitmask / summary heatmaps ────────────────────────────────────────────────


def _make_bitmask_heatmap(bitmask_seg: np.ndarray) -> go.Figure:
    """Binary heatmap of each SegmentBitMask flag across segments."""
    names = []
    data = []
    for name, value in SegmentBitMask.__members__.items():
        names.append(name)
        data.append((bitmask_seg & int(value)) > 0)
    z = np.vstack(data).astype(int)
    fig = make_subplots(1, 1, fig_layout=FIG_LAYOUT)
    fig.add_heatmap(z=z, y=names, colorscale="rdylgn_r", zmin=0, zmax=1)
    fig.update_xaxes(title="Segment Id")
    fig.update_layout(title={"text": "Segment Bitmask"})
    return fig


def _make_seg_data_heatmap(name: str, data: np.ndarray, trace_kw: dict) -> go.Figure:
    """Single-row heatmap for one property across segments."""
    fig = make_subplots(1, 1, fig_layout=FIG_LAYOUT)
    fig.add_heatmap(z=data[np.newaxis, :], y=[name], colorscale="rdylgn_r", **trace_kw)
    fig.update_xaxes(title="Segment Id")
    fig.update_layout(title={"text": name})
    return fig


def _make_summary_fig(bitmask_seg: np.ndarray, seg_data_items: list) -> go.Figure:
    """Composite figure: bitmask row + per-property rows, shared x-axis."""
    grid = SubplotGrid(fig_layout=FIG_LAYOUT)
    grid.add_subplot(
        row=1,
        col=1,
        fig=_make_bitmask_heatmap(bitmask_seg),
        row_height=1,
    )
    row0 = grid.shape[0] + 1
    for i, (name, value, trace_kw) in enumerate(seg_data_items):
        grid.add_subplot(
            row=row0 + i,
            col=1,
            fig=_make_seg_data_heatmap(name, value, trace_kw),
            row_height=0.5 / len(seg_data_items),
        )
    fig = grid.make_figure(
        shared_xaxes="all",
        vertical_spacing=40 / 1200,
        fig_layout={"height": 1200},
    )
    fig.update_xaxes(
        rangeslider={
            "autorange": True,
            "range": [0, bitmask_seg.shape[0]],
            "thickness": 0.05,
        },
        row=grid.shape[0],
        col=1,
    )
    return fig


# ── Peak-level summary figures ────────────────────────────────────────────────


def make_d21_summary_fig(view: KidsFindView) -> go.Figure:
    """Plot D21 peak flags and measured properties.

    Parameters
    ----------
    view : KidsFindView
        Saved finding results; their configuration supplies the color limits.

    Returns
    -------
    plotly.graph_objects.Figure
        Flags, Qr, SNR, FWHM and peak-height heatmaps with a shared range slider.
    """
    ctd = view
    cfg_kf = view.config
    peak_info = ctd.d21_peaks.peaks
    height = _to_value(peak_info["height"], u.Hz**-1)
    seg_data_items = [
        (
            "Qr",
            np.asarray(peak_info["Qr"]),
            {"zmin": cfg_kf.Qr_min, "zmax": cfg_kf.Qr_dark_min},
        ),
        (
            "SNR",
            np.asarray(peak_info["snr"]),
            {"zmin": 0, "zmax": float(np.quantile(peak_info["snr"], 0.9))},
        ),
        ("FWHM", _to_value(peak_info["width"], u.Hz), {"zmin": 1000, "zmax": 120000}),
        ("height", height, {"zmin": 0, "zmax": float(np.quantile(height, 0.9))}),
    ]
    return _make_summary_fig(ctd.d21_peaks.bitmask, seg_data_items)


def make_s21_summary_fig(view: KidsFindView) -> go.Figure:
    """Plot S21 peak flags and measured properties.

    Parameters
    ----------
    view : KidsFindView
        Saved finding results; their configuration supplies the color limits.

    Returns
    -------
    plotly.graph_objects.Figure
        Flags, Qr, SNR, FWHM, linear height and dB-height heatmaps.
    """
    ctd = view
    cfg_kf = view.config
    peak_info = ctd.s21_peaks.peaks
    height = _to_value(peak_info["height"], u.dimensionless_unscaled)
    seg_data_items = [
        (
            "Qr",
            np.asarray(peak_info["Qr"]),
            {"zmin": cfg_kf.Qr_min, "zmax": cfg_kf.Qr_dark_min},
        ),
        (
            "SNR",
            np.asarray(peak_info["snr"]),
            {"zmin": 0, "zmax": float(np.quantile(peak_info["snr"], 0.9))},
        ),
        ("FWHM", _to_value(peak_info["width"], u.Hz), {"zmin": 1000, "zmax": 120000}),
        ("height", height, {"zmin": 0, "zmax": float(np.quantile(height, 0.9))}),
    ]
    seg_data_items.append(
        ("height_db", np.asarray(peak_info["height_db"]), {"zmin": 0, "zmax": 1})
    )
    return _make_summary_fig(ctd.s21_peaks.bitmask, seg_data_items)


def make_det_summary_fig(view: KidsFindView) -> go.Figure:
    """Plot detection flags, frequencies, and quality factors.

    Parameters
    ----------
    view : KidsFindView
        Saved detection results, from ``KidsFind.view``.

    Returns
    -------
    plotly.graph_objects.Figure
        Detection summary.
    """
    ctd = view
    cfg_kf = view.config
    det_groups = ctd.det_groups
    seg_data_items = [
        ("f", _to_value(det_groups["f"], u.MHz), {}),
        (
            "Qr",
            np.asarray(det_groups["Qr"]),
            {"zmin": cfg_kf.Qr_min, "zmax": cfg_kf.Qr_dark_min},
        ),
    ]
    return _make_summary_fig(ctd.det_groups["bitmask"], seg_data_items)


# ── Peaks overview figure (3-panel) ──────────────────────────────────────────


def make_peaks_fig(
    dt: xr.Dataset | xr.DataTree,
    view: KidsFindView,
) -> go.Figure:
    """Plot detected D21/S21 peaks and their checked sweep samples.

    Parameters
    ----------
    dt : xarray.Dataset or xarray.DataTree
        Corrected sweep used for finding, from ``SweepCheck.view(block).sweep``.
    view : KidsFindView
        Saved finding results for that sweep.

    Returns
    -------
    plotly.graph_objects.Figure
        Three panels with peak masks, widths, baseline selection and sample
        uncertainties. The sweep panel displays every fourth sample.
    """
    ctd = view
    fs, as21_db, as21_unc_db = _get_raw_arrays(dt)

    fig = make_subplots(
        3,
        1,
        shared_xaxes="all",
        vertical_spacing=40 / 1000,
        fig_layout=FIG_LAYOUT | {"showlegend": True, "height": 1000},
    )
    d21_kw = {"row": 1, "col": 1}
    s21_kw = {"row": 2, "col": 1}
    s21d_kw = {"row": 3, "col": 1}

    color_cycle = COLOR_PALETTE.cycle()

    def _add_peaks_panel(  # noqa: PLR0913 -- independent panel units and overlays
        name: str,
        peaks: KidsFindPeakView,
        *,
        x_unit: u.UnitBase,
        y_unit: u.UnitBase,
        panel_kw: dict,
        overlay_masks: list,
    ) -> None:
        peak_info = peaks.peaks
        labels = peaks.labels
        x = _to_value(peaks.x, x_unit)
        y = _to_value(peaks.y, y_unit)
        height = _to_value(peak_info["height"], y_unit)
        base = _to_value(peak_info["base"], y_unit)

        for ll in np.unique(labels):
            m = labels == ll
            fig.add_scatter(
                x=x[m],
                y=y[m],
                mode="lines",
                line={"color": next(color_cycle)},
                name=f"peak {ll}",
                showlegend=True,
                **panel_kw,
            )

        customdata_cols = [
            ("label", ".0f"),
            ("snr", ".3f"),
            ("Qr", ".3f"),
            ("height_db", ".3f"),
            ("halfmax_size", ".0f"),
            ("lookahead", ".0f"),
        ] + [(c, ".0f") for c in peak_info.colnames if c.startswith("sbm")]
        customdata_cols = [c for c in customdata_cols if c[0] in peak_info.colnames]

        fig.add_scatter(
            x=_to_value(peak_info["x"], x_unit),
            y=height / 2 + base,
            error_x={
                "type": "data",
                "array": _to_value(peak_info["width"], x_unit) / 2,
                "width": 0,
                "color": "green",
            },
            error_y={"type": "data", "array": height / 2, "width": 0, "color": "green"},
            customdata=np.stack([peak_info[ci[0]] for ci in customdata_cols]).T,
            hovertemplate=(
                "f: %{x:.3f}<br>d21: %{y:.3f}"
                + "".join(
                    f"<br>{c[0]}: %{{customdata[{i}]:{c[1]}}}"
                    for i, c in enumerate(customdata_cols)
                )
            ),
            mode="markers",
            marker={"color": "orange", "size": 4},
            name="peak info",
            **panel_kw,
        )

        for mask_name, mask, mask_color in overlay_masks:
            fig.add_scatter(
                x=x[mask],
                y=y[mask],
                mode="lines",
                line={"color": mask_color},
                name=mask_name,
                **panel_kw,
            )

        fig.update_yaxes(title={"text": f"{name} ({y_unit})"}, **panel_kw)

    _add_peaks_panel(
        "|D21|",
        ctd.d21_peaks,
        x_unit=u.MHz,
        y_unit=u.Hz**-1,
        panel_kw=d21_kw,
        overlay_masks=[
            ("not real", ctd.d21_peaks.mask_not_real, "gray"),
            ("dark", ctd.d21_peaks.mask_dark, "red"),
            ("baseline", ctd.d21_peaks.mask_baseline, "black"),
        ],
    )

    _add_peaks_panel(
        "|S21|",
        ctd.s21_peaks,
        x_unit=u.MHz,
        y_unit=u.dimensionless_unscaled,
        panel_kw=s21_kw,
        overlay_masks=[
            ("not real", ctd.s21_peaks.mask_not_real, "gray"),
            ("edge", ctd.s21_peaks.mask_edge.ravel(), "cyan"),
        ],
    )

    # D21-detected vertical lines on S21 panel
    d21_dets = ctd.d21_peaks.peaks[~np.asarray(ctd.d21_peaks.peaks["sbm_not_real"])]
    s21_y_max = float(np.max(_to_value(ctd.s21_peaks.y)))
    fig.add_scatter(
        x=_to_value(d21_dets["x"], u.MHz),
        y=np.zeros(len(d21_dets)),
        error_y={
            "type": "constant",
            "value": s21_y_max,
            "valueminus": 0,
            "width": 0,
            "color": "cyan",
            "thickness": 0.5,
        },
        **s21_kw,
    )

    mask_baseline = ctd.mask_baseline
    colors = COLOR_PALETTE.cycle_alternated(1, 0.5)
    ds = slice(None, None, 4)
    for ci in range(fs.shape[0]):
        color, faded = next(colors), next(colors)
        fig.add_scattergl(
            x=fs[ci, ds],
            y=as21_db[ci, ds],
            error_y={
                "type": "data",
                "array": as21_unc_db[ci, ds],
                "width": 0,
                "color": faded,
            },
            mode="markers",
            marker={
                "size": 4,
                "color": [color if m else "black" for m in mask_baseline[ci, ds]],
            },
            name=f"S21 {ci}",
            showlegend=False,
            **s21d_kw,
        )

    # Detection markers on panel 3
    det_groups = ctd.det_groups
    f_det = _to_value(det_groups["f"], u.MHz)
    fwhm_det = _to_value(det_groups["fwhm"], u.MHz)
    det_d_max = _to_value(det_groups["d_max"], u.MHz)

    as21_min_idx = np.argmin(as21_db, axis=1, keepdims=True)
    as21_max_idx = np.argmax(as21_db, axis=1, keepdims=True)
    f_min = np.take_along_axis(fs, as21_min_idx, axis=1).ravel()
    f_max = np.take_along_axis(fs, as21_max_idx, axis=1).ravel()
    isort_min = np.argsort(f_min)
    isort_max = np.argsort(f_max)
    as21_det = np.interp(
        f_det,
        f_min[isort_min],
        np.take_along_axis(as21_db, as21_min_idx, axis=1).ravel()[isort_min],
    )
    as21_base = np.interp(
        f_det,
        f_max[isort_max],
        np.take_along_axis(as21_db, as21_max_idx, axis=1).ravel()[isort_max],
    )

    customdata_cols = [
        ("group", ".0f"),
        ("size", ".0f"),
        ("d_min", ".3f"),
        ("d_max", ".3f"),
    ]
    fig.add_scatter(
        x=f_det,
        y=as21_det + 0.1,
        mode="markers",
        marker={"size": 4},
        error_x={
            "type": "data",
            "array": det_d_max * 0.5,
            "width": 0,
            "color": "orange",
        },
        **s21d_kw,
    )
    fig.add_scatter(
        x=f_det,
        y=as21_det,
        mode="markers",
        marker={"size": 4},
        error_x={
            "type": "data",
            "array": fwhm_det * 0.5,
            "width": 0,
            "color": "orange",
        },
        error_y={
            "type": "data",
            "array": (as21_base - as21_det) * 2,
            "arrayminus": np.zeros(f_det.shape),
            "width": 0,
            "color": "orange",
        },
        customdata=np.stack(
            [det_groups[c[0]] for c in customdata_cols if c[0] in det_groups.colnames]
        ).T,
        hovertemplate=(
            "f: %{x:.3f}<br>s21: %{y:.3f}"
            + "".join(
                f"<br>{c[0]}: %{{customdata[{i}]:{c[1]}}}"
                for i, c in enumerate(customdata_cols)
                if c[0] in det_groups.colnames
            )
        ),
        **s21d_kw,
    )

    fig.update_yaxes(title={"text": "|S21| (dB)"}, **s21d_kw)
    fig.update_xaxes(title={"text": "Frequency (MHz)"}, **s21d_kw)
    return fig


# ── Peak properties scatter figure (4-panel) ─────────────────────────────────


def make_peak_props_fig(
    view: KidsFindView,
) -> go.Figure:
    """Plot peak height and signal-to-noise against quality factor.

    Parameters
    ----------
    view : KidsFindView
        Saved finding results with the thresholds used for their selection.

    Returns
    -------
    plotly.graph_objects.Figure
        Four scatter panels with selection thresholds and rejected/dark/edge
        peak overlays.
    """
    ctd = view
    cfg_kf = view.config
    fig = make_subplots(
        2,
        2,
        shared_xaxes="rows",
        fig_layout=FIG_LAYOUT | {"showlegend": True, "height": 1000},
    )
    d21h_kw = {"row": 1, "col": 1}
    d21s_kw = {"row": 1, "col": 2}
    s21h_kw = {"row": 2, "col": 1}
    s21s_kw = {"row": 2, "col": 2}
    lim_kw = {"line": {"dash": "dot", "color": "black"}}

    def _add_props_scatter(  # noqa: PLR0913 -- independent scatter axes and overlays
        name, peaks, *, x_col, y_col, x_unit, y_unit, panel_kw, overlays
    ):
        peak_info = peaks.peaks
        x = _to_value(peak_info[x_col], x_unit)
        y = _to_value(peak_info[y_col], y_unit)

        custom_cols = [
            ("label", ".0f"),
            ("snr", ".3f"),
            ("Qr", ".3f"),
            ("height_db", ".3f"),
            ("halfmax_size", ".0f"),
            ("lookahead", ".0f"),
        ] + [(c, ".0f") for c in peak_info.colnames if c.startswith("sbm")]
        custom_cols = [c for c in custom_cols if c[0] in peak_info.colnames]

        fig.add_scatter(
            x=x,
            y=y,
            mode="markers",
            marker={"color": "green", "size": 6},
            customdata=np.stack([peak_info[c[0]] for c in custom_cols]).T,
            hovertemplate=(
                "f: %{x:.3f}<br>d21: %{y:.3f}"
                + "".join(
                    f"<br>{c[0]}: %{{customdata[{i}]:{c[1]}}}"
                    for i, c in enumerate(custom_cols)
                )
            ),
            name="all peaks",
            showlegend=True,
            **panel_kw,
        )
        for mask_name, mask, mask_color in overlays:
            fig.add_scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                marker={
                    "symbol": "circle-open",
                    "line": {"width": 1, "color": mask_color},
                    "size": 8,
                },
                name=mask_name,
                **panel_kw,
            )
        fig.update_xaxes(title={"text": f"{x_col} ({x_unit})"}, **panel_kw)
        fig.update_yaxes(title={"text": f"{name} ({y_unit})"}, **panel_kw)

    # D21 panels
    for pname, y_col, y_unit, panel_kw, y_lim in [
        (
            "D21 Qr vs Height",
            "height",
            u.Hz**-1,
            d21h_kw,
            cfg_kf.d21_peak_min.to_value(u.Hz**-1),
        ),
        ("D21 Qr vs SNR", "snr", None, d21s_kw, cfg_kf.d21_snr_min),
    ]:
        _add_props_scatter(
            pname,
            ctd.d21_peaks,
            x_col="Qr",
            y_col=y_col,
            x_unit=None,
            y_unit=y_unit,
            panel_kw=panel_kw,
            overlays=[
                (
                    "not real",
                    (ctd.d21_peaks.bitmask & int(SegmentBitMask.not_real)) > 0,
                    "gray",
                ),
                ("dark", (ctd.d21_peaks.bitmask & int(SegmentBitMask.dark)) > 0, "red"),
            ],
        )
        fig.add_hline(y=y_lim, **panel_kw, **lim_kw)
        fig.add_vline(x=cfg_kf.Qr_min, **panel_kw, **lim_kw)
        fig.add_vline(x=cfg_kf.Qr_dark_min, **panel_kw, **lim_kw)
        fig.add_vline(x=cfg_kf.Qr_dark_max, **panel_kw, **lim_kw)

    # S21 panels
    for pname, y_col, panel_kw, y_lim in [
        ("S21 Qr vs Height", "height_db", s21h_kw, cfg_kf.peak_db_min),
        ("S21 Qr vs SNR", "snr", s21s_kw, cfg_kf.snr_min),
    ]:
        _add_props_scatter(
            pname,
            ctd.s21_peaks,
            x_col="Qr",
            y_col=y_col,
            x_unit=None,
            y_unit=None,
            panel_kw=panel_kw,
            overlays=[
                (
                    "not real",
                    (ctd.s21_peaks.bitmask & int(SegmentBitMask.not_real)) > 0,
                    "gray",
                ),
                (
                    "edge",
                    (ctd.s21_peaks.bitmask & int(SegmentBitMask.edge)) > 0,
                    "cyan",
                ),
            ],
        )
        fig.add_hline(y=y_lim, **panel_kw, **lim_kw)
        fig.add_vline(x=cfg_kf.Qr_min, **panel_kw, **lim_kw)
        fig.add_vline(x=cfg_kf.Qr_dark_max, **panel_kw, **lim_kw)

    return fig


# ── Matched-detection figure (3-panel) ───────────────────────────────────────


def make_matched_fig(matched: Match1DView, ref_name: str) -> go.Figure:
    """Plot phase differences, matched tones, and alignment density.

    Parameters
    ----------
    matched : Match1DView
        Matched detections.
    ref_name : str
        Label for the reference detections.

    Returns
    -------
    plotly.graph_objects.Figure
        Matching summary.
    """
    fig = make_subplots(
        3,
        1,
        vertical_spacing=40 / 1200,
        fig_layout=FIG_LAYOUT | {"showlegend": False, "height": 1200},
    )
    dist_kw = {"row": 1, "col": 1}
    match_kw = {"row": 2, "col": 1}
    density_kw = {"row": 3, "col": 1}

    tbl_matched = matched.matched.copy()
    tbl_matched.sort("adist_shifted")
    tbl_matched = unique(tbl_matched, keys="idx_query")

    d_phi_good_max = 5 << u.deg
    d_phi_ok_max = d_phi_good_max * 3
    d_phi = tbl_matched["d_phi"]
    ad_phi = np.abs(d_phi)
    m_good = ad_phi < d_phi_good_max
    m_ok = (ad_phi >= d_phi_good_max) & (ad_phi < d_phi_ok_max)
    m_bad = ad_phi >= d_phi_ok_max
    m_dup = (tbl_matched["bitmask_det"] & int(SegmentBitMask.blended)) > 0

    d_phi_good_max_v = d_phi_good_max.to_value(u.deg)
    bins = (
        np.arange(
            -90 - d_phi_good_max_v / 2,
            90 + d_phi_good_max_v * 1.1 / 2,
            d_phi_good_max_v,
        )
        << u.deg
    )
    x = (0.5 * (bins[1:] + bins[:-1])).to_value(u.deg)

    def _hist(mask):
        y, _ = np.histogram(d_phi[mask], bins=bins)
        return y

    c00, c25, c75, c100 = plotly.colors.sample_colorscale(
        "rdylgn",
        samplepoints=[0, 0.25, 0.75, 1],
    )
    for y, name, color in [
        (_hist(m_bad & ~m_dup), "bad", c75),
        (_hist(m_bad & m_dup), "bad_dup", c25),
        (_hist(m_ok & ~m_dup), "ok", c75),
        (_hist(m_ok & m_dup), "ok_dup", c25),
        (_hist(m_good & ~m_dup), "good", c100),
        (_hist(m_good & m_dup), "good_dup", c00),
    ]:
        fig.add_bar(x=x, y=y, marker={"color": color}, name=name, **dist_kw)

    for x0, x1, opt in [
        (bins[0].to_value(u.deg), -d_phi_ok_max.to_value(u.deg), 0.3),
        (-d_phi_ok_max.to_value(u.deg), -d_phi_good_max_v, 0.15),
        (-d_phi_good_max_v, d_phi_good_max_v, 0.0),
        (d_phi_good_max_v, d_phi_ok_max.to_value(u.deg), 0.15),
        (d_phi_ok_max.to_value(u.deg), bins[-1].to_value(u.deg), 0.3),
    ]:
        fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="black", opacity=opt)

    fig.update_yaxes(title="Count", **dist_kw)
    fig.update_xaxes(title="phi (deg)", **dist_kw)

    _plot_match_pairs(
        matched.matched,
        matched.query_sorted,
        matched.ref_sorted,
        fig=fig,
        panel_kw=match_kw,
        label_value="Frequency (MHz)",
        label_ref=ref_name,
        label_query="Detect",
    )
    _plot_match_density(
        matched.cost,
        matched.index_query,
        matched.index_ref,
        fig=fig,
        panel_kw=density_kw,
        label_ref=f"Ref Id ({ref_name})",
        label_query="Detect Id",
    )
    fig.update_layout(barmode="stack")
    return fig


# ── Top-level entry point ─────────────────────────────────────────────────────


def make_kids_find_figs(
    dt: xr.Dataset | xr.DataTree,
    view: KidsFindView,
) -> dict[str, go.Figure]:
    """Plot KID detection results.

    Parameters
    ----------
    dt : xarray.Dataset or xarray.DataTree
        Sweep data used for detection.
    view : KidsFindView
        Saved detection results, from ``KidsFind.view``.

    Returns
    -------
    dict of str to plotly.graph_objects.Figure
        Figures keyed by ``peaks``, ``peak_props``, ``d21_summary``,
        ``s21_summary``, ``det_summary``, ``matched`` and ``matched_ref``.
        Construction loads the required arrays without running computation.
    """
    figs: dict[str, go.Figure] = {
        "peaks": make_peaks_fig(dt, view),
        "peak_props": make_peak_props_fig(view),
        "d21_summary": make_d21_summary_fig(view),
        "s21_summary": make_s21_summary_fig(view),
        "det_summary": make_det_summary_fig(view),
    }
    ctd = view
    cfg_kf = view.config
    figs["matched"] = make_matched_fig(ctd.matched, "Chan")
    figs["matched_ref"] = make_matched_fig(
        ctd.matched_ref, cfg_kf.match_ref.capitalize()
    )
    return figs
