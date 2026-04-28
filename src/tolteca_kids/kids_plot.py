"""Diagnostic plot functions for the KIDs reduction pipeline.

These are pure functions that take v3 pipeline data structures and return
plotly figures, mirroring the v2 ``KidsFindPlot`` step.  They can be called
from scripts or from tolteca_web callbacks.

Main entry point
----------------
``make_kids_find_figs(dt, kids_find_context, cfg_kf)``

Returns a dict of ``go.Figure`` objects:

* ``peaks``           — |D21| / |S21| peaks + raw S21 traces (3-panel)
* ``peak_props``      — Qr vs height/SNR scatter (4-panel)
* ``d21_summary``     — D21 bitmask heatmap + Qr/SNR/FWHM/height rows
* ``s21_summary``     — S21 bitmask heatmap + same rows
* ``det_summary``     — Detection bitmask heatmap + f/Qr rows
"""

from __future__ import annotations

from typing import ClassVar

import astropy.units as u
import numpy as np
import plotly.graph_objects as go
import xarray as xr

from tollan.plot.plotly import SubplotGrid, make_subplots, update_subplot_layout

import plotly.colors
from astropy.table import unique

from .kids_find import KidsFindConfig, KidsFindContext, SegmentBitMask
from .match1d import Match1DResult
from .peaks1d import Peaks1DResult
from .sweep_check import _extract_sweep_arrays

__all__ = ["make_kids_find_figs"]


# ── Default figure layout (matches v2 PlotMixin.fig_layout_default) ─────────

_FIG_LAYOUT = {
    "xaxis": {
        "showline": True,
        "showgrid": False,
        "showticklabels": True,
        "linecolor": "black",
        "linewidth": 1,
        "ticks": "outside",
    },
    "yaxis": {
        "showline": True,
        "showgrid": False,
        "showticklabels": True,
        "linecolor": "black",
        "linewidth": 1,
        "ticks": "outside",
    },
    "plot_bgcolor": "white",
    "autosize": True,
    "margin": {"autoexpand": True, "l": 0, "b": 0, "t": 20},
    "modebar": {"orientation": "v"},
}


# ── Unit-safe helpers ─────────────────────────────────────────────────────────

def _to_value(arr, unit=None):
    """Strip units from arr, converting to ``unit`` if Quantity, else return as-is."""
    if hasattr(arr, "to_value"):
        if unit is not None:
            return arr.to_value(unit)
        return arr.value
    return np.asarray(arr)


# ── Colour helpers ────────────────────────────────────────────────────────────

_COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _color(i: int) -> str:
    return _COLOR_CYCLE[i % len(_COLOR_CYCLE)]


def _alternating_colors(i: int, alpha_hi: float = 1.0, alpha_lo: float = 0.5):
    """Alternate between full and dimmed colours for per-channel S21 traces."""
    base = _color(i // 2)
    return base if i % 2 == 0 else base


# ── Data extraction helpers ───────────────────────────────────────────────────

def _get_raw_arrays(dt: xr.DataTree):
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
    fig = make_subplots(1, 1, fig_layout=_FIG_LAYOUT)
    fig.add_heatmap(z=z, y=names, colorscale="rdylgn_r", zmin=0, zmax=1)
    fig.update_xaxes(title="Segment Id")
    fig.update_layout(title={"text": "Segment Bitmask"})
    return fig


def _make_seg_data_heatmap(name: str, data: np.ndarray, trace_kw: dict) -> go.Figure:
    """Single-row heatmap for one property across segments."""
    fig = make_subplots(1, 1, fig_layout=_FIG_LAYOUT)
    fig.add_heatmap(z=data[np.newaxis, :], y=[name], colorscale="rdylgn_r", **trace_kw)
    fig.update_xaxes(title="Segment Id")
    fig.update_layout(title={"text": name})
    return fig


def _make_summary_fig(bitmask_seg: np.ndarray, seg_data_items: list) -> go.Figure:
    """Composite figure: bitmask row + per-property rows, shared x-axis."""
    grid = SubplotGrid(fig_layout=_FIG_LAYOUT)
    grid.add_subplot(
        row=1, col=1,
        fig=_make_bitmask_heatmap(bitmask_seg),
        row_height=1,
    )
    row0 = grid.shape[0] + 1
    for i, (name, value, trace_kw) in enumerate(seg_data_items):
        grid.add_subplot(
            row=row0 + i, col=1,
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
        row=grid.shape[0], col=1,
    )
    return fig


# ── Peak-level summary figures ────────────────────────────────────────────────

def make_d21_summary_fig(ctx_kf: KidsFindContext, cfg_kf: KidsFindConfig) -> go.Figure:
    """D21 bitmask + Qr/SNR/FWHM/height heatmap rows."""
    ctd = ctx_kf.data
    peak_info = ctd.d21_peaks.peaks
    height = _to_value(peak_info["height"], u.Hz**-1)
    seg_data_items = [
        ("Qr", np.asarray(peak_info["Qr"]), {"zmin": cfg_kf.Qr_min, "zmax": cfg_kf.Qr_dark_min}),
        ("SNR", np.asarray(peak_info["snr"]), {"zmin": 0, "zmax": float(np.quantile(peak_info["snr"], 0.9))}),
        ("FWHM", _to_value(peak_info["width"], u.Hz), {"zmin": 1000, "zmax": 120000}),
        ("height", height, {"zmin": 0, "zmax": float(np.quantile(height, 0.9))}),
    ]
    return _make_summary_fig(ctd.bitmask_d21, seg_data_items)


def make_s21_summary_fig(ctx_kf: KidsFindContext, cfg_kf: KidsFindConfig) -> go.Figure:
    """S21 bitmask + Qr/SNR/FWHM/height heatmap rows."""
    ctd = ctx_kf.data
    peak_info = ctd.s21_peaks.peaks
    height = _to_value(peak_info["height"], u.dimensionless_unscaled)
    seg_data_items = [
        ("Qr", np.asarray(peak_info["Qr"]), {"zmin": cfg_kf.Qr_min, "zmax": cfg_kf.Qr_dark_min}),
        ("SNR", np.asarray(peak_info["snr"]), {"zmin": 0, "zmax": float(np.quantile(peak_info["snr"], 0.9))}),
        ("FWHM", _to_value(peak_info["width"], u.Hz), {"zmin": 1000, "zmax": 120000}),
        ("height", height, {"zmin": 0, "zmax": float(np.quantile(height, 0.9))}),
    ]
    return _make_summary_fig(ctd.bitmask_s21, seg_data_items)


def make_det_summary_fig(ctx_kf: KidsFindContext, cfg_kf: KidsFindConfig) -> go.Figure:
    """Detection bitmask + f/Qr heatmap rows."""
    ctd = ctx_kf.data
    det_groups = ctd.det_groups
    seg_data_items = [
        ("f (MHz)", _to_value(det_groups["f"], u.MHz), {}),
        ("Qr", np.asarray(det_groups["Qr"]), {"zmin": cfg_kf.Qr_min, "zmax": cfg_kf.Qr_dark_min}),
    ]
    return _make_summary_fig(ctd.bitmask_det, seg_data_items)


# ── Peaks overview figure (3-panel) ──────────────────────────────────────────

def make_peaks_fig(
    dt: xr.DataTree,
    ctx_kf: KidsFindContext,
) -> go.Figure:
    """3-panel figure: |D21| peaks / |S21| peaks / raw S21 traces.

    Panels (top→bottom):
    1. |D21| (Hz⁻¹) with peak markers and mask overlays
    2. |S21| (dimensionless) with peak markers and mask overlays
    3. Raw per-channel |S21| (dB) with detection markers
    """
    ctd = ctx_kf.data
    fs, as21_db, as21_unc_db = _get_raw_arrays(dt)

    fig = make_subplots(
        3, 1,
        shared_xaxes="all",
        vertical_spacing=40 / 1000,
        fig_layout=_FIG_LAYOUT | {"showlegend": True, "height": 1000},
    )
    d21_kw = {"row": 1, "col": 1}
    s21_kw = {"row": 2, "col": 1}
    s21d_kw = {"row": 3, "col": 1}

    def _add_peaks_panel(
        name,
        peaks: Peaks1DResult,
        x_unit,
        y_unit,
        panel_kw,
        overlay_masks,
    ):
        peak_info = peaks.peaks
        labels = peaks.labels
        x = _to_value(peaks.x, x_unit)
        y = _to_value(peaks.y, y_unit)
        height = _to_value(peak_info["height"], y_unit)
        base = _to_value(peak_info["base"], y_unit)

        for ii, ll in enumerate(np.unique(labels)):
            m = labels == ll
            fig.add_scatter(
                x=x[m], y=y[m],
                mode="lines",
                line={"color": _color(ii)},
                name=f"peak {ll}",
                showlegend=True,
                **panel_kw,
            )

        customdata_cols = [
            ("label", ".0f"), ("snr", ".3f"), ("Qr", ".3f"),
            ("height_db", ".3f"), ("halfmax_size", ".0f"), ("lookahead", ".0f"),
        ] + [(c, ".0f") for c in peak_info.colnames if c.startswith("sbm")]
        customdata_cols = [c for c in customdata_cols if c[0] in peak_info.colnames]

        fig.add_scatter(
            x=_to_value(peak_info["x"], x_unit),
            y=height / 2 + base,
            error_x={"type": "data", "array": _to_value(peak_info["width"], x_unit) / 2,
                     "width": 0, "color": "green"},
            error_y={"type": "data", "array": height / 2, "width": 0, "color": "green"},
            customdata=np.stack([peak_info[ci[0]] for ci in customdata_cols]).T,
            hovertemplate=(
                "f: %{x:.3f}<br>val: %{y:.3f}"
                + "".join(f"<br>{c[0]}: %{{customdata[{i}]:{c[1]}}}"
                          for i, c in enumerate(customdata_cols))
            ),
            mode="markers",
            marker={"color": "orange", "size": 4},
            name="peak info",
            **panel_kw,
        )

        for mask_name, mask, mask_color in overlay_masks:
            fig.add_scatter(
                x=x[mask], y=y[mask],
                mode="lines",
                line={"color": mask_color},
                name=mask_name,
                **panel_kw,
            )

        fig.update_yaxes(title={"text": f"{name} ({y_unit})"}, **panel_kw)

    _add_peaks_panel(
        "|D21|", ctd.d21_peaks,
        x_unit=u.MHz, y_unit=u.Hz**-1,
        panel_kw=d21_kw,
        overlay_masks=[
            ("not real", ctd.d21_mask_not_real, "gray"),
            ("dark", ctd.d21_mask_dark, "red"),
            ("baseline", ctd.d21_mask_baseline, "black"),
        ],
    )

    _add_peaks_panel(
        "|S21|", ctd.s21_peaks,
        x_unit=u.MHz, y_unit=u.dimensionless_unscaled,
        panel_kw=s21_kw,
        overlay_masks=[
            ("not real", ctd.s21_mask_not_real, "gray"),
            ("edge", ctd.s21_mask_edge.ravel(), "cyan"),
        ],
    )

    # D21-detected vertical lines on S21 panel
    d21_dets = ctd.d21_detected
    s21_y_max = float(np.max(_to_value(ctd.s21_peaks.y)))
    fig.add_scatter(
        x=_to_value(d21_dets["x"], u.MHz),
        y=np.zeros(len(d21_dets)),
        error_y={"type": "constant", "value": s21_y_max, "valueminus": 0,
                 "width": 0, "color": "cyan", "thickness": 0.5},
        **s21_kw,
    )

    # Raw S21 traces (panel 3) — batch all channels into two traces for speed
    # Baseline channels (colored) and non-baseline (black), separated by None
    mask_baseline = ctd.mask_baseline
    ds = slice(None, None, 4)
    x_base: list = []
    y_base: list = []
    x_other: list = []
    y_other: list = []
    for ci in range(fs.shape[0]):
        m = mask_baseline[ci, ds]
        xs = fs[ci, ds].tolist()
        ys = as21_db[ci, ds].tolist()
        if m.any():
            x_base += xs + [None]
            y_base += ys + [None]
        else:
            x_other += xs + [None]
            y_other += ys + [None]
    if x_other:
        fig.add_scattergl(
            x=x_other, y=y_other,
            mode="lines", line={"color": "black", "width": 1},
            showlegend=False, **s21d_kw,
        )
    if x_base:
        fig.add_scattergl(
            x=x_base, y=y_base,
            mode="lines", line={"color": "steelblue", "width": 1},
            name="baseline", showlegend=False, **s21d_kw,
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
        f_det, f_min[isort_min],
        np.take_along_axis(as21_db, as21_min_idx, axis=1).ravel()[isort_min],
    )
    as21_base = np.interp(
        f_det, f_max[isort_max],
        np.take_along_axis(as21_db, as21_max_idx, axis=1).ravel()[isort_max],
    )

    customdata_cols = [("group", ".0f"), ("size", ".0f"), ("d_min", ".3f"), ("d_max", ".3f")]
    fig.add_scatter(
        x=f_det, y=as21_det + 0.1,
        mode="markers",
        marker={"size": 4},
        error_x={"type": "data", "array": det_d_max * 0.5, "width": 0, "color": "orange"},
        **s21d_kw,
    )
    fig.add_scatter(
        x=f_det, y=as21_det,
        mode="markers",
        marker={"size": 4},
        error_x={"type": "data", "array": fwhm_det * 0.5, "width": 0, "color": "orange"},
        error_y={"type": "data",
                 "array": (as21_base - as21_det) * 2,
                 "arrayminus": np.zeros(f_det.shape),
                 "width": 0, "color": "orange"},
        customdata=np.stack(
            [det_groups[c[0]] for c in customdata_cols if c[0] in det_groups.colnames]
        ).T,
        hovertemplate=(
            "f: %{x:.3f}<br>s21: %{y:.3f}"
            + "".join(f"<br>{c[0]}: %{{customdata[{i}]:{c[1]}}}"
                      for i, c in enumerate(customdata_cols)
                      if c[0] in det_groups.colnames)
        ),
        **s21d_kw,
    )

    fig.update_yaxes(title={"text": "|S21| (dB)"}, **s21d_kw)
    fig.update_xaxes(title={"text": "Frequency (MHz)"}, **s21d_kw)
    return fig


# ── Peak properties scatter figure (4-panel) ─────────────────────────────────

def make_peak_props_fig(
    ctx_kf: KidsFindContext,
    cfg_kf: KidsFindConfig,
) -> go.Figure:
    """4-panel scatter: D21 Qr vs height/SNR + S21 Qr vs height_db/SNR."""
    ctd = ctx_kf.data
    fig = make_subplots(
        2, 2,
        shared_xaxes="rows",
        fig_layout=_FIG_LAYOUT | {"showlegend": True, "height": 1000},
    )
    d21h_kw = {"row": 1, "col": 1}
    d21s_kw = {"row": 1, "col": 2}
    s21h_kw = {"row": 2, "col": 1}
    s21s_kw = {"row": 2, "col": 2}
    lim_kw = {"line": {"dash": "dot", "color": "black"}}

    def _add_props_scatter(name, peaks, x_col, y_col, x_unit, y_unit, panel_kw, overlays):
        peak_info = peaks.peaks
        x = _to_value(peak_info[x_col], x_unit)
        y = _to_value(peak_info[y_col], y_unit)

        custom_cols = [
            ("label", ".0f"), ("snr", ".3f"), ("Qr", ".3f"),
            ("height_db", ".3f"), ("halfmax_size", ".0f"), ("lookahead", ".0f"),
        ] + [(c, ".0f") for c in peak_info.colnames if c.startswith("sbm")]
        custom_cols = [c for c in custom_cols if c[0] in peak_info.colnames]

        fig.add_scatter(
            x=x, y=y,
            mode="markers",
            marker={"color": "green", "size": 6},
            customdata=np.stack([peak_info[c[0]] for c in custom_cols]).T,
            hovertemplate=(
                "x: %{x:.3f}<br>y: %{y:.3f}"
                + "".join(f"<br>{c[0]}: %{{customdata[{i}]:{c[1]}}}"
                          for i, c in enumerate(custom_cols))
            ),
            name="all peaks",
            showlegend=True,
            **panel_kw,
        )
        for mask_name, mask, mask_color in overlays:
            fig.add_scatter(
                x=x[mask], y=y[mask],
                mode="markers",
                marker={"symbol": "circle-open",
                        "line": {"width": 1, "color": mask_color}, "size": 8},
                name=mask_name,
                **panel_kw,
            )
        fig.update_xaxes(title={"text": f"{x_col} ({x_unit})"}, **panel_kw)
        fig.update_yaxes(title={"text": f"{name} ({y_unit})"}, **panel_kw)

    # D21 panels
    for pname, y_col, y_unit, panel_kw, y_lim in [
        ("D21 Qr vs Height", "height", u.Hz**-1, d21h_kw,
         cfg_kf.d21_peak_min.to_value(u.Hz**-1)),
        ("D21 Qr vs SNR", "snr", None, d21s_kw, cfg_kf.d21_snr_min),
    ]:
        _add_props_scatter(
            pname, ctd.d21_peaks, "Qr", y_col, None, y_unit, panel_kw,
            [("not real", (ctd.bitmask_d21 & int(SegmentBitMask.not_real)) > 0, "gray"),
             ("dark", (ctd.bitmask_d21 & int(SegmentBitMask.dark)) > 0, "red")],
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
            pname, ctd.s21_peaks, "Qr", y_col, None, None, panel_kw,
            [("not real", (ctd.bitmask_s21 & int(SegmentBitMask.not_real)) > 0, "gray"),
             ("edge", (ctd.bitmask_s21 & int(SegmentBitMask.edge)) > 0, "cyan")],
        )
        fig.add_hline(y=y_lim, **panel_kw, **lim_kw)
        fig.add_vline(x=cfg_kf.Qr_min, **panel_kw, **lim_kw)
        fig.add_vline(x=cfg_kf.Qr_dark_max, **panel_kw, **lim_kw)

    return fig


# ── Matched-detection figure (3-panel) ───────────────────────────────────────

def make_matched_fig(matched: Match1DResult, ref_name: str) -> go.Figure:
    """3-panel figure: phi distribution / match lines / DTW density.

    Mirrors v2 ``KidsFindPlot.make_matched_fig``.
    """
    fig = make_subplots(
        3, 1,
        vertical_spacing=40 / 1200,
        fig_layout=_FIG_LAYOUT | {"showlegend": False, "height": 1200},
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
        "rdylgn", samplepoints=[0, 0.25, 0.75, 1],
    )
    for y, name, color in [
        (_hist(m_bad & ~m_dup),  "bad",      c75),
        (_hist(m_bad & m_dup),   "bad_dup",  c25),
        (_hist(m_ok & ~m_dup),   "ok",       c75),
        (_hist(m_ok & m_dup),    "ok_dup",   c25),
        (_hist(m_good & ~m_dup), "good",     c100),
        (_hist(m_good & m_dup),  "good_dup", c00),
    ]:
        fig.add_bar(x=x, y=y, marker={"color": color}, name=name, **dist_kw)

    for x0, x1, opt in [
        (bins[0].to_value(u.deg),          -d_phi_ok_max.to_value(u.deg),   0.3),
        (-d_phi_ok_max.to_value(u.deg),    -d_phi_good_max_v,               0.15),
        (-d_phi_good_max_v,                 d_phi_good_max_v,               0.0),
        (d_phi_good_max_v,                  d_phi_ok_max.to_value(u.deg),   0.15),
        (d_phi_ok_max.to_value(u.deg),      bins[-1].to_value(u.deg),       0.3),
    ]:
        fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="black", opacity=opt)

    fig.update_yaxes(title="Count", **dist_kw)
    fig.update_xaxes(title="phi (deg)", **dist_kw)

    matched.make_plotly_fig(
        type="match", fig=fig, panel_kw=match_kw,
        label_value="Frequency (MHz)",
        label_ref=ref_name,
        label_query="Detect",
    )
    matched.make_plotly_fig(
        type="density", fig=fig, panel_kw=density_kw,
        label_ref=f"Ref Id ({ref_name})",
        label_query="Detect Id",
    )
    fig.update_layout(barmode="stack")
    return fig


# ── Top-level entry point ─────────────────────────────────────────────────────

def make_kids_find_figs(
    dt: xr.DataTree,
    ctx_kf: KidsFindContext,
    cfg_kf: KidsFindConfig,
) -> dict[str, go.Figure]:
    """Generate all diagnostic figures for a KidsFind result.

    Parameters
    ----------
    dt : xr.DataTree
        Raw sweep DataTree (from ``_build_datatree_from_zarr``).
    ctx_kf : KidsFindContext
        Populated KidsFindContext (from ``KidsFind.run()``).
    cfg_kf : KidsFindConfig
        The KidsFindConfig used during the run.

    Returns
    -------
    dict mapping figure name → ``go.Figure``.
    """
    figs: dict[str, go.Figure] = {
        "peaks": make_peaks_fig(dt, ctx_kf),
        "peak_props": make_peak_props_fig(ctx_kf, cfg_kf),
        "d21_summary": make_d21_summary_fig(ctx_kf, cfg_kf),
        "s21_summary": make_s21_summary_fig(ctx_kf, cfg_kf),
        "det_summary": make_det_summary_fig(ctx_kf, cfg_kf),
    }
    ctd = ctx_kf.data
    if ctd.matched is not ...:
        figs["matched"] = make_matched_fig(ctd.matched, "Chan")
    if ctd.matched_ref is not ...:
        figs["matched_ref"] = make_matched_fig(
            ctd.matched_ref, cfg_kf.match_ref.capitalize()
        )
    return figs
