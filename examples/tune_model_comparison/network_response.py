#!/usr/bin/env python
"""Compare solved loss/detuning trajectories with an independent Skydip cohort.

The empirical slopes include sky, intrinsic fluctuations and readout/model error.
They are not optical responsivities. Conditional projection uses the independently
measured Skydip direction, learns weights on the first half, and assesses the
second half without changing calibration, flags or detector selection.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.table import Table
from plotly.subplots import make_subplots
from scipy.signal import csd

from tolteca_kidsproc.analysis import PSDAnalysis

SUMMARY_BAND_HZ = (5, 20)
FAILURE_MASK = 65535 ^ 32
BINS = [
    ("0-0.01", 0, 0.01),
    ("0.01-0.2", 0.01, 0.2),
    ("0.2-0.75", 0.2, 0.75),
    (">0.75", 0.75, np.inf),
]
COLORS = ["#495057", "#228be6", "#e67700", "#ae3ec9"]


def spectral_matrix(x, r, rate):
    """One-sided Welch spectral matrix ordered (x_opt, r), in 1/Hz."""
    options = {
        "window": "hann",
        "nperseg": 1024,
        "noverlap": 512,
        "detrend": "constant",
        "scaling": "density",
        "return_onesided": True,
    }
    frequency, power = PSDAnalysis.compute_psd(
        np.column_stack([x, r]), f_smp=rate, axis=0, **options
    )
    _, cross = csd(x, r, fs=rate, **options)
    matrix = np.zeros((len(frequency), 2, 2), complex)
    matrix[:, 0, 0], matrix[:, 1, 1] = power.T
    matrix[:, 0, 1], matrix[:, 1, 0] = cross, cross.conj()
    return frequency, matrix


def binned_trajectory(x, r, rate, seconds=1):
    """Average complete consecutive windows; do not shuffle or fill gaps."""
    width = round(rate * seconds)
    count = len(x) // width
    return np.array(
        [v[: count * width].reshape(count, width).mean(axis=1) for v in (x, r)]
    )


def empirical_slope(x, r):
    """Return centered OLS slope and correlation, without an optical attribution."""
    x, r = x - np.mean(x), r - np.mean(r)
    xx, rr, xr = np.dot(x, x), np.dot(r, r), np.dot(x, r)
    return (
        float(xr / xx) if xx > 0 else np.nan,
        float(xr / np.sqrt(xx * rr)) if xx > 0 and rr > 0 else np.nan,
    )


def projection_weights(covariance, beta):
    """Minimum band variance with unit response to (delta x, beta*delta x).

    Covariance describes measured uncleaned fluctuations, not isolated readout
    noise. No singular regularization or assumed covariance is substituted.
    """
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance)[0] <= 0:
        msg = "Measured two-coordinate covariance is not positive definite"
        raise ValueError(msg)
    signal = np.array([1.0, beta])
    weights = np.linalg.solve(covariance, signal)
    return weights / np.dot(signal, weights)


def assess(x, r, rate, beta, beta_limits):
    """Measure trajectory and held-out noise; keep optical direction conditional."""
    xb, rb = binned_trajectory(x, r, rate)
    slope, corr = empirical_slope(xb, rb)
    raw_slope, raw_corr = empirical_slope(x, r)
    split = len(x) // 2
    frequency, train = spectral_matrix(x[:split], r[:split], rate)
    _, test = spectral_matrix(x[split:], r[split:], rate)
    band = (frequency >= SUMMARY_BAND_HZ[0]) & (frequency <= SUMMARY_BAND_HZ[1])
    training = np.mean(train[band].real, axis=0)
    heldout = np.mean(test[band].real, axis=0)
    weights = projection_weights(training, beta)
    variance = float(weights @ heldout @ weights)
    x_asd = float(np.sqrt(heldout[0, 0]))
    projected_asd = float(np.sqrt(variance))
    limits = weights[0] + weights[1] * np.asarray(beta_limits)
    residual = np.array([-beta, 1.0])
    result = {
        "beta_1s": slope,
        "correlation_1s": corr,
        "beta_samples": raw_slope,
        "correlation_samples": raw_corr,
        "x_asd_5_20_hz": x_asd,
        "r_asd_5_20_hz": float(np.sqrt(heldout[1, 1])),
        "projected_asd_5_20_hz": projected_asd,
        "projected_to_x_asd": projected_asd / x_asd,
        "weight_x": float(weights[0]),
        "weight_r": float(weights[1]),
        "assumed_signal_transfer": float(weights @ [1.0, beta]),
        "transfer_beta16": float(limits[0]),
        "transfer_beta84": float(limits[1]),
        "residual_asd_5_20_hz": float(np.sqrt(residual @ heldout @ residual)),
        "noise_band_correlation": float(
            heldout[0, 1] / np.sqrt(heldout[0, 0] * heldout[1, 1])
        ),
        "status": "measured_conditional_projection",
    }
    for seconds in (0.5, 2):
        means = binned_trajectory(x, r, rate, seconds)
        result[f"beta_{seconds:g}s"] = empirical_slope(*means)[0]
    return result, (xb, rb)


def component_records(folder):
    """Yield one deterministic estimate per component with complete native output.

    For a singlet, choose the complete probe nearest its fitted resonance. Joint
    components use their full-group solved frequency/loss with the nearest probe
    defining the detuning coordinate. Selection never uses a measured noise level.
    """
    request = json.loads((folder / "request.json").read_text())
    report = Table.read(request["ecsv"], format="ascii.ecsv")
    coverage = pd.read_csv(
        folder / "component-coverage.csv", dtype={"group_id": str, "resonator_id": str}
    )
    channels = pd.read_csv(
        folder.parent / "block1/channels.csv", dtype={"group_id": str}
    )
    shape = request["samples"], request["channels"]
    native = folder / "native/stub"
    loss, x = [
        np.fromfile(native / f"{key}.bin", dtype="float64").reshape(shape)
        for key in ("r", "x")
    ]
    flags = np.fromfile(native / "flags.bin", dtype="uint16").reshape(shape)
    complete = np.all((flags & FAILURE_MASK) == 0, axis=0)
    lookup = {int(row["chan"]): row for row in report}
    groups = {str(g["group_id"]): g for g in report.meta["joint_models"]["groups"]}
    saved = {}
    for row in coverage.itertuples():
        base = {
            "group_id": row.group_id,
            "resonator_id": row.resonator_id,
            "multiplicity": row.multiplicity,
            "a_fit": row.a_nl,
            "fit_status": row.fit_status,
            "fit_at_bound": row.at_bound,
            "status": "no_complete_valid_estimate",
        }
        gid = str(row.group_id)
        if row.native_route == "joint":
            group = groups[gid]
            if gid not in saved:
                path = native / f"group-{gid}"
                gf = np.fromfile(path / "flags.bin", dtype="uint16")
                saved[gid] = None
                if np.all((gf & FAILURE_MASK) == 0):
                    saved[gid] = [
                        np.fromfile(path / f"{k}.bin", dtype="float64").reshape(
                            len(gf), len(group["resonator_ids"])
                        )
                        for k in ("fr", "r")
                    ]
            arrays = saved[gid]
            if arrays is not None:
                j = list(map(str, group["resonator_ids"])).index(str(row.resonator_id))
                fr, r = (v[:, j] for v in arrays)
                p = int(np.argmin(np.abs(np.asarray(group["probes"]) - row.fr)))
                fp = group["probes"][p]
                base.update(
                    probe=group["channels"][p],
                    f_probe_hz=fp,
                    r_reference=row.r,
                    native_route="joint",
                )
                yield base, fp / fr - 1, r
                continue
        elif row.native_route == "independent":
            options = [
                int(c)
                for c in channels.loc[channels.group_id == gid, "chan"]
                if complete[c]
            ]
            if options:
                chan = min(options, key=lambda c: (abs(lookup[c]["f_in"] - row.fr), c))
                base.update(
                    probe=chan,
                    f_probe_hz=float(lookup[chan]["f_in"]),
                    r_reference=row.r,
                    native_route="independent",
                )
                yield base, x[:, chan], loss[:, chan]
                continue
        yield base, None, None


def response_panels(root):
    """Save component/matrix evidence and return data-driven comparison figures."""
    skydip = json.loads((root / "references/skydip-network.json").read_text())
    reference = skydip["summary"]
    beta = reference["beta_median"]
    limits = [reference["beta_percentiles"][str(p)] for p in (16, 84)]
    rows, series = [], {}
    for family, name in [("Affine", "coverage"), ("Delay", "coverage-delay")]:
        folder = root / name / "measured"
        rate = json.loads((folder / "output-quality-method.json").read_text())[
            "sample_rate_hz"
        ]
        for base, x, r in component_records(folder):
            base["family"] = family
            if x is not None:
                try:
                    result, binned = assess(x, r, rate, beta, limits)
                except ValueError as exc:
                    base.update(status=str(exc))
                else:
                    base.update(result)
                    key = f"{family}-{base['group_id']}-{base['resonator_id']}"
                    series[key] = binned
            rows.append(base)
    table = pd.DataFrame(rows)
    table.to_csv(root / "response-components.csv", index=False)
    np.savez_compressed(root / "response-trajectories.npz", **series)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Affine: singlets",
            "Affine: multiplets",
            "Delay: singlets",
            "Delay: multiplets",
        ],
    )
    matrix = []
    for plot_row, family in enumerate(["Affine", "Delay"], 1):
        for plot_col, kind in enumerate(["Singlet", "Multiplet"], 1):
            member = (
                table.multiplicity == 1 if kind == "Singlet" else table.multiplicity > 1
            )
            for (label, lo, hi), color in zip(BINS, COLORS, strict=True):
                part = table.loc[
                    (table.family == family)
                    & member
                    & (table.a_fit >= lo if lo == 0 else table.a_fit > lo)
                    & (table.a_fit <= hi)
                ]
                good = part.loc[part.status == "measured_conditional_projection"]
                matrix.append(
                    {
                        "Gain": family,
                        "Group type": kind,
                        "Fitted a bin": label,
                        "Fitted": len(part),
                        "Complete": len(good),
                        "Median 1 s slope": good.beta_1s.median(),
                        "Median 1 s correlation": good.correlation_1s.median(),
                        "x ASD [1/sqrt(Hz)]": good.x_asd_5_20_hz.median(),
                        "Projected/x ASD": good.projected_to_x_asd.median(),
                        "Median gain at beta16": good.transfer_beta16.median(),
                        "Median gain at beta84": good.transfer_beta84.median(),
                    }
                )
                fig.add_trace(
                    go.Box(
                        y=good.beta_1s,
                        name=label,
                        marker_color=color,
                        boxpoints="all",
                        jitter=0.4,
                        pointpos=0,
                        legendgroup=label,
                        showlegend=plot_row == 1 and plot_col == 1,
                        customdata=np.c_[good.group_id, good.resonator_id],
                        hovertemplate=(
                            "Group %{customdata[0]}, component %{customdata[1]}"
                            "<br>1 s trajectory slope=%{y:.4g}<extra></extra>"
                        ),
                    ),
                    row=plot_row,
                    col=plot_col,
                )
            fig.add_hrect(
                y0=limits[0],
                y1=limits[1],
                line_width=0,
                fillcolor="#1971c2",
                opacity=0.12,
                row=plot_row,
                col=plot_col,
            )
            fig.add_hline(
                y=beta,
                line_dash="dash",
                line_color="#1971c2",
                row=plot_row,
                col=plot_col,
            )
    fig.update_yaxes(title_text="Δr / Δx_opt trajectory slope")
    fig.update_layout(
        title="145609 trajectory slopes; shaded Skydip detector spread",
        height=850,
        legend={"orientation": "h", "y": -0.16},
    )
    matrix = pd.DataFrame(matrix)
    matrix.to_csv(root / "response-matrix.csv", index=False)
    assumptions = {
        "skydip_beta": beta,
        "skydip_beta16_beta84": limits,
        "trajectory": (
            "OLS of consecutive 122-sample means (~1 s); "
            "mixed fluctuations, not optical calibration"
        ),
        "alternate_windows_s": [0.5, 2],
        "projection": "w^T (1,beta)=1; weights from first half, PSD from second half",
        "covariance_band_hz": [5, 20],
        "covariance": "uncleaned measured spectral matrix; not isolated readout noise",
        "gain_range": (
            "beta16/beta84 are detector spread in another observation, "
            "not an uncertainty interval for 145609"
        ),
        "multiplicity": (
            "Joint r and frequency use all fitted probes; "
            "no singlet approximation for multiplets"
        ),
        "exclusions": (
            "Incomplete estimates remain rows; no interpolation; "
            "no science acceptance cut"
        ),
        "not_measured": [
            "absolute optical responsivity",
            "source S/N",
            "mapping speed",
            "b versus optical loading in 145609",
        ],
        "identity": (
            "No cross-observation detector match; "
            "no network 11/12 data in this network-0 example"
        ),
    }
    (root / "response-method.json").write_text(json.dumps(assumptions, indent=2) + "\n")
    return matrix.to_html(
        index=False, border=0, na_rep="--", float_format=lambda v: f"{v:.3g}"
    ), fig


def skydip_figure(root):
    """Plot independent sweep-fit measurements, never a derived a(t) trace."""
    data = json.loads((root / "references/skydip-network.json").read_text())
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Loss and detuning: 112 traces",
            "Nonlinearity versus Qr",
        ],
    )
    for item in data["contributors"]:
        points = item["points"]
        fr = np.array([p["fr"] for p in points])
        r = np.array([p["r"] for p in points])
        x = item["fref"] / fr - 1
        fig.add_trace(
            go.Scatter(
                x=x - x[0],
                y=r - r[0],
                mode="lines+markers",
                name=f"UID {item['uids']}",
                line={"width": 1, "color": "#228be6"},
                opacity=0.35,
                showlegend=False,
                customdata=[p["scan"] for p in points],
                hovertemplate=(
                    f"{item['key']}<br>Scan %{{customdata}}"
                    "<br>Δx=%{x:.4g}<br>Δr=%{y:.4g}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        a = np.array([p["a_nl"] for p in points])
        qr = np.array([p["Qr"] for p in points])
        # No ratio of fitted a: a near zero must not create a divergent proxy.
        fig.add_trace(
            go.Scatter(
                x=qr,
                y=a,
                mode="lines+markers",
                showlegend=False,
                name=f"UID {item['uids']}",
                line={"width": 1, "color": "#e67700"},
                opacity=0.35,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(
        title_text="Δx_opt (dimensionless)", tickformat=".1e", row=1, col=1
    )
    fig.update_yaxes(title_text="Δr (dimensionless)", tickformat=".1e", row=1, col=1)
    fig.update_xaxes(title_text="Fitted Qr", row=1, col=2)
    fig.update_yaxes(title_text="Fitted a_nl", row=1, col=2)
    fig.update_layout(
        height=520, title="Independent Skydip evidence: each point is a sweep fit"
    )
    return fig
