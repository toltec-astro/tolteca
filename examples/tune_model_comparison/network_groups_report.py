#!/usr/bin/env python
"""Render group-aware identity, response and readout evidence."""

from __future__ import annotations

import itertools
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).parent
METHODS = runpy.run_path(str(HERE / "network_response.py"))
BAND_HZ = (5, 20)
FAMILIES = [("Affine", "coverage"), ("Delay", "coverage-delay")]


def matrix(table):
    """Count every component; noise summaries use qualified response evidence."""
    result = []
    bins = [
        ("0-0.01", 0, 0.01),
        ("0.01-0.2", 0.01, 0.2),
        ("0.2-0.75", 0.2, 0.75),
        ("≥0.75", 0.75, np.inf),
    ]
    for family, kind in itertools.product(
        ("Affine", "Delay"), ("Singlet", "Multiplet")
    ):
        for label, lo, hi in bins:
            part = table[
                (table.family == family)
                & (table.kind == kind)
                & (table.a_fit >= lo)
                & (table.a_fit < hi)
            ]
            good = part[part.population == "active_suitable"]
            result.append(
                {
                    "Model": family,
                    "Components": kind,
                    "a_ref": label,
                    "All": len(part),
                    "Comparison pass": int(part.readout_suitable.sum()),
                    "Response comparison": len(good),
                    "Median frequency ASD (Hz/√Hz)": good.frequency_asd.median(),
                    "Median loss/detuning slope, Skydip": (
                        good.loss_detuning_slope_at_probe.median()
                    ),
                    "Median slope, 1 s": good.loss_detuning_slope_1s.median(),
                    "Median slope, 8 s": good.loss_detuning_slope_8s.median(),
                    "Median held-out projected/x ASD": good.projected_to_x_asd.median(),
                }
            )
    return pd.DataFrame(result)


def inspector(root, output, table, laws):
    """Plot full joint solutions without treating blended IQ as an isolated circle."""
    spectra = np.load(output / "spectra.npz")
    trajectories = np.load(output / "trajectories.npz")
    cases, pairs = [], []
    for family, name in FAMILIES:
        folder = root / name / "measured"
        rate = json.loads((folder / "output-quality-method.json").read_text())[
            "sample_rate_hz"
        ]
        ledger = table[table.family == family].set_index(["group_id", "resonator_id"])
        groups = {}
        for base, x, r in METHODS["component_records"](folder):
            row = ledger.loc[base["group_id"], base["resonator_id"]]
            if x is None:
                continue
            if row.multiplicity > 1:
                groups.setdefault(base["group_id"], []).append(
                    (base["resonator_id"], x, r)
                )
            if row.population != "active_suitable":
                continue
            key = f"{family}-{base['group_id']}-{base['resonator_id']}"
            source = laws[int(row.apt_uid)]
            weights = np.array([row.weight_x, row.weight_r])
            fold_slopes = (
                np.array([f["slope"] for f in source["folds"]])
                * source["fref"]
                / row.f_probe_hz
            )
            transfer = weights[0] + weights[1] * fold_slopes
            cases.append(
                {
                    "key": key,
                    "family": family,
                    "uid": int(row.apt_uid),
                    "group": base["group_id"],
                    "component": base["resonator_id"],
                    "multiplicity": int(row.multiplicity),
                    "native_route": row.native_route,
                    "a_ref": row.a_fit,
                    "slope": row.loss_detuning_slope_at_probe,
                    "slope_1s": row.loss_detuning_slope_1s,
                    "source_slope": source["fit"]["slope"],
                    "source_chi2": source["fit"]["chi2"] / source["fit"]["dof"],
                    "source_x": source["x_opt"],
                    "source_r": source["r"],
                    "source_x_error": np.sqrt(
                        np.array(source["covariance_x_r"])[:, 0, 0]
                    ).tolist(),
                    "source_r_error": np.sqrt(
                        np.array(source["covariance_x_r"])[:, 1, 1]
                    ).tolist(),
                    "source_scans": [p["scan"] for p in source["points"]],
                    "source_group": source["group"],
                    "source_component": source["component"],
                    "trajectory": trajectories[key].tolist(),
                    "time": (np.arange(0, len(x), 16) / rate).tolist(),
                    "x": x[::16].tolist(),
                    "r": r[::16].tolist(),
                    "frequency": spectra["frequency"].tolist(),
                    "psd": spectra[key][:2].tolist(),
                    "projected_ratio": row.projected_to_x_asd,
                    "fold_transfer": [float(transfer.min()), float(transfer.max())],
                }
            )
        for gid, components in groups.items():
            for a, b in itertools.combinations(components, 2):
                frequency, cov = METHODS["spectral_matrix"](a[1], b[1], rate)
                band = (frequency >= BAND_HZ[0]) & (frequency <= BAND_HZ[1])
                avg = cov[band].mean(axis=0)
                pairs.append(
                    {
                        "family": family,
                        "group_id": gid,
                        "component_a": a[0],
                        "component_b": b[0],
                        "x_coherence": float(
                            abs(avg[0, 1]) ** 2 / (avg[0, 0].real * avg[1, 1].real)
                        ),
                    }
                )
    pd.DataFrame(pairs).to_csv(output / "joint-covariance.csv", index=False)
    (output / "cases.json").write_text(json.dumps(cases, allow_nan=False))
    return cases


def diagnostic_html(frame, labels):
    """Use readable labels in HTML while preserving machine codes in CSV."""
    return (
        frame.rename(columns=labels)
        .replace({"_": " ", ";": "; "}, regex=True)
        .to_html(index=False, na_rep="—", float_format=lambda v: f"{v:.4g}")
    )


def render(root, output):
    """Build tables and interactive figures from the saved comparison ledger."""
    table = pd.read_csv(
        output / "signal-noise.csv", dtype={"group_id": str, "resonator_id": str}
    )
    response = json.loads((output / "skydip-response.json").read_text())
    laws = {r["uid"]: r for r in response if "fit" in r}
    counts = pd.read_csv(output / "coverage.csv")
    cross = pd.read_csv(output / "evidence-readout.csv")
    noise = (
        table.groupby(
            ["family", "kind", "evidence_state", "readout_state"], dropna=False
        )
        .agg(
            components=("resonator_id", "size"),
            complete_records=("frequency_asd", "count"),
            median_frequency_asd=("frequency_asd", "median"),
            median_loss_asd=("r_asd", "median"),
        )
        .reset_index()
    )
    noise.to_csv(output / "noise-populations.csv", index=False)
    summary = matrix(table)
    summary.to_csv(output / "matrix.csv", index=False)
    cases = inspector(root, output, table, laws)
    source_rows = [
        {
            "uid": r["uid"],
            "group": r["group"],
            "component": r["component"],
            "response_fit": "fit" in r,
            "gaps": ";".join(r["gaps"]),
            "quality": ";".join(r["quality"]),
            "reduced_chi2": r["fit"]["chi2"] / r["fit"]["dof"]
            if "fit" in r
            else np.nan,
        }
        for r in response
    ]
    pd.DataFrame(source_rows).to_csv(output / "source-response-ledger.csv", index=False)
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Independent Skydip direction and measured trajectory",
            "Measured frequency fluctuations",
        ),
    )
    for (family, kind), part in table[table.population == "active_suitable"].groupby(
        ["family", "kind"]
    ):
        color = "#1971c2" if family == "Affine" else "#c05621"
        symbol = "circle" if kind == "Singlet" else "diamond"
        label = f"{family} · {kind} · n={len(part)}"
        custom = part[["apt_uid", "group_id", "resonator_id", "a_fit"]]
        figure.add_trace(
            go.Scatter(
                x=part.loss_detuning_slope_at_probe,
                y=part.loss_detuning_slope_1s,
                mode="markers",
                name=label,
                customdata=custom,
                marker={"color": color, "symbol": symbol, "size": 7},
                hovertemplate=(
                    "UID %{customdata[0]} · group %{customdata[1]} / "
                    "%{customdata[2]}<br>a=%{customdata[3]:.3g}"
                    "<br>Skydip slope=%{x:.4g}<br>1-s slope=%{y:.4g}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=part.a_fit,
                y=part.frequency_asd,
                mode="markers",
                name=label,
                showlegend=False,
                customdata=custom,
                marker={"color": color, "symbol": symbol, "size": 7},
                hovertemplate=(
                    "UID %{customdata[0]} · group %{customdata[1]} / "
                    "%{customdata[2]}<br>a=%{x:.3g}<br>"
                    "ASD=%{y:.3g} Hz/√Hz<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    figure.update_xaxes(title_text="Skydip Rᵣ/Rₓ", row=1, col=1)
    figure.update_yaxes(title_text="145609 Δr/Δx_opt (1 s)", row=1, col=1)
    figure.update_xaxes(title_text="Fitted a_ref (diagnostic only)", row=1, col=2)
    figure.update_yaxes(
        title_text="Frequency ASD (Hz/√Hz), 5-20 Hz", type="log", row=1, col=2
    )
    figure.update_layout(
        template="plotly_white",
        height=480,
        font={"size": 15},
        margin={"t": 90},
        legend={"orientation": "h", "y": 1.2},
    )
    old = pd.read_csv(
        root / "active-response/components.csv", dtype={"resonator_id": str}
    )
    paired = table.merge(
        old[["family", "resonator_id", "apt_uid", "population"]],
        on=["family", "resonator_id"],
        suffixes=("", "_old"),
    )
    same = paired[(paired.association == "supported") & paired.apt_uid_old.notna()]
    if (same.apt_uid != same.apt_uid_old).any():
        msg = "A previously supported identity changed: inspect before reporting"
        raise ValueError(msg)
    paired[
        [
            "family",
            "resonator_id",
            "apt_uid_old",
            "apt_uid",
            "population_old",
            "population",
        ]
    ].to_csv(output / "comparison-changes.csv", index=False)
    transitions = pd.crosstab(
        paired[paired.family == "Affine"].population_old,
        paired[paired.family == "Affine"].population,
    )
    supported = table[(table.family == "Affine") & (table.association == "supported")]
    eligible = table[table.population == "active_suitable"].groupby("family").size()
    headline = (
        f"{len(supported)} provisional identities are supported in both Tune blocks. "
        f"{eligible['Affine']} affine and "
        f"{eligible['Delay']} delay components have "
        "both an independent loading-response measurement "
        "and passing comparison checks."
    )
    diagnostic_columns = [
        "family",
        "kind",
        "route_status",
        "fit_adequacy",
        "numerical_validity",
        "parity_status",
        "selection_policy_limits",
    ]
    diagnostic_counts = (
        table.fillna({"selection_policy_limits": "none"})
        .groupby(diagnostic_columns)
        .size()
        .reset_index(name="components")
    )
    diagnostic_counts.to_csv(output / "diagnostic-counts.csv", index=False)
    review = table[
        (table.multiplicity > 1)
        & table.population.isin(["active_suitable", "active_poor"])
    ].copy()
    review["residual_percent"] = 100 * review.residual_to_sweep_variation
    review = review[
        [
            "family",
            "apt_uid",
            "group_id",
            "resonator_id",
            "residual_percent",
            "native_fraction",
            "route_status",
            "fit_adequacy",
            "numerical_validity",
            "numerical_flags",
            "parity_status",
            "selection_policy_limits",
            "readout_suitable",
        ]
    ]
    review["apt_uid"] = review.apt_uid.astype(int)
    review = review.sort_values(["family", "apt_uid"])
    review.to_csv(output / "response-multiplets.csv", index=False)
    labels = {
        "family": "Model",
        "kind": "Component type",
        "apt_uid": "UID",
        "group_id": "Group",
        "resonator_id": "Component",
        "residual_percent": "Residual (%)",
        "native_fraction": "Valid fraction",
        "route_status": "Route",
        "fit_adequacy": "Fit adequacy",
        "numerical_validity": "Numerical validity",
        "numerical_flags": "Numerical flags",
        "parity_status": "Python/native",
        "selection_policy_limits": "Policy limits",
        "readout_suitable": "Comparison pass",
        "components": "Count",
    }
    replacements = {
        "HEADLINE": headline,
        "ROOT": str(root.relative_to(Path.cwd()))
        if root.is_relative_to(Path.cwd())
        else str(root),
        "SOURCE_PREFIX": "../source"
        if (root / "replay-source.json").exists()
        else "..",
        "DIAGNOSTICS": diagnostic_html(diagnostic_counts, labels),
        "MULTIPLETS": diagnostic_html(review, labels),
        "COVERAGE": counts.to_html(index=False),
        "CROSS": cross.to_html(index=False),
        "MATRIX": summary.to_html(
            index=False, na_rep="—", float_format=lambda x: f"{x:.4g}"
        ),
        "FIGURE": figure.to_html(full_html=False, include_plotlyjs=False),
        "TRANSITIONS": transitions.to_html(),
        "CASECOUNT": str(len(cases)),
        "LAWCOUNT": str(len(laws)),
        "NETWORKNOISE": noise[noise.readout_state == "pass"].to_html(
            index=False, na_rep="—", float_format=lambda x: f"{x:.4g}"
        ),
    }
    html = (HERE / "network_groups.html").read_text()
    for key, value in replacements.items():
        html = html.replace("{{" + key + "}}", value)
    (output / "index.html").write_text(html)
