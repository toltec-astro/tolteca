#!/usr/bin/env python
"""Render a complete component ledger and conditional post-solving performance."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

HERE = Path(__file__).parent
BAND = (5, 20)


def summaries(table):
    """Preserve all populations; distinguish measured noise from signal response."""
    rows = []
    for keys, part in table.groupby(["family", "kind", "population"], dropna=False):
        complete = part[part.diagnostic_status == "complete"]
        physical = complete[complete.population == "active_suitable"]
        rows.append(
            {
                "Model": keys[0],
                "Components": keys[1],
                "Evidence cohort": keys[2],
                "All": len(part),
                "Complete": len(complete),
                "Comparison pass": int(part.readout_suitable.sum()),
                "Response comparison": len(physical),
                "Median ASD(x_opt)/ASD(x_prime)": (
                    complete.coordinate_asd_ratio.median()
                ),
                "Median signal-referred ASD ratio": (
                    physical.signal_referred_asd_ratio.median()
                ),
            }
        )
    return pd.DataFrame(rows)


def matrix(table):
    """Summarize identical modeling across diagnostic nonlinearity bins."""
    rows = []
    bins = [
        ("0-0.01", 0, 0.01),
        ("0.01-0.2", 0.01, 0.2),
        ("0.2-0.75", 0.2, 0.75),
        ("≥0.75", 0.75, np.inf),
    ]
    for family, kind, (label, lo, hi) in itertools.product(
        ("Affine", "Delay"), ("Singlet", "Multiplet"), bins
    ):
        part = table[
            (table.family == family)
            & (table.kind == kind)
            & (table.a_fit >= lo)
            & (table.a_fit < hi)
        ]
        physical = part[part.population == "active_suitable"]
        row = {
            "Model": family,
            "Components": kind,
            "a_ref": label,
            "All": len(part),
            "Complete": int((part.diagnostic_status == "complete").sum()),
            "Comparison pass": int(part.readout_suitable.sum()),
            "Response comparison": len(physical),
        }
        for field, name in [
            ("coordinate_asd_ratio", "Coordinate ASD ratio"),
            ("signal_referred_asd_ratio", "Signal-referred ASD ratio"),
        ]:
            for quantile, suffix in [(0.16, "p16"), (0.5, "median"), (0.84, "p84")]:
                row[f"{name} {suffix}"] = (
                    physical[field].quantile(quantile) if len(physical) else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def figure(table, metric, title, *, physical=False):
    """Plot components without pooling correlated detector noise."""
    result = go.Figure()
    shown = table[table.diagnostic_status == "complete"]
    if physical:
        shown = shown[shown.population == "active_suitable"]
    for (family, kind), part in shown.groupby(["family", "kind"]):
        result.add_trace(
            go.Scattergl(
                x=part.a_fit,
                y=part[metric],
                mode="markers",
                name=f"{family} {kind}",
                marker={
                    "symbol": "circle" if kind == "Singlet" else "diamond",
                    "size": 7,
                    "opacity": 0.65,
                    "color": "#176da4" if family == "Affine" else "#ae571a",
                },
                text=[
                    f"UID {uid} · group {gid}/{rid} · {pop} · comparison pass {ok}"
                    for uid, gid, rid, pop, ok in zip(
                        part.apt_uid,
                        part.group_id,
                        part.resonator_id,
                        part.population,
                        part.readout_suitable,
                        strict=True,
                    )
                ],
                hovertemplate="%{text}<br>a_ref=%{x:.3g}<br>ratio=%{y:.3g}<extra></extra>",
            )
        )
    result.add_hline(y=1, line_dash="dash", line_color="#64748b")
    result.update_layout(
        template="plotly_white",
        height=430,
        font={"size": 15},
        margin={"t": 65, "b": 70},
        title=title,
        legend={"orientation": "h", "y": 1.16},
        xaxis_title="Fitted a_ref (diagnostic only)",
        yaxis_title="5-20 Hz ASD ratio",
        yaxis_type="log",
    )
    return result.to_html(full_html=False, include_plotlyjs=False)


def cross_components(output):
    """Retain frequency-resolved complex spectra and a pairwise coherence ledger."""
    rows = []
    for path in sorted((output / "arrays").glob("*-cross.npz")):
        family, _, gid, _ = path.stem.split("-")
        with np.load(path) as data:
            f = data["frequency"]
            cross = data["cross_real"] + 1j * data["cross_imag"]
            band = (f >= BAND[0]) & (f <= BAND[1])
            ids = data["resonator_ids"]
            for i, j in itertools.combinations(range(len(ids)), 2):
                coherence = abs(cross[:, i, j]) ** 2 / (
                    cross[:, i, i].real * cross[:, j, j].real
                )
                rows.append(
                    {
                        "family": family,
                        "group_id": gid,
                        "component_a": ids[i],
                        "component_b": ids[j],
                        "mean_coherence_5_20_hz": float(np.mean(coherence[band])),
                        "spectra": str(path.relative_to(output)),
                    }
                )
    pd.DataFrame(rows).to_csv(output / "cross-components.csv", index=False)
    return len(rows)


def write_catalog(records, path):
    """Preserve small scientific coefficients; encode missing values as null."""
    values = [
        {key: None if pd.isna(value) else value for key, value in row.items()}
        for row in records.to_dict("records")
    ]
    path.write_text(json.dumps(values, allow_nan=False) + "\n")


def render(table, output):
    """Build the browser report with every component available for inspection."""
    summary, grid = summaries(table), matrix(table)
    summary.to_csv(output / "performance.csv", index=False)
    grid.to_csv(output / "matrix.csv", index=False)
    records = table.sort_values(["family", "group_id", "resonator_id"])
    write_catalog(records, output / "catalog.json")
    tables = {"COVERAGE": summary, "MATRIX": grid}
    replacements = {
        key: value.to_html(index=False, na_rep="—", float_format=lambda v: f"{v:.4g}")
        for key, value in tables.items()
    }
    replacements.update(
        {
            "RAWFIG": figure(
                table, "coordinate_asd_ratio", "All complete trajectories"
            ),
            "PHYSFIG": figure(
                table,
                "signal_referred_asd_ratio",
                "Qualified independent response comparison",
                physical=True,
            ),
            "PAIRS": str(cross_components(output)),
        }
    )
    text = (HERE / "network_postsolve.html").read_text()
    for key, value in replacements.items():
        text = text.replace("{{" + key + "}}", value)
    (output / "index.html").write_text(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(
        pd.read_csv(
            args.output / "components.csv", dtype={"group_id": str, "resonator_id": str}
        ),
        args.output,
    )
