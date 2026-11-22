#!/usr/bin/env python
"""Audit response evidence without treating numerical coverage as science yield.

This recipe consumes saved fits and the explicit dark-candidate APT study. It
does not classify dark detectors from Qr or transfer model-local resonator IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import zarr


def verify_measurements(current, reference):
    """Require identical acquired sweeps and finding evidence before a join."""
    hashes = {}
    fields = {
        "sweep": ("I", "Q", "f_chan", "f_lo"),
        "kids_find/d21/peaks": ("idx", "x", "Qr", "height", "bitmask"),
        "kids_find/det_grouped": ("group", "peak_id", "subdet", "x"),
    }
    for group, names in fields.items():
        for name in names:
            key = f"{group}/{name}"
            left, right = np.asarray(current[key]), np.asarray(reference[key])
            equal_nan = left.dtype.kind in "fc"
            if left.dtype != right.dtype or not np.array_equal(
                left, right, equal_nan=equal_nan
            ):
                msg = f"Different acquisition/finding evidence: {key}"
                raise ValueError(msg)
            hashes[key] = hashlib.sha256(left.tobytes()).hexdigest()
    return hashes


def candidate_links(rows, links):
    """Resolve original peaks to current components, retaining tentative status.

    The caller must first verify the exact acquisition and peak measurements.
    A multi-peak reference is admitted only when all peaks resolve to one current
    component. Neither a matching model-local integer nor frequency is a join.
    """
    by_peak = links.set_index("peak_id")
    if not by_peak.index.is_unique:
        msg = "Duplicate original peak identities"
        raise ValueError(msg)
    result = {}
    for row in rows:
        if row["detector"] is None:
            continue
        peaks = [int(p) for p in row["peaks"].split(",") if p.strip()]
        targets = set()
        for peak in peaks:
            if peak not in by_peak.index or by_peak.loc[peak, "status"] != "linked":
                break
            link = by_peak.loc[peak]
            targets.add((str(link.group_id), str(link.resonator_id)))
        else:
            if len(targets) == 1:
                key = targets.pop()
                result.setdefault(key, []).append(
                    {
                        "apt_uid": row["detector"],
                        "status": row["status"],
                        "peaks": peaks,
                    }
                )
    return result


def qualify_components(components, links, dark_peaks, candidates):
    """Keep physical class, association, finding flags and quality independent."""
    flagged = links[(links.status == "linked") & links.peak_id.isin(dark_peaks)]
    flagged_keys = set(zip(flagged.group_id, flagged.resonator_id, strict=True))
    rows = []
    for row in components.to_dict("records"):
        key = str(row["group_id"]), str(row["resonator_id"])
        evidence = candidates.get(key, [])
        row.update(
            response_class="dark_candidate" if evidence else "unknown",
            association_status=";".join(sorted({e["status"] for e in evidence}))
            or "model_local_only",
            apt_uids=";".join(str(e["apt_uid"]) for e in evidence),
            candidate_evidence=json.dumps(evidence),
            finder_dark_candidate=key in flagged_keys,
            calibration_suitability="not_assessed",
            science_qualified=False,
            science_exclusion="Optical response and readout suitability unverified",
        )
        row["response_evidence"] = (
            "APT dark candidate (tentative association)"
            if evidence
            else "Finder dark flag only"
            if key in flagged_keys
            else "No response classification"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def group_frame(group):
    """Read a one-dimensional saved evidence table without changing its store."""
    return pd.DataFrame({name: group[name][:] for name in group.array_keys()})


def build(root, identity_study):
    """Write a qualification ledger for both fit families and both Tune blocks."""
    source_rows = json.loads((identity_study / "rows.json").read_text())[
        "descendant-n0"
    ]
    products = json.loads((identity_study / "model-products.json").read_text())
    publication = json.loads((identity_study / "publications.json").read_text())[
        "descendant"
    ]
    old = zarr.open_group(Path(products["models"]["0"]) / "array.zarr", mode="r")
    evidence = {
        "interpretation": "Dark-candidate evidence; physical classes unconfirmed",
        "source_study": str(identity_study),
        "source_apt": publication,
        "source_models": products["models"]["0"],
        "source_files": {
            name: hashlib.sha256((identity_study / name).read_bytes()).hexdigest()
            for name in ("rows.json", "publications.json", "model-products.json")
        },
        "verified_measurements": {},
    }
    tables = []
    for family, folder, run_name in (
        ("Affine", "coverage", "0000-models"),
        ("Delay", "coverage-delay", "0001-models"),
    ):
        product = (
            root
            / "dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc"
            / run_name
        )
        current = zarr.open_group(product / "array.zarr", mode="r")
        for block_id in (0, 1):
            scope = f"toltec0/blocks/{block_id}"
            block = current[scope]
            if not isinstance(block, zarr.Group):
                msg = f"Expected acquired block group: {scope}"
                raise TypeError(msg)
            evidence["verified_measurements"][f"{family}/{scope}"] = (
                verify_measurements(block, old[scope])
            )
            links = group_frame(block["model_fit/peak_links"])
            reference = [r for r in source_rows if r["block"] == block_id]
            candidates = candidate_links(reference, links)
            peaks = group_frame(block["kids_find/d21/peaks"])
            dark_peaks = peaks.loc[peaks.sbm_dark.astype(bool), "idx"]
            file = (
                root / folder / "measured/output-quality.csv"
                if block_id == 1
                else root / folder / "block0/components.csv"
            )
            components = pd.read_csv(file, dtype={"group_id": str, "resonator_id": str})
            result = qualify_components(components, links, dark_peaks, candidates)
            result["family"], result["block"] = family, block_id
            result["fit_index"] = str(product / "index.yaml")
            result["apt_index"] = publication["index"]
            tables.append(result)
    table = pd.concat(tables, ignore_index=True)
    out = root / "response-qualification"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "components.csv", index=False)
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (out / "reference-associations.json").write_text(
        json.dumps(source_rows, indent=2) + "\n"
    )
    return table


def panels(root):
    """Summarize evidence first; Qr and a are measurements, not class labels."""
    table = pd.read_csv(root / "response-qualification/components.csv")
    final = table[table.block == 1]
    rows = []
    for (family, evidence), group in final.groupby(
        ["family", "response_evidence"], sort=False
    ):
        rows.append(
            {
                "Model": family,
                "Response evidence": evidence,
                "Fitted": len(group),
                "Complete native record": int((group.psd_status == "measured").sum()),
                "Fit at bound": int(group.at_bound.sum()),
                "Qualified for optical sensitivity": int(group.science_qualified.sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "response-qualification/summary.csv", index=False)
    figure = go.Figure()
    selected = final[
        (final.family == "Affine")
        & (final.multiplicity == 1)
        & (final.psd_status == "measured")
    ]
    for label, color, symbol in (
        ("No response classification", "#868e96", "circle"),
        ("Finder dark flag only", "#e67700", "diamond"),
        ("APT dark candidate (tentative association)", "#1971c2", "star"),
    ):
        group = selected[selected.response_evidence == label]
        figure.add_trace(
            go.Scatter(
                x=group.Qr,
                y=group.frequency_asd_5_20_hz,
                mode="markers",
                name=label,
                marker={"color": color, "symbol": symbol, "size": 9},
                customdata=group[
                    ["group_id", "resonator_id", "a_nl", "at_bound"]
                ].to_numpy(),
                hovertemplate="Group %{customdata[0]}, component %{customdata[1]}"
                "<br>Qr=%{x:.0f}<br>ASD=%{y:.3g} Hz/√Hz"
                "<br>a=%{customdata[2]:.3g}<br>At bound=%{customdata[3]}"
                "<extra></extra>",
            )
        )
    figure.update_layout(
        title="145609 final block: response evidence before sensitivity comparisons",
        xaxis={"title": "Fitted Qr", "type": "log"},
        yaxis={"title": "Frequency ASD, 5-20 Hz [Hz/√Hz]", "type": "log"},
        legend={"orientation": "h", "y": -0.2},
        height=560,
        margin={"b": 120},
    )
    return summary.to_html(index=False, border=0), figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("identity_study", type=Path)
    args = parser.parse_args()
    data = build(args.root.resolve(), args.identity_study.resolve())
    panels(args.root.resolve())
    print(data.groupby(["family", "block", "response_class"]).size().to_string())  # noqa: T201
