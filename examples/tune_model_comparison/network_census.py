#!/usr/bin/env python
"""Account for every 145609 component independently of response-law selection.

Read the saved comparison and the latest consolidated Skydip/APT receipts.
Reconcile the existing correspondence edges; do not refit or allocate identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import zarr
from plotly.subplots import make_subplots
from zarr.storage import ZipStore

EVIDENCE = {
    "individual_law": "Individual Skydip response law",
    "loading_only": "Loading response; outside individual-law study",
    "response_conflict": "Dark indicator / loading-response conflict",
    "transfer_missing": "145609 identity transfer unresolved",
}
READOUT = {
    "pass": "Passes comparison checks",
    "quality_screen": "Complete solving; comparison excludes",
    "invalid_samples": "Selected; incomplete valid samples",
    "unselected": "No selected solver route",
}
COLORS = {
    "pass": "#1971c2",
    "quality_screen": "#e67700",
    "invalid_samples": "#a61e4d",
    "unselected": "#868e96",
}
FAMILIES = (
    ("Affine", "coverage", "0000-models"),
    ("Delay", "coverage-delay", "0001-models"),
)


def readout_state(row):
    """Distinguish missing selection, invalid samples and quality exclusions."""
    if row["native_route"] == "unselected":
        return "unselected"
    if row["psd_status"] != "measured":
        return "invalid_samples"
    return "pass" if row["readout_suitable"] else "quality_screen"


def evidence_state(row):
    """Keep missing correspondence separate from response and readout evidence."""
    if row["association"] != "supported":
        return "transfer_missing"
    if row["dark_evidence"]:
        return "response_conflict"
    return "individual_law" if pd.notna(row["beta_skydip"]) else "loading_only"


def association_reason(row):
    """Explain the existing consensus rule without turning it into fit quality."""
    if row["association"] == "supported":
        return "supported"
    if row["association"] != "unresolved":
        return row["association"]
    alternatives = str(row["alternatives"]) if pd.notna(row["alternatives"]) else ""
    if ";" in alternatives:
        return "competing_roster_uids"
    if alternatives:
        return "fewer_than_five_unique_scans"
    return "no_roster_receipt_inside_match_window"


def latest_associations(edges, continuity, established, recipe):
    """Resolve saved finding edges against current exact accepted receipts."""
    lookup = {}
    for group in continuity.values():
        for component in group.values():
            uid = component["uid"]
            if uid not in established:
                continue
            for observation in component["observations"]:
                if observation["uids"] == [uid] and observation["baseline"]:
                    key = recipe["finding_key"](
                        observation["binding"], observation["row"]
                    )
                    lookup.setdefault(key, set()).add(uid)
    updated = edges.copy()
    updated["uids"] = [
        ";".join(
            map(
                str,
                sorted(
                    lookup.get(
                        recipe["finding_key"](
                            json.loads(row.reference_binding), row.ref_row
                        ),
                        set(),
                    )
                ),
            )
        )
        for row in updated.itertuples()
    ]
    result = recipe["consensus"](updated)
    result["reason"] = result.apply(association_reason, axis=1)
    return result, int((updated.uids != edges.uids).sum())


def census_tables(ledger):
    """Return exhaustive counts and an evidence/readout cross-tabulation."""
    rows, cross = [], []
    for family, group in ledger.groupby("family", sort=False):
        for kind in ("Singlet", "Multiplet"):
            subset = group[group.kind == kind]
            rows.append(
                {
                    "Model": family,
                    "Components": kind,
                    "Fitted": len(subset),
                    **{
                        label: int((subset.readout_state == state).sum())
                        for state, label in READOUT.items()
                    },
                }
            )
            for evidence, label in EVIDENCE.items():
                part = subset[subset.evidence_state == evidence]
                cross.append(
                    {
                        "Model": family,
                        "Components": kind,
                        "Evidence": label,
                        "Total": len(part),
                        **{
                            name: int((part.readout_state == state).sum())
                            for state, name in READOUT.items()
                        },
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(cross)


def quality_detail(root, ledger):
    """Decode recorded fit bits and expose overlapping exclusion checks."""
    rows, selection = [], []
    for family, folder, run in FAMILIES:
        data = ledger[ledger.family == family]
        model = (
            root
            / "dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc"
            / run
            / "array.zarr"
        )
        block = zarr.open_group(model, mode="r")["toltec0/blocks/1/model_fit"]
        assert isinstance(block, zarr.Group)
        quality = block["quality"]
        assert isinstance(quality, zarr.Group)
        flags = quality.attrs["flag_definitions"]
        assert isinstance(flags, str)
        definitions = json.loads(flags)
        for name, bit in definitions.items():
            count = int(((data.group_flag.astype(int) & bit) != 0).sum())
            rows.append({"Model": family, "Check": "fit: " + name, "Components": count})
        for reason in (
            "sweep_residual",
            "python_native_tolerance",
            "bad_readout_channel",
        ):
            count = int(
                data.comparison_exclusions.fillna("")
                .str.split(";")
                .map(lambda x, reason=reason: reason in x)
                .sum()
            )
            rows.append({"Model": family, "Check": reason, "Components": count})
        plan = block["plan"]
        assert isinstance(plan, zarr.Group)
        plan = pd.DataFrame({key: value[:] for key, value in plan.arrays()})
        channels = pd.read_csv(
            root / folder / "block1/channels.csv", dtype={"group_id": str}
        ).set_index("chan")
        for row in data[data.native_route == "unselected"].itertuples():
            channel_ids = plan.loc[plan.group_id == row.group_id, "chan"].unique()
            channel_rows = channels.loc[channel_ids]
            selection.append(
                {
                    "family": family,
                    "group_id": row.group_id,
                    "resonator_id": row.resonator_id,
                    "channels": ";".join(map(str, channel_ids)),
                    "reasons": ";".join(sorted(channel_rows.recovery_reason.unique())),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(selection)


def roster_ledger(coverage, assessment, response, associations):
    """Account for established UIDs whether or not transfer to 145609 succeeds."""
    laws = {uid for row in response["contributors"] for uid in row["uids"]}
    by_uid = {row["uid"]: row for row in assessment}
    target = associations[associations.block == 1]
    rows = []
    for detector in coverage["detectors"]:
        uid = detector["uid"]
        supported = target[
            (target.association == "supported") & (target.apt_uid == uid)
        ]
        alternatives = target.alternatives.fillna("").map(
            lambda x, uid=uid: str(uid) in str(x).split(";")
        )
        rows.append(
            {
                "apt_uid": uid,
                "roster_state": detector["state"],
                "skydip_motion": by_uid.get(uid, {}).get("motion", "not_reassessed"),
                "individual_loss_law": uid in laws,
                "transfer_supported": len(supported) == 1,
                "target_finding": ";".join(supported.finding_id),
                "candidate_findings": ";".join(target.loc[alternatives, "finding_id"]),
                "source_reference": json.dumps(coverage["reference"], sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def source_cohort(cohort):
    """Count source-study selection, not detector loss or target-fit failure."""
    rows = []
    for row in cohort:
        state = (
            "Excluded by isolated-response-study scope"
            if len(row["traces"]) > 1
            else "Individual response-law contributor"
            if row.get("eligible")
            else "Isolated trace; source-study quality/evidence checks exclude"
        )
        rows.append(
            {
                "group": row["key"],
                "traces": len(row["traces"]),
                "state": state,
                "reasons": "; ".join(row["reasons"]),
            }
        )
    return pd.DataFrame(rows)


def plot(ledger):
    """Show the whole network with quality independent of evidence coverage."""
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("Singlet components", "Multiplet components"),
    )
    labels = [
        "Individual Skydip<br>response law",
        "Loading response;<br>outside individual-law study",
        "Dark indicator /<br>loading-response conflict",
        "145609 identity<br>transfer unresolved",
    ]
    for col, kind in enumerate(("Singlet", "Multiplet"), 1):
        for state, label in READOUT.items():
            part = ledger[(ledger.family == "Affine") & (ledger.kind == kind)]
            counts = [
                int(((part.evidence_state == ev) & (part.readout_state == state)).sum())
                for ev in EVIDENCE
            ]
            figure.add_trace(
                go.Bar(
                    x=counts,
                    y=list(EVIDENCE.values()),
                    name=label,
                    orientation="h",
                    marker_color=COLORS[state],
                    legendgroup=state,
                    showlegend=col == 1,
                    text=[str(n) if n else "" for n in counts],
                    textposition="inside",
                    hovertemplate=(
                        "%{y}<br>%{x} components<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        figure.update_xaxes(
            title_text="Model components", range=[0, 265], row=1, col=col
        )
        figure.update_yaxes(
            autorange="reversed",
            tickvals=list(EVIDENCE.values()),
            ticktext=labels,
            row=1,
            col=col,
        )
    figure.update_layout(
        barmode="stack",
        template="plotly_white",
        height=610,
        font={"size": 15},
        margin={"l": 230, "b": 140},
        legend={"orientation": "h", "y": -0.25},
    )
    return figure.to_html(
        full_html=False, include_plotlyjs=False, div_id="evidence-readout"
    )


def render(output, tables, ledger, summary):
    """Render the maintained census and preserve the previous comparison."""
    template = Path(__file__).with_suffix(".html").read_text()
    style = (
        Path(__file__)
        .with_name("network_active.html")
        .read_text()
        .split("<style>")[1]
        .split("</style>")[0]
    )
    values = {
        "STYLE": style,
        "FIGURE": plot(ledger),
        "COUNTS": json.dumps(summary, indent=2),
    }
    for name, source_table in tables.items():
        table = source_table.copy()
        if name == "transfer":
            table = table.pivot_table(
                index="reason", columns="block", values="Components"
            ).reset_index()
            table["reason"] = table.reason.map(
                {
                    "supported": "Supported in both Tune blocks",
                    "no_roster_receipt_inside_match_window": (
                        "No roster-linked edge inside match window"
                    ),
                    "fewer_than_five_unique_scans": (
                        "Fewer than five distinct unique Skydip scans"
                    ),
                    "competing_roster_uids": "Competing roster UIDs",
                    "not_supported_in_both_tune_blocks": (
                        "Support missing from other Tune block"
                    ),
                }
            )
            table = table.rename(
                columns={
                    "reason": "Association evidence",
                    0: "Tune block 0",
                    1: "Tune block 1",
                }
            )
        elif name == "quality":
            table = table[(table.Affine > 0) | (table.Delay > 0)]
        elif name == "source":
            table = table.rename(columns={"state": "Source-study selection"})
        table.columns.name = None
        values[name.upper()] = table.to_html(
            index=False, border=0, float_format=lambda v: f"{v:.4g}"
        )
    for name, value in values.items():
        template = template.replace("{{" + name + "}}", value)
    (output / "index.html").write_text(template)


def main(root, evidence):  # noqa: PLR0915 -- one read-only audit with explicit accounting.
    """Rebuild the network census from pinned numerical results and source receipts."""
    output = root / "network-census"
    output.mkdir(exist_ok=True)
    active = root / "active-response"
    source = evidence / "nw0-workflow-consolidation-20261121"
    response_dir = evidence / "nw0-network-response-20261121"
    paths = {
        "components": active / "components.csv",
        "old_associations": active / "associations.csv",
        "edges": active / "match-edges.csv",
        "coverage": source / "coverage.json",
        "continuity": source / "continuity-with-uids.json",
        "assessment": source / "assessment-with-uids.json",
        "roster_pointer": source / "roster-index.json",
        "response": response_dir / "results.json",
        "response_cohort": response_dir / "cohort.json",
    }
    data = {
        key: json.loads(path.read_text())
        for key, path in paths.items()
        if path.suffix == ".json"
    }
    roster_path = Path(data["roster_pointer"]["index"])
    paths.update(
        roster=roster_path / "roster.zarr.zip", roster_index=roster_path / "index.yaml"
    )
    manifest = {
        key: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for key, path in paths.items()
    }
    (output / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    recipe = runpy.run_path(str(Path(__file__).with_name("network_active.py")))
    established = {row["uid"] for row in data["coverage"]["detectors"]}
    with ZipStore(paths["roster"], mode="r") as store:
        saved = zarr.open_group(store, mode="r")["detectors"]
        assert isinstance(saved, zarr.Group)
        detectors = pd.DataFrame({key: value[:] for key, value in saved.arrays()})
    assert established == set(
        detectors.loc[(detectors.nw == 0) & (detectors.status == "accepted"), "uid"]
    )
    edges = pd.read_csv(
        paths["edges"], dtype={"finding_id": str}, keep_default_na=False
    )
    latest, changed_edges = latest_associations(
        edges, data["continuity"], established, recipe
    )
    latest.to_csv(output / "associations.csv", index=False)
    previous = pd.read_csv(paths["old_associations"], dtype={"finding_id": str})
    merged = latest.merge(
        previous, on=["block", "finding_id"], suffixes=("_latest", "_previous")
    )
    changes = merged[
        (merged.association_latest != merged.association_previous)
        | (merged.apt_uid_latest.fillna(-1) != merged.apt_uid_previous.fillna(-1))
    ]
    changes.to_csv(output / "association-changes.csv", index=False)
    # A published numerical comparison cannot silently inherit a different UID.
    for block in (0, 1):
        old = (
            previous[(previous.block == block) & (previous.association == "supported")]
            .set_index("finding_id")
            .apt_uid
        )
        new = (
            latest[(latest.block == block) & (latest.association == "supported")]
            .set_index("finding_id")
            .apt_uid
        )
        pd.testing.assert_series_equal(old.sort_index(), new.sort_index())
    ledger = pd.read_csv(
        paths["components"], dtype={"group_id": str, "resonator_id": str}
    )
    ledger = ledger.drop(
        columns=["association", "alternatives", "apt_uid", "matched_scans"]
    )
    transfer = latest[latest.block == 1].drop(
        columns=["block", "match_residual_median_hz"]
    )
    ledger = ledger.merge(
        transfer, left_on="resonator_id", right_on="finding_id", validate="many_to_one"
    )
    ledger = ledger.rename(columns={"apt_roster_index": "comparison_apt_roster_index"})
    ledger["apt_roster_index"] = str(paths["roster_index"])
    ledger["apt_roster_reference"] = json.dumps(
        data["coverage"]["reference"], sort_keys=True
    )
    ledger["kind"] = np.where(ledger.multiplicity == 1, "Singlet", "Multiplet")
    ledger["evidence_state"] = ledger.apply(evidence_state, axis=1)
    ledger["readout_state"] = ledger.apply(readout_state, axis=1)
    ledger.to_csv(output / "components.csv", index=False)
    census, cross = census_tables(ledger)
    quality, selection = quality_detail(root, ledger)
    roster = roster_ledger(
        data["coverage"], data["assessment"], data["response"], latest
    )
    roster.to_csv(output / "roster-coverage.csv", index=False)
    cohort = source_cohort(data["response_cohort"])
    cohort.to_csv(output / "source-response-selection.csv", index=False)
    transfer_table = (
        latest.groupby(["block", "reason"], sort=False)
        .size()
        .rename("Components")
        .reset_index()
    )
    sources = (
        cohort.groupby("state", sort=False)
        .agg(Groups=("group", "size"), Traces=("traces", "sum"))
        .reset_index()
    )
    affine = ledger[ledger.family == "Affine"]
    partitions = pd.crosstab(
        affine.evidence_state.map(EVIDENCE), affine.readout_state.map(READOUT)
    ).reindex(
        index=list(EVIDENCE.values()), columns=list(READOUT.values()), fill_value=0
    )
    partitions["Total"] = partitions.sum(axis=1)
    partitions = partitions.reset_index()
    partitions.columns.name = None
    partitions = partitions.rename(columns={"evidence_state": "Response evidence"})
    warning_table = quality.pivot_table(
        index="Check", columns="Model", values="Components"
    ).reset_index()
    tables = {
        "census": census,
        "cross": cross,
        "source": sources,
        "transfer": transfer_table,
        "partitions": partitions,
        "quality": warning_table,
    }
    for key, table in tables.items():
        table.to_csv(output / f"{key}.csv", index=False)
    selection.to_csv(output / "unselected-components.csv", index=False)
    summary = {
        "established_network0_uids": len(established),
        "reassessed_uids": int((roster.roster_state == "reassessed").sum()),
        "retained_reviewed_uids": int((roster.roster_state == "retained_review").sum()),
        "response_laws": int(roster.individual_loss_law.sum()),
        "matched_to_145609": int(roster.transfer_supported.sum()),
        "unresolved_transfer_uids": int((~roster.transfer_supported).sum()),
        "final_block_components_per_family": len(affine),
        "changed_correspondence_receipts": changed_edges,
        "changed_supported_associations": 0,
        "changed_pending_associations": len(changes),
        "qualified_comparison": int((affine.population == "active_suitable").sum()),
        "source_physical_classification": (
            "Not inferred from identity, empirical motion or a fit failure"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in manifest.values():
        assert (
            hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
        )
    assert len(ledger) == census.Fitted.sum() == cross.Total.sum()
    assert set(roster.apt_uid) == established
    assert roster.apt_uid.is_unique
    assert not ledger.duplicated(["family", "group_id", "resonator_id"]).any()
    render(output, tables, ledger, summary)
    print(json.dumps(summary, indent=2))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    main(args.root.resolve(), args.evidence.resolve())
