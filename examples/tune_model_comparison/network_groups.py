#!/usr/bin/env python
"""Reconcile complete 145609 groups against exact Skydip/APT receipts.

This local evidence recipe makes provisional observation correspondences. It
never writes the roster or changes saved fits, stubs, or native solved samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tollan.utils.log import logger
from zarr.storage import ZipStore

from tolteca_kidsproc.calibration import group_covariance
from tolteca_kidsproc.response import fit_loss_response
from tolteca_kidsproc.segmentation import ordered_match_support

HERE = Path(__file__).parent
ACTIVE = runpy.run_path(str(HERE / "network_active.py"))
RESPONSE = runpy.run_path(str(HERE / "network_response.py"))
CENSUS = runpy.run_path(str(HERE / "network_census.py"))
SCANS = tuple(range(2, 9))
SPAN_SNR_MIN = 10
SOURCE_FILES = {}


def read_json(path):
    """Read a pinned local evidence artifact."""
    content = path.read_bytes()
    SOURCE_FILES[str(path.resolve())] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return json.loads(content)


def dump(path, value):
    """Write portable strict JSON; nonfinite floating values become null."""

    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(w) for k, w in v.items()}
        if isinstance(v, (tuple, list)):
            return [clean(w) for w in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def grouped_edges(query, reference, edges):
    """Constrain all gated candidate edges by complete local group topology.

    Both lists include findings without an APT UID. Entire native/finding groups
    remain together, including members with no edge. Strict frequency order is
    conditional on the same no-crossing assumption as the source DTW relation.
    """
    edges = list(edges)
    query, reference = query.reset_index(drop=True), reference.reset_index(drop=True)
    nq, nr = len(query), len(reference)
    qi = {v: i for i, v in enumerate(query.key)}
    ri = {v: i for i, v in enumerate(reference.key)}
    links = []
    for data, offset in ((query, 0), (reference, nq)):
        for _, part in data.groupby("group", sort=False):
            indices = part.index.to_numpy() + offset
            links.extend((indices[0], k) for k in indices)
    links.extend((qi[a], nq + ri[b]) for a, b in edges)
    if links:
        a, b = np.asarray(links).T
        graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(nq + nr, nq + nr))
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(nq + nr)
    outputs, groups = [], []
    edge_set = set(edges)
    for group in np.unique(labels):
        q = query.iloc[np.flatnonzero(labels[:nq] == group)].sort_values("frequency")
        r = reference.iloc[np.flatnonzero(labels[nq:] == group)].sort_values(
            "frequency"
        )
        allowed = np.array(
            [[(a, b) in edge_set for b in r.key] for a in q.key], bool
        ).reshape(len(q), len(r))
        ordered = all(np.all(np.diff(d.frequency) > 0) for d in (q, r))
        feasible, forced = (
            ordered_match_support(allowed)
            if ordered
            else (np.zeros_like(allowed), np.zeros_like(allowed))
        )
        groups.append(
            {
                "group": int(group),
                "query": q.key.tolist(),
                "reference": r.key.tolist(),
                "ordered": ordered,
                "complete": bool(feasible.any()),
                "edges": int(allowed.sum()),
                "feasible_edges": int(feasible.sum()),
                "forced_edges": int(forced.sum()),
            }
        )
        for i, a in enumerate(q.key):
            for j, b in enumerate(r.key):
                if allowed[i, j]:
                    outputs.append(
                        {
                            "finding_id": a,
                            "ref_row": b,
                            "reconciliation_group": int(group),
                            "feasible": bool(feasible[i, j]),
                            "forced": bool(forced[i, j]),
                            "complete": bool(feasible.any()),
                        }
                    )
    return pd.DataFrame(outputs), groups


def reconcile(root, evidence, output):  # noqa: C901 -- retain the complete correspondence audit.
    """Use latest exact receipts; retain candidates, gaps and both-block checks."""
    consolidated = evidence / "nw0-workflow-consolidation-20261121"
    continuity = read_json(consolidated / "continuity-with-uids.json")
    assessments = read_json(consolidated / "assessment-with-uids.json")
    records = read_json(evidence / "skydip-groups-followup-20261112/all-inputs.json")[
        "records"
    ]
    lookup, owners = {}, {}
    for name, components in continuity.items():
        result = read_json(consolidated / f"groups/{name}.json")["result"]
        for record in result["records"]:
            for peak in record["peaks"]:
                key = ACTIVE["finding_key"](record["binding"], peak["row"])
                if key in owners and owners[key] != name:
                    msg = "A finding is owned by multiple source groups"
                    raise ValueError(msg)
                owners[key] = name
        for component in components.values():
            uid = component["uid"]
            for observation in component["observations"]:
                if (
                    uid is not None
                    and observation["uids"] == [uid]
                    and observation["baseline"]
                ):
                    key = ACTIVE["finding_key"](
                        observation["binding"], observation["row"]
                    )
                    lookup.setdefault(key, set()).add(uid)
    edges = pd.read_csv(
        root / "active-response/match-edges.csv",
        dtype={"finding_id": str},
        keep_default_na=False,
    )
    tree = zarr.open_group(
        root
        / "dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc"
        / "0000-models/array.zarr",
        mode="r",
    )
    audits, topology = [], []
    for block in (0, 1):
        acquired = tree[f"toltec0/blocks/{block}"]
        if not isinstance(acquired, zarr.Group):
            msg = "Expected the acquired block group"
            raise TypeError(msg)
        detected = ACTIVE["checked_finding_plan"](acquired)
        plan = ACTIVE["frame"](acquired["model_fit/plan"])
        group_by_id = (
            plan.drop_duplicates("resonator_id").set_index("resonator_id").group_id
        )
        query = pd.DataFrame(
            {"key": detected.idx.astype(int).astype(str), "frequency": detected.f}
        )
        query["group"] = query.key.map(group_by_id)
        query["group"] = query.group.fillna("unplanned-" + query.key)
        for record in records:
            scan = record["scan"]
            if scan not in SCANS:
                continue
            ref = pd.DataFrame(
                [
                    {
                        "key": p["row"],
                        "frequency": p["frequency"],
                        "group": owners.get(
                            ACTIVE["finding_key"](record["binding"], p["row"]),
                            f"unknown-{p['row']}",
                        ),
                    }
                    for p in record["peaks"]
                ]
            )
            part = edges[(edges.block == block) & (edges.scan == scan)].copy()
            part["uids"] = [
                ";".join(
                    map(
                        str,
                        sorted(
                            lookup.get(
                                ACTIVE["finding_key"](record["binding"], row), set()
                            )
                        ),
                    )
                )
                for row in part.ref_row
            ]
            near = part[part.residual_hz.abs() <= ACTIVE["MATCH_GATE_HZ"]]
            constrained, groups = grouped_edges(
                query, ref, zip(near.finding_id, near.ref_row, strict=True)
            )
            # Only geometry removes an alternative. An unmatchable complete group
            # keeps all its candidates as vetoes, including unknown identities.
            part = part.merge(constrained, on=["finding_id", "ref_row"], how="left")
            part["retained"] = part.residual_hz.abs().le(ACTIVE["MATCH_GATE_HZ"]) & (
                ~part.complete.fillna(value=False).astype(bool)
                | part.feasible.fillna(value=False).astype(bool)
            )
            part["support"] = part.forced.fillna(value=False).astype(bool)
            audits.append(part)
            topology.extend(dict(block=block, scan=scan, **g) for g in groups)
    audit = pd.concat(audits, ignore_index=True)
    audit.to_csv(output / "group-edges.csv", index=False)
    dump(output / "group-topology.json", topology)
    associations = reconcile_consensus(audit)
    associations.to_csv(output / "associations.csv", index=False)
    dump(output / "assessment.json", assessments)
    return associations, continuity, lookup


def reconcile_consensus(audit, min_scans=5):
    """Require forced repeated support, preserving unknown and competing identities."""
    rows = []
    for (block, finding), group in audit.groupby(["block", "finding_id"], sort=False):
        retained = group[group.retained]
        uids = {int(u) for s in retained.uids for u in s.split(";") if u}
        unknown = bool((retained.uids == "").any())
        count = int(
            group[
                group.support & group.uids.ne("") & ~group.uids.str.contains(";")
            ].scan.nunique()
        )
        supported = len(uids) == 1 and not unknown and count >= min_scans
        reason = (
            "supported"
            if supported
            else (
                "unknown_reference_candidate"
                if unknown
                else "competing_roster_uids"
                if len(uids) > 1
                else "insufficient_group_support"
                if uids
                else "no_roster_correspondence"
            )
        )
        rows.append(
            {
                "block": block,
                "finding_id": finding,
                "apt_uid": next(iter(uids)) if supported else None,
                "matched_scans": count,
                "alternatives": ";".join(map(str, sorted(uids))),
                "association": reason,
            }
        )
    associations = pd.DataFrame(rows)
    selected = associations[associations.association == "supported"]
    duplicate = selected.duplicated(["block", "apt_uid"], keep=False)
    associations.loc[selected.index[duplicate], "association"] = (
        "conflicting_target_findings"
    )
    for _, group in associations[associations.association == "supported"].groupby(
        "apt_uid"
    ):
        if set(group.block) != {0, 1}:
            associations.loc[group.index, "association"] = (
                "not_supported_in_both_tune_blocks"
            )
    return associations


def native_point(source):
    """Read a labelled component marginal from its full native group covariance."""
    with xr.open_datatree(source["path"], engine="zarr", consolidated=True) as tree:
        model = tree["tolteca_kidsproc.analysis.model_fit"]
        fit = model["fit"].to_dataset()
        rid, gid = source["resonator_id"], source["group_id"]
        fit = fit.isel(
            row=np.flatnonzero(
                (fit.resonator_id.to_numpy() == rid) & (fit.group_id.to_numpy() == gid)
            )
        )
        if not fit.sizes["row"]:
            msg = "Missing explicitly bound source component"
            raise ValueError(msg)
        fr, r, a = [float(fit[k].to_numpy()[0]) for k in ("fr", "r", "a_nl")]
        for key in ("fr", "r", "a_nl"):
            np.testing.assert_allclose(fit[key], fit[key].to_numpy()[0])
        labels = [f"resonator:{rid}:fr", f"resonator:{rid}:r"]
        cov = (
            group_covariance(model, gid)
            .sel(parameter=labels, parameter_other=labels)
            .to_numpy()
        )
        if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() <= 0:
            msg = "No positive definite component covariance"
            raise ValueError(msg)
        return {
            "fr": fr,
            "r": r,
            "a_nl": a,
            "covariance_fr_r": cov.tolist(),
            "source": source,
        }


def response_laws(evidence, output, lookup):  # noqa: C901, PLR0912, PLR0915 -- independent evidence and quality checks.
    """Remove the isolation-only restriction; require independent exact receipts.

    Seven predeclared Skydip acquisitions supply the same loss/detuning experiment
    for singlets and group constituents. Unknown/ambiguous model bindings remain
    gaps. Quality warnings are retained separately from missing evidence.
    """
    source_root = evidence / "nw0-population-baseline-20261119"
    assessment = read_json(output / "assessment.json")
    rows, tasks = [], {}
    for item in assessment:
        uid = item["uid"]
        row = {
            "uid": uid,
            "group": item["group"],
            "component": item["component"],
            "motion": item["motion"],
            "baseline_state": item["baseline_state"],
            "points": [],
            "gaps": [],
            "quality": [],
        }
        rows.append(row)
        if uid is None or item["baseline_state"] != "supported":
            row["gaps"].append("no_supported_identity_baseline")
            continue
        for scan in SCANS:
            path = source_root / item["group"] / f"controls/scan{scan}.json"
            if not path.exists():
                row["gaps"].append(f"scan{scan}:missing_native_control")
                continue
            control = read_json(path)
            states = [
                s for s in control["states"] if s["component"] == item["component"]
            ]
            if len(states) != 1:
                row["gaps"].append(f"scan{scan}:model_component_not_unique")
                continue
            state = states[0]
            # A local integer in the control is insufficient: require the exact
            # source finding to have this accepted roster observation receipt.
            result = read_json(source_root / item["group"] / "result.json")
            record = next(r for r in result["records"] if r["scan"] == scan)
            peaks = [
                p
                for p in record["peaks"]
                if p["resonator_id"] == state["source"]["resonator_id"]
            ]
            if len(peaks) != 1 or lookup.get(
                ACTIVE["finding_key"](control["binding"], peaks[0]["row"])
            ) != {uid}:
                row["gaps"].append(f"scan{scan}:no_exact_accepted_finding_receipt")
                continue
            q = state["quality"]
            reasons = []
            if not state["usable"] or q["native_flags"] != "none" or q["bound_names"]:
                reasons.append("source_fit_warning")
            if q["uncertainty_status"] != "local_covariance":
                row["gaps"].append(f"scan{scan}:no_local_covariance")
                continue
            if control["holdout_exceptions"]:
                reasons.append("source_edge_coverage")
            if any(c["bad"] for c in control["channel_flags"].values()):
                reasons.append("source_bad_channel")
            # Multiplicity is not a quality veto. Measured feature failures are.
            forbidden = {"edge", "peak_small", "snr_low", "not_real"}
            reasons.extend(sorted(forbidden.intersection(peaks[0]["flag_names"])))
            if not {"d21", "s21"} <= set(peaks[0]["flag_names"]):
                row["gaps"].append(f"scan{scan}:no_dual_finding_support")
            row["quality"].extend(f"scan{scan}:{r}" for r in reasons)
            point = {
                "scan": scan,
                "binding": control["binding"],
                "finding_row": peaks[0]["row"],
                "source": state["source"],
                "reasons": reasons,
            }
            row["points"].append(point)
            key = json.dumps(state["source"], sort_keys=True)
            tasks[key] = state["source"]

    def load(item):
        key, source = item
        try:
            return key, native_point(source)
        except (ValueError, KeyError, OSError) as exc:
            return key, {"error": str(exc)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        points = {}
        for i, (key, point) in enumerate(pool.map(load, tasks.items()), start=1):
            points[key] = point
            if i % 100 == 0:
                logger.info("Read {} / {} source covariance records", i, len(tasks))
    laws = {}
    for row in rows:
        for point in row["points"]:
            value = points[json.dumps(point["source"], sort_keys=True)]
            if "error" in value:
                row["gaps"].append(
                    f"scan{point['scan']}:covariance_unavailable:{value['error']}"
                )
            else:
                point.update(value)
        if row["gaps"] or row["quality"] or len(row["points"]) != len(SCANS):
            continue
        p = row["points"]
        fref = float(round(p[0]["fr"] / 1e6) * 1e6)
        fr = np.array([v["fr"] for v in p])
        x, loss = fref / fr - 1, np.array([v["r"] for v in p])
        transform = np.column_stack([-fref / fr**2, np.ones(len(fr))])
        cov = (
            np.array([v["covariance_fr_r"] for v in p])
            * transform[:, :, None]
            * transform[:, None, :]
        )
        row["detuning_span_snr"] = float(np.ptp(x) / np.sqrt(2 * cov[:, 0, 0].max()))
        if row["detuning_span_snr"] < SPAN_SNR_MIN:
            row["gaps"].append("insufficient_loading_leverage")
            continue
        try:
            fit = fit_loss_response(x, loss, cov)
            folds = [
                fit_loss_response(
                    np.delete(x, i), np.delete(loss, i), np.delete(cov, i, axis=0)
                )
                for i in range(len(x))
            ]
        except ValueError as exc:
            row["quality"].append("response_fit:" + str(exc))
            continue
        row.update(
            fref=fref,
            fit=fit,
            folds=folds,
            x_opt=x.tolist(),
            r=loss.tolist(),
            covariance_x_r=cov.tolist(),
            beta=1 / fit["slope"] if fit["slope"] else None,
        )
        # Explicit bridge to the older comparison implementation: its historical
        # beta argument means R_r/R_x, NOT the current conventional beta.
        laws[row["uid"]] = {
            "uids": [row["uid"]],
            "key": row["group"],
            "fref": fref,
            "fit": {"beta": fit["slope"], "beta_error": fit["slope_error"]},
        }
    dump(output / "skydip-response.json", rows)
    logger.info("Prepared {} independent response laws", len(laws))
    return laws


def compare(root, evidence, output, associations, laws):
    """Preserve all target components; compare supported joint native solutions."""
    consolidated = evidence / "nw0-workflow-consolidation-20261121"
    roster = Path(read_json(consolidated / "roster-index.json")["index"])
    sources = {
        "roster_index": str(roster / "index.yaml"),
        "response_by_uid": laws,
        "assessment_by_uid": {
            r["uid"]: r
            for r in read_json(output / "assessment.json")
            if r["uid"] is not None
        },
    }
    ledger = ACTIVE["qualify"](root, sources, associations, output)
    measured, _, _ = ACTIVE["compare"](root, ledger, output)
    # Use slope throughout public output; beta is reserved for R_x/R_r.
    renamed = {
        c: c.replace("beta", "loss_detuning_slope")
        for c in measured.columns
        if "beta" in c
    }
    measured = measured.rename(columns=renamed)
    measured["beta_skydip"] = 1 / measured.loss_detuning_slope_skydip
    measured["kind"] = np.where(measured.multiplicity == 1, "Singlet", "Multiplet")
    measured["readout_state"] = measured.apply(CENSUS["readout_state"], axis=1)
    measured["evidence_state"] = np.select(
        [
            measured.association != "supported",
            measured.dark_evidence,
            measured.loss_detuning_slope_skydip.notna(),
        ],
        ["transfer_missing", "response_conflict", "individual_law"],
        default="loading_only",
    )
    measured.to_csv(output / "signal-noise.csv", index=False)
    ledger = ledger.rename(
        columns={
            c: c.replace("beta", "loss_detuning_slope")
            for c in ledger.columns
            if "beta" in c
        }
    )
    ledger.to_csv(output / "components.csv", index=False)
    counts, cross = CENSUS["census_tables"](measured)
    cross["Evidence"] = cross.Evidence.replace(
        {
            "Loading response; outside individual-law study": (
                "Loading response; no qualified component law"
            )
        }
    )
    counts.to_csv(output / "coverage.csv", index=False)
    cross.to_csv(output / "evidence-readout.csv", index=False)
    return measured, counts, cross


def persist_sources(root, evidence, output):
    """Pin receipts and source-model payloads and account for every roster UID."""
    consolidated = evidence / "nw0-workflow-consolidation-20261121"
    paths = set(SOURCE_FILES)
    paths.update(str(p.resolve()) for p in consolidated.glob("groups/*.json"))
    paths.update(
        str((consolidated / name).resolve())
        for name in (
            "continuity-with-uids.json",
            "assessment-with-uids.json",
            "roster-index.json",
            "publication.json",
        )
    )
    paths.add(str((root / "active-response/match-edges.csv").resolve()))
    response = read_json(output / "skydip-response.json")
    for row in response:
        source = evidence / "nw0-population-baseline-20261119" / row["group"]
        paths.add(str((source / "result.json").resolve()))
        paths.update(str(p.resolve()) for p in (source / "controls").glob("scan*.json"))
    # The response artifact snapshots every numerical marginal used by this
    # comparison. Hash that snapshot and each source schema; avoid a manifest
    # containing hundreds of thousands of irrelevant Zarr sample chunks.
    paths.add(str((output / "skydip-response.json").resolve()))
    for path in {p["source"]["path"] for row in response for p in row["points"]}:
        paths.add(str((Path(path) / ".zmetadata").resolve()))
    pointer = read_json(consolidated / "roster-index.json")
    roster = Path(pointer["index"])
    paths.update(
        str((roster / name).resolve()) for name in ("index.yaml", "roster.zarr.zip")
    )
    paths.update(str(p.resolve()) for p in HERE.glob("network_*.py"))
    paths.update(
        str((HERE.parents[1] / "src/tolteca_kidsproc" / name).resolve())
        for name in ("segmentation.py", "response.py")
    )
    manifest = {}
    for name in sorted(paths):
        path = Path(name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in SOURCE_FILES and SOURCE_FILES[name]["sha256"] != digest:
            msg = f"Source changed during reconciliation: {path}"
            raise ValueError(msg)
        manifest[name] = {"path": name, "sha256": digest}
    dump(output / "sources.json", manifest)
    with ZipStore(roster / "roster.zarr.zip", mode="r") as store:
        detectors = ACTIVE["frame"](zarr.open_group(store, mode="r")["detectors"])
    associations = pd.read_csv(output / "associations.csv")
    supported = set(
        associations.loc[associations.association == "supported", "apt_uid"].astype(int)
    )
    detectors = detectors[(detectors.nw == 0) & (detectors.status == "accepted")].copy()
    if not supported <= set(detectors.uid):
        msg = "Correspondence references an unestablished roster identity"
        raise ValueError(msg)
    detectors["target_association"] = np.where(
        detectors.uid.isin(supported), "supported", "unresolved_transfer"
    )
    detectors["roster_index"] = str(roster / "index.yaml")
    detectors.to_csv(output / "roster-coverage.csv", index=False)
    runpy.run_path(str(HERE / "network_active_verify.py"))["verify"](
        root, output, direction_column="loss_detuning_slope_at_probe"
    )


def main():
    """Run the maintained reconciliation and comparison from local saved evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    associations, _continuity, lookup = reconcile(args.root, args.evidence, args.output)
    laws = response_laws(args.evidence, args.output, lookup)
    compare(args.root, args.evidence, args.output, associations, laws)
    persist_sources(args.root, args.evidence, args.output)
    runpy.run_path(str(HERE / "network_groups_report.py"))["render"](
        args.root, args.output
    )


if __name__ == "__main__":
    main()
