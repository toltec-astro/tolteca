#!/usr/bin/env python
"""Compare 145609 only after attaching Skydip response and APT evidence.

Run with the Grant comparison environment and the local evidence directory.
Matching creates reviewable observation associations, never new detector UIDs.
Raw samples, saved fits, stubs, native output and the source roster are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from functools import cache
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import zarr
from astropy.table import Table
from plotly.subplots import make_subplots
from tollan.utils.log import logger
from zarr.storage import ZipStore

from tolteca_kids.match1d import Match1D

DIAGNOSTICS = runpy.run_path(str(Path(__file__).with_name("network_diagnostics.py")))

FAMILIES = (
    ("Affine", "coverage", "0000-models"),
    ("Delay", "coverage-delay", "0001-models"),
)
SCANS = tuple(range(2, 9))
# The established observation-APT matcher uses a 200 kHz correspondence window.
# This study additionally requires repeated unique full-DTW edges and both blocks.
MATCH_GATE_HZ = 200_000
MATCH_MIN_SCANS = 5
RESIDUAL_MAX = DIAGNOSTICS["RESIDUAL_MAX"]
SUMMARY_BAND_HZ = (5, 20)
LABELS = {
    "active_suitable": "Skydip-supported active; suitable readout",
    "active_poor": "Individual response law; readout excluded or pending",
    "loading_only": "Loading response; individual loss law unqualified",
    "dark_candidate": "Dark candidate",
    "response_conflict": "Dark flag / loading-response conflict",
    "unresolved": "Unresolved identity or response",
}
COLORS = {
    "active_suitable": "#1971c2",
    "active_poor": "#e67700",
    "loading_only": "#40a99c",
    "dark_candidate": "#ae3ec9",
    "response_conflict": "#a61e4d",
    "unresolved": "#868e96",
}
FAMILY_COLORS = {"Affine": "#1971c2", "Delay": "#e67700"}


def frame(group):
    """Load a saved one-dimensional table without changing its owner."""
    return pd.DataFrame({key: group[key][:] for key in group.array_keys()})


def finding_key(binding, row):
    """Bind a finding row to one exact processed block, not just an obsnum."""
    return (
        binding["product"]["namespace_uid"],
        binding["product"]["uid"],
        binding["uid_raw_obs"],
        binding["interface"],
        binding["block"],
        int(row),
    )


def read_sources(evidence_root, output):
    """Pin the actual roster and baseline receipts, not a same-number UID join."""
    paths = {
        "findings": evidence_root / "skydip-groups-followup-20261112/all-inputs.json",
        "continuity": evidence_root / "nw0-baseline-replay-20261121/continuity.json",
        "assessment": evidence_root / "nw0-baseline-replay-20261121/assessment.json",
        "response": evidence_root / "nw0-network-response-20261121/results.json",
        "roster_pointer": evidence_root
        / "nw0-companion-promotion-20261120/roster-index.json",
        "baseline_product": evidence_root
        / "nw0-baseline-replay-20261121/publication.json",
    }
    data = {key: json.loads(path.read_text()) for key, path in paths.items()}
    roster = Path(data["roster_pointer"]["index"])
    paths.update(roster=roster / "roster.zarr.zip", roster_index=roster / "index.yaml")
    paths["dark_evidence"] = output.parent / "response-qualification/components.csv"
    paths["dark_sources"] = output.parent / "response-qualification/evidence.json"
    for family, folder, run in FAMILIES:
        paths[f"{family}_model_index"] = (
            output.parent
            / "dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc"
            / run
            / "index.yaml"
        )
        paths[f"{family}_solver_request"] = (
            output.parent / folder / "measured/request.json"
        )
    with ZipStore(paths["roster"], mode="r") as store:
        saved = zarr.open_group(store, mode="r")
        detectors = frame(saved["detectors"])
    established = set(
        detectors.loc[(detectors.nw == 0) & (detectors.status == "accepted"), "uid"]
    )
    lookup = {}
    for components in data["continuity"].values():
        for component in components.values():
            uid = component["uid"]
            if uid is None or uid not in established:
                continue
            for observation in component["observations"]:
                if observation["uids"] != [uid] or not observation["baseline"]:
                    continue
                binding = observation["binding"]
                key = finding_key(binding, observation["row"])
                lookup.setdefault(key, set()).add(uid)
    source_manifest = {
        key: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for key, path in paths.items()
    }
    (output / "sources.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    data["lookup"] = lookup
    data["roster_index"] = str(paths["roster_index"])
    data["response_by_uid"] = {
        row["uids"][0]: row
        for row in data["response"]["contributors"]
        if len(row["uids"]) == 1
    }
    data["assessment_by_uid"] = {
        row["uid"]: row for row in data["assessment"] if row["uid"] is not None
    }
    return data


def checked_finding_plan(block):
    """Verify the stored plan actually uses these finding candidates.

    A fitted model integer has no cross-acquisition meaning. Within this one
    computation, the saved plan explicitly records the original finding ID,
    initial frequency, initial loss and finding group. Require all four.
    """
    detected = frame(block["kids_find/detected"])
    plan = frame(block["model_fit/plan"])
    if not detected.idx.is_unique:
        msg = "Finding IDs must be unique within this acquired block"
        raise ValueError(msg)
    by_id = detected.set_index(detected.idx.astype(str))
    for row in plan.itertuples():
        finding = by_id.loc[row.resonator_id]
        if (
            row.fr_initial != finding.f
            or row.r_initial != 1 / (2 * finding.Qr)
            or row.finding_group_id != str(int(finding.group))
        ):
            msg = "The saved plan is not bound to these finding candidates"
            raise ValueError(msg)
    return detected


def match_evidence(root, sources, output):
    """Save every correspondence; qualify repeatability without looking at noise."""
    path = (
        root
        / "dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc"
        / "0000-models/array.zarr"
    )
    current = zarr.open_group(path, mode="r")
    edges, shifts = [], []
    for block in (0, 1):
        detected = checked_finding_plan(current[f"toltec0/blocks/{block}"])
        for record in sources["findings"]["records"]:
            scan = record["scan"]
            if scan not in SCANS:
                continue
            ref = record["peaks"]
            frequencies = np.array([p["frequency"] for p in ref])
            result = Match1D()(
                query=detected.f.to_numpy() * u.Hz,
                ref=frequencies * u.Hz,
                shift_kw={"dx": 2000 * u.Hz, "shift_max": 1e6 * u.Hz},
            )
            shifts.append(
                {
                    "block": block,
                    "scan": scan,
                    "shift_hz": float(result.shift.to_value(u.Hz)),
                }
            )
            for edge in result.matched:
                i, j = int(edge["idx_query"]), int(edge["idx_ref"])
                uids = sources["lookup"].get(
                    finding_key(record["binding"], ref[j]["row"]), set()
                )
                edges.append(
                    {
                        "block": block,
                        "scan": scan,
                        "finding_id": str(int(detected.idx.iloc[i])),
                        "ref_row": ref[j]["row"],
                        "uids": ";".join(map(str, sorted(uids))),
                        "residual_hz": float(edge["dist_shifted"].to_value(u.Hz)),
                        "unique": bool(
                            edge["n_matched_query"] == 1 and edge["n_matched_ref"] == 1
                        ),
                        "reference_binding": json.dumps(
                            record["binding"], sort_keys=True
                        ),
                    }
                )
    edges = pd.DataFrame(edges)
    edges.to_csv(output / "match-edges.csv", index=False)
    pd.DataFrame(shifts).to_csv(output / "match-shifts.csv", index=False)
    return edges


def consensus(edges, gate_hz=MATCH_GATE_HZ, min_scans=MATCH_MIN_SCANS):
    """Require repeated unique support and retain any competing roster identity.

    Nonunique edges can veto a claim, never supply positive support. A UID cannot
    be assigned to two findings in one block. Both blocks must support it; no
    majority vote silently discards a competing identity.
    """
    rows = []
    for (block, finding), group in edges.groupby(["block", "finding_id"], sort=False):
        near = group[(group.residual_hz.abs() <= gate_hz) & (group.uids != "")]
        alternatives = {int(uid) for value in near.uids for uid in value.split(";")}
        unique = near[near.unique & ~near.uids.str.contains(";")]
        supported = len(alternatives) == 1 and unique.scan.nunique() >= min_scans
        rows.append(
            {
                "block": block,
                "finding_id": str(finding),
                "apt_uid": next(iter(alternatives)) if supported else None,
                "matched_scans": int(unique.scan.nunique()),
                "alternatives": ";".join(map(str, sorted(alternatives))),
                "association": "supported" if supported else "unresolved",
                "match_residual_median_hz": float(unique.residual_hz.abs().median()),
            }
        )
    table = pd.DataFrame(rows)
    supported = table[table.association == "supported"]
    duplicates = supported.duplicated(["block", "apt_uid"], keep=False)
    table.loc[supported.index[duplicates], "association"] = (
        "conflicting_target_findings"
    )
    for _, group in table[table.association == "supported"].groupby("apt_uid"):
        if set(group.block) != {0, 1}:
            table.loc[group.index, "association"] = "not_supported_in_both_tune_blocks"
    return table


def readout_quality(block):
    """Assess fit residuals against measured sweep variation, not total IQ level."""
    quality = block["model_fit/quality"]
    groups = pd.DataFrame(
        {
            key: quality[key][:]
            for key in ("group_id", "group_flag", "reduced_chi2", "residual_fraction")
        }
    )
    samples = frame(block["model_fit/tolteca_kidsproc.analysis.model_fit/samples"])
    samples = samples[samples.fit_selected.astype(bool)]
    variation = {}
    for group, rows in samples.groupby("group_id"):
        measured = rows.I.to_numpy() + 1j * rows.Q.to_numpy()
        residual = rows.I_residual.to_numpy() + 1j * rows.Q_residual.to_numpy()
        # Center within each probe; a gain difference between probes is not a
        # measured resonant excursion. No residual is removed or detrended.
        centered = measured.copy()
        for chan in rows.chan.unique():
            mask = rows.chan.to_numpy() == chan
            centered[mask] -= measured[mask].mean()
        denominator = np.vdot(centered, centered).real
        variation[group] = (
            np.sqrt(np.vdot(residual, residual).real / denominator)
            if denominator > 0
            else np.inf
        )
    groups["residual_to_sweep_variation"] = groups.group_id.map(variation)
    return groups


def suitability(row, *, residual_max=RESIDUAL_MAX):
    """Return explicit exclusions without conflating review cuts with failures."""
    reasons = DIAGNOSTICS["assess"](row, residual_max=residual_max)[
        "comparison_exclusions"
    ]
    return reasons.split(";") if reasons else []


def response_population(*, dark_evidence, motion, has_loss_law, readout_suitable):
    """Keep contradictory response evidence out of both optical and dark cohorts."""
    responsive = motion == "loading_responsive"
    if dark_evidence:
        return "response_conflict" if responsive else "dark_candidate"
    if responsive and has_loss_law:
        return "active_suitable" if readout_suitable else "active_poor"
    return "loading_only" if responsive else "unresolved"


def qualify(root, sources, associations, output):  # noqa: PLR0915 -- explicit independent evidence checks.
    """Attach response evidence first, then independent acquired-readout checks."""
    old = pd.read_csv(
        root / "response-qualification/components.csv",
        dtype={"group_id": str, "resonator_id": str},
    )
    rows = []
    for family, folder, run in FAMILIES:
        product = (
            root / "dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc" / run
        )
        tree = zarr.open_group(product / "array.zarr", mode="r")
        block = tree["toltec0/blocks/1"]
        if not isinstance(block, zarr.Group):
            msg = "Expected a saved acquired-block group"
            raise TypeError(msg)
        detected = checked_finding_plan(block)
        affine = zarr.open_group(
            root
            / "dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc"
            / "0000-models/array.zarr",
            mode="r",
        )
        for key in ("idx", "f", "Qr", "group"):
            reference = affine[f"toltec0/blocks/1/kids_find/detected/{key}"]
            if not isinstance(reference, zarr.Array):
                msg = "Expected a saved finding column"
                raise TypeError(msg)
            np.testing.assert_array_equal(detected[key], reference[:])
        coverage = pd.read_csv(
            root / folder / "measured/output-quality.csv",
            dtype={"group_id": str, "resonator_id": str},
        )
        quality = readout_quality(block).set_index("group_id")
        channels = frame(block["sweep_check/channels"]).set_index("chan")
        plan = frame(block["model_fit/plan"])
        parity = json.loads((root / folder / "measured/solving.json").read_text())
        parity_groups = {
            row["group_id"]: row["parity"]["stub"] for row in parity["joint"]
        }
        assoc = associations[associations.block == 1].set_index("finding_id")
        prior = old[(old.family == family) & (old.block == 1)].set_index(
            ["group_id", "resonator_id"]
        )
        for row in coverage.to_dict("records"):
            gid, rid = row["group_id"], row["resonator_id"]
            row.update(quality.loc[gid].to_dict())
            row.update(assoc.loc[rid].to_dict())
            row["family"] = family
            row["fit_index"] = str(product / "index.yaml")
            row["apt_roster_index"] = sources["roster_index"]
            used = plan.loc[plan.group_id == gid, "chan"].unique()
            row["bad_readout_channel"] = bool(channels.loc[used, "mask_chan_bad"].any())
            check = (
                parity_groups.get(gid)
                if row["multiplicity"] > 1
                else parity["independent"]["stub"]
            )
            row["parity_status"] = "not_assessed"
            row["parity_max_r_difference"] = np.nan
            row["parity_max_frequency_difference_hz"] = np.nan
            if check is not None:
                masks = check["flags_equal"] and all(
                    v["same_finite_mask"]
                    for k, v in check.items()
                    if k != "flags_equal"
                )
                values = all(
                    v["within_tolerance"]
                    for k, v in check.items()
                    if k != "flags_equal"
                )
                row["parity_status"] = (
                    "mask_mismatch"
                    if not masks
                    else ("pass" if values else "tolerance_exceeded")
                )
                row["parity_max_r_difference"] = check["r"]["max_abs_difference"]
                if "fr" in check:
                    row["parity_max_frequency_difference_hz"] = check["fr"][
                        "max_abs_difference"
                    ]
            row["parity_ok"] = row["parity_status"] == "pass"
            row.update(DIAGNOSTICS["assess"](row))
            reasons = suitability(row)
            row["readout_suitable"] = not reasons
            previous = prior.loc[gid, rid]
            row["dark_candidate_evidence"] = previous.candidate_evidence
            row["dark_apt_index"] = previous.apt_index
            row["finder_dark_candidate"] = bool(previous.finder_dark_candidate)
            row["dark_evidence"] = bool(
                previous.finder_dark_candidate
                or previous.response_class == "dark_candidate"
            )
            uid = int(row["apt_uid"]) if pd.notna(row["apt_uid"]) else None
            response = (
                sources["response_by_uid"].get(uid)
                if row["association"] == "supported"
                else None
            )
            baseline = (
                sources["assessment_by_uid"].get(uid)
                if row["association"] == "supported"
                else None
            )
            row["skydip_motion"] = baseline["motion"] if baseline else "unresolved"
            row["population"] = response_population(
                dark_evidence=row["dark_evidence"],
                motion=row["skydip_motion"],
                has_loss_law=response is not None,
                readout_suitable=not reasons,
            )
            row["beta_skydip"] = row["beta_skydip_error"] = np.nan
            row["skydip_fref_hz"] = np.nan
            if row["population"] in {"active_suitable", "active_poor"}:
                assert response is not None
                row["beta_skydip"] = response["fit"]["beta"]
                row["beta_skydip_error"] = response["fit"]["beta_error"]
                row["skydip_fref_hz"] = response["fref"]
                row["response_group"] = response["key"]
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output / "components.csv", index=False)
    return result


@cache
def _coordinate_inputs(folder):
    """Reuse immutable acquired arrays within this one comparison process."""
    request = json.loads((folder / "request.json").read_text())
    model = Table.read(request["ecsv"])
    shape = request["samples"], request["channels"]
    iq = (
        sum(
            np.fromfile(request[k], dtype="float64").reshape(shape) * factor
            for k, factor in (("I", 1), ("Q", 1j))
        )
        / request["normalization"]
    )
    return model, iq


def independent_coordinates(folder, probe):
    """Read x-prime from the actual IQ and the same singlet gain used natively."""
    table, iq = _coordinate_inputs(folder)
    model = table[np.asarray(table["chan"]) == probe][0]
    delta = model["f_in"] - model["f_g"]
    gain = complex(model["H0_re"], model["H0_im"])
    if "H1_re" in model.colnames:
        gain += complex(model["H1_re"], model["H1_im"]) * delta
    else:
        gain *= np.exp((model["a"] - 2j * np.pi * model["tau"]) * delta)
    w = complex(model["c_re"], model["c_im"]) / (1 - iq[:, probe] / gain)
    return w, float(model["b"])


def response_at_probe(beta, error, reference_frequency, probe_frequency):
    """Express a measured loss/detuning slope in the target probe coordinate.

    Skydip uses x = f_ref / fr - 1; the timestream uses x = f_probe / fr - 1.
    Loss has the same normalization in both, so beta scales by f_ref / f_probe.
    """
    scale = reference_frequency / probe_frequency
    return beta * scale, error * scale


def longer_trajectories(x, r, rate, methods):
    """Keep averaging-scale sensitivity and its small-window count explicit."""
    result = {}
    for seconds in (4, 8):
        binned = methods["binned_trajectory"](x, r, rate, seconds)
        slope, correlation = methods["empirical_slope"](*binned)
        result[f"beta_{seconds}s"] = slope
        result[f"correlation_{seconds}s"] = correlation
        result[f"windows_{seconds}s"] = binned.shape[1]
    return result


def compare(root, table, output):
    """Use a detector's own Skydip direction only in its qualified cohort."""
    methods = runpy.run_path(str(Path(__file__).with_name("network_response.py")))
    measured, spectra, trajectories = [], {}, {}
    for family, name, _ in FAMILIES:
        folder = root / name / "measured"
        rate = json.loads((folder / "output-quality-method.json").read_text())[
            "sample_rate_hz"
        ]
        ledger = table[table.family == family].set_index(["group_id", "resonator_id"])
        for base, x, r in methods["component_records"](folder):
            row = ledger.loc[base["group_id"], base["resonator_id"]].to_dict()
            row.update(base)
            if x is None:
                measured.append(row)
                continue
            frequency, covariance = methods["spectral_matrix"](x, r, rate)
            band = (frequency >= SUMMARY_BAND_HZ[0]) & (frequency <= SUMMARY_BAND_HZ[1])
            row["x_asd"] = float(np.sqrt(covariance[band, 0, 0].real.mean()))
            row["r_asd"] = float(np.sqrt(covariance[band, 1, 1].real.mean()))
            row["frequency_asd"] = row["x_asd"] * base["f_probe_hz"]
            key = f"{family}-{base['group_id']}-{base['resonator_id']}"
            spectra[key] = np.array(
                [covariance[:, 0, 0].real, covariance[:, 1, 1].real]
            )
            trajectories[key] = methods["binned_trajectory"](x, r, rate)
            row["beta_1s"], row["correlation_1s"] = methods["empirical_slope"](
                *trajectories[key]
            )
            if row["population"] == "active_suitable":
                beta, error = response_at_probe(
                    row["beta_skydip"],
                    row["beta_skydip_error"],
                    row["skydip_fref_hz"],
                    row["f_probe_hz"],
                )
                row["beta_at_probe"], row["beta_at_probe_error"] = beta, error
                assessed, _ = methods["assess"](
                    x, r, rate, beta, [beta - error, beta + error]
                )
                row.update(assessed)
                row.update(longer_trajectories(x, r, rate, methods))
            if row["native_route"] == "independent":
                w, b = independent_coordinates(folder, row["probe"])
                np.testing.assert_allclose(r, w.real, rtol=0, atol=1e-12)
                np.testing.assert_allclose(
                    x, w.imag - b / abs(w) ** 2, rtol=0, atol=1e-12
                )
                _, prime = methods["spectral_matrix"](w.imag, r, rate)
                xp, r0 = float(np.median(w.imag)), float(np.median(r))
                denominator = r0 * r0 + xp * xp
                D, mu = 1 + 2 * b * xp / denominator**2, 2 * b * r0 / denominator**2
                row.update(
                    D=D, mu=mu, xprime_asd=float(np.sqrt(prime[band, 0, 0].real.mean()))
                )
                predicted = (
                    D * D * prime[:, 0, 0].real
                    + mu * mu * prime[:, 1, 1].real
                    + 2 * D * mu * prime[:, 0, 1].real
                )
                row["coordinate_asd_ratio"] = row["x_asd"] / row["xprime_asd"]
                row["covariance_predicted_ratio"] = float(
                    np.sqrt(predicted[band].mean()) / row["xprime_asd"]
                )
                if row["population"] == "active_suitable":
                    transfer = (1 - mu * row["beta_at_probe"]) / D
                    row["xprime_signal_transfer"] = transfer
                    row["xprime_signal_referred_asd"] = row["xprime_asd"] / abs(
                        transfer
                    )
                    row["xopt_to_signal_referred_xprime"] = (
                        row["x_asd"] / row["xprime_signal_referred_asd"]
                    )
                spectra[key] = np.vstack([spectra[key], prime[:, 0, 0].real, predicted])
            measured.append(row)
    result = pd.DataFrame(measured)
    result.to_csv(output / "signal-noise.csv", index=False)
    np.savez_compressed(output / "spectra.npz", frequency=frequency, **spectra)
    np.savez_compressed(output / "trajectories.npz", **trajectories)
    return result, spectra, frequency


def response_figure(table):
    """Compare the independent response direction and signal-referred noise."""
    response_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Skydip response versus 145609 trajectory",
            "Singlets: signal transfer and coordinate noise",
        ),
    )
    subset = table[table.population == "active_suitable"]
    for family, symbol in (("Affine", "circle"), ("Delay", "diamond")):
        part = subset[subset.family == family]
        response_fig.add_trace(
            go.Scatter(
                x=part.beta_at_probe,
                y=part.beta_1s,
                mode="markers",
                name=family,
                marker={"symbol": symbol, "size": 8, "color": FAMILY_COLORS[family]},
                customdata=part[
                    ["apt_uid", "group_id", "resonator_id", "correlation_1s"]
                ],
                hovertemplate="APT %{customdata[0]} / group %{customdata[1]}, "
                "component %{customdata[2]}<br>Skydip β=%{x:.4g}"
                "<br>145609 1-s slope=%{y:.4g}"
                "<br>Correlation=%{customdata[3]:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        singlets = part[part.multiplicity == 1]
        response_fig.add_trace(
            go.Scatter(
                x=singlets.coordinate_asd_ratio,
                y=singlets.xopt_to_signal_referred_xprime,
                mode="markers",
                name=family,
                showlegend=False,
                marker={"symbol": symbol, "size": 8, "color": FAMILY_COLORS[family]},
                customdata=singlets[["apt_uid", "a_nl", "xprime_signal_transfer"]],
                hovertemplate="APT %{customdata[0]}<br>a=%{customdata[1]:.3g}"
                "<br>x-prime signal transfer=%{customdata[2]:.3g}"
                "<br>Raw ASD ratio=%{x:.3g}<br>Signal-referred ratio=%{y:.3g}"
                "<extra></extra>",
            ),
            row=1,
            col=2,
        )
    response_fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=0.06,
        y1=0.06,
        line={"dash": "dash", "color": "#868e96"},
        row=1,
        col=1,
    )
    response_fig.add_hline(y=1, line_dash="dash", row=1, col=2)
    response_fig.update_xaxes(title_text="Individual Skydip β = Rᵣ/Rₓ", row=1, col=1)
    response_fig.update_yaxes(
        title_text="Measured Δr / Δx_opt, 1-s averages", row=1, col=1
    )
    response_fig.update_xaxes(
        title_text="Measured ASD(x_opt) / ASD(x′)",  # noqa: RUF001 -- detuning prime.
        row=1,
        col=2,
    )
    response_fig.update_yaxes(
        title_text="ASD(x_opt) / [ASD(x′) / |transfer|]",  # noqa: RUF001 -- detuning prime.
        row=1,
        col=2,
    )
    # Show the central cloud without losing the large measured slope. The full
    # range is one explicit action away and all values remain in the CSV.
    response_fig.update_yaxes(range=[-0.06, 0.1], row=1, col=1)
    response_fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.18,
                "buttons": [
                    {
                        "label": "Central slopes",
                        "method": "relayout",
                        "args": [
                            {"yaxis.range": [-0.06, 0.1], "yaxis.autorange": False}
                        ],
                    },
                    {
                        "label": "All slopes",
                        "method": "relayout",
                        "args": [{"yaxis.autorange": True}],
                    },
                ],
            }
        ]
    )
    return response_fig


def noise_figure(table, spectra, frequency):
    """Keep readout failures and response populations apart in spectral summaries."""
    noise_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Affine: singlets",
            "Affine: multiplets",
            "Delay: singlets",
            "Delay: multiplets",
        ),
    )
    for i, (family, _, _) in enumerate(FAMILIES, 1):
        for j, multiple in ((1, False), (2, True)):
            for category, label in LABELS.items():
                part = table[
                    (table.family == family)
                    & (table.population == category)
                    & ((table.multiplicity > 1) == multiple)
                    & (table.psd_status == "measured")
                ]
                values = [
                    np.sqrt(spectra[f"{family}-{r.group_id}-{r.resonator_id}"][0])
                    * r.f_probe_hz
                    for r in part.itertuples()
                ]
                if not values:
                    continue
                noise_fig.add_trace(
                    go.Scatter(
                        x=frequency[1:],
                        y=np.median(values, axis=0)[1:],
                        name=label,
                        hovertemplate=f"{label}, n={len(values)}"
                        "<br>%{x:.3g} Hz<br>%{y:.3g} Hz/√Hz<extra></extra>",
                        legendgroup=category,
                        showlegend=(i == 1 and j == 1),
                        line={"color": COLORS[category]},
                    ),
                    row=i,
                    col=j,
                )
    noise_fig.update_xaxes(type="log", title_text="Temporal frequency [Hz]")
    noise_fig.update_yaxes(type="log", title_text="Median frequency ASD [Hz/√Hz]")
    return noise_fig


def render(output, table, spectra, frequency, *, evidence):
    """Render separated populations, matched model families and optical transfer."""
    associations, sensitivity = evidence
    summary = (
        table.groupby(["family", "population"], sort=False)
        .agg(
            Fitted=("group_id", "size"),
            Complete=("psd_status", lambda s: int((s == "measured").sum())),
            Suitable=("readout_suitable", "sum"),
            Median_frequency_ASD_Hz=("frequency_asd", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    subset = table[table.population == "active_suitable"]
    response_fig = response_figure(table)
    noise_fig = noise_figure(table, spectra, frequency)
    # Duplicate identity rows are a contract failure, never silently averaged.
    matched = subset.pivot(  # noqa: PD010 -- duplicate identities must raise.
        index="apt_uid", columns="family", values="frequency_asd"
    ).dropna()
    matched.to_csv(output / "matched-families.csv")
    compare_rows = []
    for family, part in subset.groupby("family"):
        singlets = part[part.multiplicity == 1]
        compare_rows.append(
            {
                "Model": family,
                "Qualified": len(part),
                "Singlets": len(singlets),
                "Multiplets": int((part.multiplicity > 1).sum()),
                "Median β, Skydip at probe": part.beta_at_probe.median(),
                "Median β, 145609 1 s": part.beta_1s.median(),
                "Median β, 145609 8 s (7 windows)": part.beta_8s.median(),
                "Median coordinate ASD ratio": singlets.coordinate_asd_ratio.median(),
                "Median signal-referred ratio": (
                    singlets.xopt_to_signal_referred_xprime.median()
                ),
                "Held-out two-coordinate / x_opt ASD": part.projected_to_x_asd.median(),
            }
        )
    comparison = pd.DataFrame(compare_rows)
    comparison.to_csv(output / "comparison.csv", index=False)
    for fig, height in ((response_fig, 540), (noise_fig, 800)):
        fig.update_layout(
            template="plotly_white",
            height=height,
            font={"family": "system-ui", "size": 15},
            margin={"b": 100},
            legend={"orientation": "h", "y": -0.25},
        )
        fig.update_annotations(font_size=15)

    def plot(fig):
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True, "displaylogo": False},
        )

    def html(frame):
        return frame.to_html(
            index=False, border=0, float_format=lambda value: f"{value:.4g}"
        )

    view = summary.rename(
        columns={
            "population": "Population",
            "family": "Model",
            "Suitable": "Readout checks pass",
            "Median_frequency_ASD_Hz": "Median frequency ASD [Hz/√Hz], 5-20 Hz",
        }
    )
    view["Population"] = view.Population.map(LABELS)
    template = Path(__file__).with_suffix(".html").read_text()
    replacements = {
        "{{SUMMARY}}": html(view),
        "{{COMPARISON}}": comparison.set_index("Model").T.to_html(
            border=0, float_format=lambda value: f"{value:.4g}"
        ),
        "{{RESPONSE_FIGURE}}": plot(response_fig),
        "{{NOISE_FIGURE}}": plot(noise_fig),
        "{{SENSITIVITY}}": html(sensitivity),
        "{{MATCHED_COUNT}}": str(len(matched)),
        "{{MATCHED_RATIO}}": f"{np.median(matched.Delay / matched.Affine):.5f}"
        if len(matched)
        else "unavailable",
        "{{ASSOCIATIONS}}": str(
            int(
                (
                    (associations.block == 1)
                    & (associations.association == "supported")
                ).sum()
            )
        ),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    (output / "index.html").write_text(template)
    return comparison


def main():
    """Run evidence matching, qualification and measured comparison locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("evidence_root", type=Path)
    args = parser.parse_args()
    logger.remove()
    root = args.root.resolve()
    output = root / "active-response"
    output.mkdir(exist_ok=True)
    sources = read_sources(args.evidence_root.resolve(), output)
    edges = match_evidence(root, sources, output)
    associations = consensus(edges)
    associations.to_csv(output / "associations.csv", index=False)
    table = qualify(root, sources, associations, output)
    sensitivity = []
    for gate in (100_000, 150_000, 200_000):
        matched = consensus(edges, gate)
        allowed = set(
            matched.loc[
                (matched.block == 1) & (matched.association == "supported"),
                "finding_id",
            ]
        )
        for threshold in (0.02, 0.05, 0.10):
            for family, _, _ in FAMILIES:
                part = table[
                    (table.family == family)
                    & table.resonator_id.isin(allowed)
                    & table.population.isin(["active_suitable", "active_poor"])
                ]
                count = sum(
                    not suitability(row, residual_max=threshold)
                    for row in part.to_dict("records")
                )
                sensitivity.append(
                    {
                        "Model": family,
                        "Match gate [kHz]": gate / 1000,
                        "Residual / sweep variation maximum": threshold,
                        "Qualified": count,
                    }
                )
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(output / "selection-sensitivity.csv", index=False)
    measured, spectra, frequency = compare(root, table, output)
    cases = runpy.run_path(str(Path(__file__).with_name("network_active_cases.py")))
    cases["render_cases"](root, output, measured, sources, spectra, frequency)
    summary = render(
        output,
        measured,
        spectra,
        frequency,
        evidence=(associations, sensitivity),
    )
    (output / "method.json").write_text(
        json.dumps(
            {
                "match_gate_hz": MATCH_GATE_HZ,
                "match_min_scans": MATCH_MIN_SCANS,
                "match_baseline_scans": SCANS,
                "residual_to_sweep_variation_max": RESIDUAL_MAX,
                "source_observation": 142440,
                "target_observation": 145609,
                "calibration_block": 1,
                "scope": "Skydip-supported active comparison; no roster promotion, "
                "absolute optical calibration or hardware-optimum claim",
                "signal": "Own Skydip beta; assumed transferable to the 145609 "
                "local fixed-b calibration",
                "noise": "Uncleaned total fluctuations; no gap filling; "
                "Welch 1024/512 Hann; band 5-20 Hz",
                "projection": "Weights trained on first half; "
                "noise evaluated on second half",
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))  # noqa: T201


if __name__ == "__main__":
    main()
