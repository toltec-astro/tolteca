#!/usr/bin/env python
"""Produce post-solving diagnostics from saved 145609 states and fitted groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from astropy.table import Table
from netCDF4 import Dataset
from tollan.utils.log import logger

from tolteca_kids.model_fit_quality import ModelFitQualityConfig, model_fit_quality
from tolteca_kids.model_joint import joint_models, joint_solver
from tolteca_kids.product import open_sweep_product
from tolteca_kidsproc.analysis.model_fit import KidsModelView
from tolteca_kidsproc.timestream_diagnostics import coordinate_spectra, spectral_matrix

HERE = Path(__file__).parent
RESPONSE = runpy.run_path(str(HERE / "network_response.py"))
FAMILIES = [
    ("Affine", "coverage", "0000-models"),
    ("Delay", "coverage-delay", "0001-models"),
]
BAND = (5, 20)
COMPONENTS_PER_FAMILY = 625


def dump(path, value):
    """Write JSON with missing numerical values explicitly represented as null."""

    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(v) for v in item]
        if isinstance(item, (np.integer, np.bool_)):
            return item.item()
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        return item

    path.write_text(json.dumps(clean(value), allow_nan=False) + "\n")


def groups_from_fit(root, folder, realization):
    """Use the saved fit and selected probe routes, including singlet groups."""
    index = (
        root
        / "dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc"
        / realization
        / "index.yaml"
    )
    table = Table.read(root / folder / "block1/kids_model.ecsv", format="ascii.ecsv")
    props = (
        pd.read_csv(
            root / folder / "block1/channels.csv",
            dtype={"group_id": str},
            float_precision="round_trip",
        )
        .fillna({"group_id": ""})
        .set_index("chan")
        .to_xarray()
    )
    props["fp"] = ("chan", np.asarray(table["f_in"]))
    with open_sweep_product(index) as tree:
        block = tree.toltec_kids.blocks[1]
        namespace = (
            block.tree["model_fit/tolteca_kidsproc.analysis.model_fit"]
            .copy(deep=False)
            .load()
        )
        namespace["chan_prop"] = xr.DataTree(dataset=props)
        view = KidsModelView.from_namespace(namespace)
        config = ModelFitQualityConfig.model_validate(
            block.tree["model_fit/quality"].attrs["thresholds"]
        )
        payload = joint_models(
            view, model_fit_quality(view, config), include_single=True
        )
    # The diagnostic view cannot quietly substitute another multiplet model.
    assert [g for g in payload["groups"] if len(g["resonator_ids"]) > 1] == table.meta[
        "joint_models"
    ]["groups"]
    return {g["group_id"]: g for g in payload["groups"]}


def component_diagnostics(evaluation, component, probe, rate, row, output):  # noqa: PLR0913, PLR0917 -- Model view, coordinate and persistence context.
    """Persist actual coordinate spectra and local first-order decomposition."""
    evaluated, reference = evaluation
    pair = evaluated.sel(chan=probe, resonator=component)
    loss = evaluated.r.sel(resonator=component).to_numpy()
    xp, xo = pair.x_prime.to_numpy(), pair.x_opt.to_numpy()
    # The same model at the median physical state supplies the linear reference.
    b = float(evaluated.b.sel(resonator=component))
    local = reference.sel(chan=probe, resonator=component)
    D, mu = float(local.D.item()), float(local.mu.item())
    spectra = coordinate_spectra(xp, xo, loss, rate, D=D, mu=mu)
    frequency = spectra.f_psd.to_numpy()
    band = (frequency >= BAND[0]) & (frequency <= BAND[1])
    cross = spectra.cross_real.to_numpy() + 1j * spectra.cross_imag.to_numpy()
    power = cross.diagonal(axis1=1, axis2=2).real
    terms = spectra.linear_terms.to_numpy()

    def mean(values):
        return float(np.mean(values[band]))

    xo_power, xp_power = mean(power[:, 1]), mean(power[:, 0])
    np.testing.assert_allclose(
        spectra.linear_predicted, spectra.linear_measured, rtol=1e-10, atol=1e-28
    )
    np.testing.assert_allclose(
        spectra.exact_reconstruction, power[:, 1], rtol=1e-9, atol=1e-28
    )
    row.update(
        diagnostic_status="complete",
        probe=probe,
        f_probe_hz=float(evaluated.f_probe.sel(chan=probe)),
        b=b,
        D=D,
        mu=mu,
        xprime_asd=np.sqrt(xp_power),
        xopt_asd=np.sqrt(xo_power),
        loss_asd=np.sqrt(mean(power[:, 2])),
        coordinate_asd_ratio=np.sqrt(xo_power / xp_power),
        linear_predicted_asd_ratio=np.sqrt(
            mean(spectra.linear_predicted.to_numpy()) / xp_power
        ),
        linear_remainder_asd_ratio=np.sqrt(
            mean(spectra.linear_remainder.to_numpy()) / xo_power
        ),
        cross_fraction=mean(terms[:, 2]) / xo_power,
        loss_fraction=mean(terms[:, 1]) / xo_power,
        prime_fraction=mean(terms[:, 0]) / xo_power,
        prime_loss_coherence=float(np.mean(spectra.coherence.to_numpy()[band, 0, 2])),
        D_min=float(pair.D.min()),
        D_max=float(pair.D.max()),
        mu_min=float(pair.mu.min()),
        mu_max=float(pair.mu.max()),
        signal_status="independent_response_not_qualified",
    )
    slope = row.get("loss_detuning_slope_at_probe", np.nan)
    if row["population"] == "active_suitable" and np.isfinite(slope):
        transfer = (1 - mu * slope) / D
        row["xprime_signal_transfer"] = transfer
        if abs(transfer) > np.finfo(float).eps:
            row.update(
                signal_status="conditional_skydip_direction",
                xprime_signal_referred_asd=np.sqrt(xp_power) / abs(transfer),
                signal_referred_asd_ratio=np.sqrt(xo_power / xp_power) * abs(transfer),
            )
        else:
            row["signal_status"] = "zero_local_signal_transfer"
    key = f"{row['family']}-{row['group_id']}-{component}"
    np.savez_compressed(
        output / "arrays" / f"{key}.npz",
        x_prime=xp,
        x_opt=xo,
        r=loss,
        frequency=frequency,
        cross_real=cross.real,
        cross_imag=cross.imag,
        linear_terms=terms,
        linear_predicted=spectra.linear_predicted.to_numpy(),
        linear_remainder=spectra.linear_remainder.to_numpy(),
        coherence=spectra.coherence.to_numpy(),
    )
    case = {
        "record": row.copy(),
        "time": np.arange(0, len(xo), 16) / rate,
        "x_prime": xp[::16],
        "x_opt": xo[::16],
        "r": loss[::16],
        "trajectory": RESPONSE["binned_trajectory"](xo, loss, rate),
        "trajectory_prime": RESPONSE["binned_trajectory"](xp, loss, rate),
        "frequency": frequency,
        "power": power,
        "cross_prime_loss_real": cross[:, 0, 2].real,
        "cross_prime_loss_imag": cross[:, 0, 2].imag,
        "coherence": spectra.coherence.to_numpy()[:, 0, 2],
        "linear_terms": terms,
        "linear_predicted": spectra.linear_predicted.to_numpy(),
    }
    dump(output / "cases" / f"{key}.json", case)
    return row


def group_diagnostics(group, members, measured, iq, rate, family, output):  # noqa: PLR0913, PLR0917 -- Saved state and measurement destinations.
    """Evaluate every accepted group jointly; preserve rejected groups in the ledger."""
    route = members[0][0]["native_route"]
    if route == "joint":
        path = measured / "native/stub" / f"group-{group['group_id']}"
        states = [
            np.fromfile(path / f"{k}.bin", dtype="float64").reshape(
                -1, len(group["resonator_ids"])
            )
            for k in ("fr", "r")
        ]
    else:
        base, x, r = members[0]
        states = [(base["f_probe_hz"] / (1 + x))[:, None], r[:, None]]
    solver = joint_solver(group)
    evaluated = solver.evaluate(*states)
    reference = solver.evaluate(*[np.median(s, axis=0, keepdims=True) for s in states])
    predicted = evaluated.I.to_numpy() + 1j * evaluated.Q.to_numpy()
    observed = iq[:, group["channels"]]
    residual = predicted - observed
    records = []
    for k, channel in enumerate(group["channels"]):
        records.append(
            {
                "family": family,
                "group_id": group["group_id"],
                "channel": channel,
                "probes": len(group["channels"]),
                "detectors": len(group["resonator_ids"]),
                "rms_iq": float(np.sqrt(np.mean(abs(residual[:, k]) ** 2))),
                "rms_relative_gain": float(
                    np.sqrt(np.mean(abs(residual[:, k] / solver.gain[k]) ** 2))
                ),
                "reference_rank": solver.diagnostics["rank"],
                "reference_condition": solver.diagnostics["condition"],
                "noise_source": group["noise_source"],
            }
        )
    # Cross-detector spectra keep jointly solved components correlated.
    if len(members) > 1:
        values = np.column_stack([x for _, x, _ in members])
        frequency, cross = spectral_matrix(values, rate)
        np.savez_compressed(
            output / "arrays" / f"{family}-group-{group['group_id']}-cross.npz",
            frequency=frequency,
            cross_real=cross.real,
            cross_imag=cross.imag,
            resonator_ids=[base["resonator_id"] for base, _, _ in members],
        )
    return (evaluated, reference), records


def fingerprints(paths):
    """Pin exact model reports, solved states and diagnostic source code."""
    return {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(paths))
    }


def validate_measurement(measured, request, rate):
    """Check the saved samples against acquisition counters and unmodified I/Q."""
    prepared = json.loads((measured / "prepared.json").read_text())
    source = Path(prepared["data_item"]["filepath"])
    shape = request["samples"], request["channels"]
    arrays = [
        np.fromfile(request[k], dtype="float64").reshape(shape) for k in ("I", "Q")
    ]
    with Dataset(source) as raw:
        ticks = raw["Data.Toltec.Ts"]
        names = [part.strip().split()[0] for part in ticks.long_name.split(",")]
        counts = np.asarray(ticks[:, names.index("PacketCount")], dtype=np.uint32)
        clocks = np.asarray(ticks[:, names.index("ClockCount")], dtype=np.uint32)
        assert len(counts) == request["samples"]
        assert np.all(np.diff(counts) == 1), "Acquisition packet gaps"
        assert np.all(np.diff(clocks) == np.diff(clocks)[0]), "Irregular sample clock"
        assert float(raw["Header.Toltec.SampleFreq"][...]) == rate
        for field, saved in zip(("Is", "Qs"), arrays, strict=True):
            np.testing.assert_array_equal(raw[f"Data.Toltec.{field}"][:], saved)
    return (arrays[0] + 1j * arrays[1]) / request["normalization"], {
        "source": str(source),
        "samples": len(counts),
        "sample_rate_hz": rate,
        "packet_gaps": 0,
        "clock_step": int(np.diff(clocks)[0]),
        "saved_iq_matches_raw": True,
    }


def source_paths(root):
    """Collect immutable measurement, solved-state, model and code inputs."""
    paths = [
        root / "group-response/signal-noise.csv",
        root / "group-response/verification.json",
    ]
    paths.extend(HERE.glob("network_postsolve*"))
    paths.extend(
        [
            HERE / "network_response.py",
            HERE.parents[1] / "src/tolteca_kidsproc/joint_timestream.py",
            HERE.parents[1] / "src/tolteca_kidsproc/timestream_diagnostics.py",
            HERE.parents[1] / "src/tolteca_kids/model_joint.py",
        ]
    )
    for _, folder, _ in FAMILIES:
        paths.extend((root / folder).glob("block1/*.ecsv"))
        paths.extend((root / folder / "measured/native/stub").rglob("*.bin"))
        paths.extend(
            [
                root / folder / "measured/request.json",
                root / folder / "block1/channels.csv",
            ]
        )
        measured = root / folder / "measured"
        request = json.loads((measured / "request.json").read_text())
        paths.extend([Path(request[k]) for k in ("I", "Q")])
        prepared = json.loads((measured / "prepared.json").read_text())
        paths.append(Path(prepared["data_item"]["filepath"]))
        paths.extend(
            [measured / "output-quality-method.json", measured / "prepared.json"]
        )
    return paths


def run(root, output):
    """Build diagnostics without altering fits, solved states or selection policy."""
    output.mkdir(parents=True, exist_ok=True)
    for sub in ("arrays", "cases"):
        (output / sub).mkdir(exist_ok=True)
    ledger_path = root / "group-response/signal-noise.csv"
    ledger = pd.read_csv(ledger_path, dtype={"group_id": str, "resonator_id": str})
    paths = source_paths(root)
    before = fingerprints(paths)
    rows, residuals, measurements = [], [], {}
    for family, folder, realization in FAMILIES:
        groups = groups_from_fit(root, folder, realization)
        measured = root / folder / "measured"
        request = json.loads((measured / "request.json").read_text())
        rate = json.loads((measured / "output-quality-method.json").read_text())[
            "sample_rate_hz"
        ]
        iq, measurements[family] = validate_measurement(measured, request, rate)
        source = ledger[ledger.family == family].set_index(["group_id", "resonator_id"])
        complete = {}
        for base, x, r in RESPONSE["component_records"](measured):
            row = source.loc[base["group_id"], base["resonator_id"]].to_dict()
            row.update(
                group_id=base["group_id"],
                resonator_id=base["resonator_id"],
                family=family,
                diagnostic_status="no_complete_valid_estimate",
            )
            if x is None:
                rows.append(row)
            else:
                complete.setdefault(base["group_id"], []).append((base, x, r))
        for gid, members in complete.items():
            evaluation, checks = group_diagnostics(
                groups[gid], members, measured, iq, rate, family, output
            )
            residuals.extend(checks)
            for base, x, _r in members:
                rid = base["resonator_id"]
                probe = base["probe"]
                np.testing.assert_allclose(
                    evaluation[0].x_opt.sel(chan=probe, resonator=rid),
                    x,
                    rtol=0,
                    atol=1e-12,
                )
                row = source.loc[gid, rid].to_dict()
                row.update(group_id=gid, resonator_id=rid, family=family)
                rows.append(
                    component_diagnostics(evaluation, rid, probe, rate, row, output)
                )
        logger.info("{}: inspected {} complete groups", family, len(complete))
    result = pd.DataFrame(rows)
    result.to_csv(output / "components.csv", index=False)
    pd.DataFrame(residuals).to_csv(output / "iq-residuals.csv", index=False)
    assert len(result) == len(ledger) == COMPONENTS_PER_FAMILY * len(FAMILIES)
    joined = result.set_index(["family", "group_id", "resonator_id"]).sort_index()
    original = ledger.set_index(["family", "group_id", "resonator_id"]).sort_index()
    for key in [
        "population",
        "readout_suitable",
        "fit_adequacy",
        "numerical_validity",
        "selection_policy_limits",
    ]:
        pd.testing.assert_series_equal(joined[key], original[key])
    assert fingerprints(paths) == before
    dump(output / "sources.json", before)
    dump(output / "measurement.json", measurements)
    dump(
        output / "method.json",
        {
            "root": str(root),
            "band_hz": BAND,
            "nperseg": 1024,
            "noverlap": 512,
            "window": "hann",
            "cross_convention": "conj(first)*second",
            "coordinate_order": ["x_prime", "x_opt", "r", "correction"],
            "derivatives": "same group model at median fr and r; fixed calibrated b",
            "missing_samples": "no spectra, no gap filling",
            "selection": "unchanged",
            "sensitivity": (
                "conditional on independent Skydip direction; no absolute mapping speed"
            ),
        },
    )
    runpy.run_path(str(HERE / "network_postsolve_report.py"))["render"](result, output)
    dump(
        output / "verification.json",
        {
            "input_files_unchanged": len(before),
            "components": len(result),
            "complete": result.groupby("family")
            .diagnostic_status.apply(lambda s: int((s == "complete").sum()))
            .to_dict(),
            "selection_unchanged": True,
            "spectral_linear_identity_checked": True,
            "spectral_exact_correction_identity_checked": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
