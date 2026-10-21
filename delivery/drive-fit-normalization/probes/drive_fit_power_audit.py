"""Audit saved drive-fit/commit power using the v2 model and Tlaloc normalization.

This diagnostic does not change fit values or instrument settings. Supply the
DAC calibration explicitly; output powers refer to the drive-attenuator output,
not the detector input. Missing tone phases are never replaced with measured
zeros. Seeded phase trials are sensitivity scenarios, not a reconstruction.
"""

from __future__ import annotations

# ruff: noqa: INP001 -- Standalone scientific audit probe.
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlparse

import astropy.units as u
import netCDF4
import numpy as np
from astropy.table import Table
from plotly.subplots import make_subplots

from tolteca_kids.drive_fit import DriveFitStep, open_drive_fit_product
from tolteca_kids.drive_fit_commit import (
    DriveFitCommitStep,
    open_drive_fit_commit_product,
)

FFT_SIZE = 2**21
SAMPLE_RATE = 512e6


def load_power_model(source):
    """Execute the original v2 per-tone power property without constructing I/Q data."""
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RoachTonePower"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "P_dac_dbm_per_tone"
    )
    method.decorator_list = []
    namespace: dict[str, Any] = {"np": np}
    exec(  # noqa: S102 -- inspected original numerical method.
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    def calculate(amps, total):
        return namespace[method.name](
            SimpleNamespace(tone_amps=np.asarray(amps), P_dac_dbm_total=total)
        )

    return calculate


def headers(source):
    """Read native tone headers, preserving absent phases as absent."""
    path = (
        Path(unquote(urlparse(source).path))
        if source.startswith("file:")
        else Path(source)
    )
    with netCDF4.Dataset(path) as ds:
        variables = ds.variables
        amp_key = next(
            k
            for k in ("Header.Toltec.ToneAmps", "Header.Toltec.ToneAmp")
            if k in variables
        )
        phase_key = "Header.Toltec.TonePhase"
        phases = None
        if phase_key in variables:
            phase = variables[phase_key]
            phases = u.Quantity(
                np.asarray(phase[:]).reshape(-1), u.Unit(getattr(phase, "units", "rad"))
            ).to_value(u.rad)
        return {
            "path": str(path),
            "freq": np.asarray(variables["Header.Toltec.ToneFreq"][:]).reshape(-1),
            "amp": np.asarray(variables[amp_key][:]).reshape(-1),
            "phase": phases,
        }


def fft_bins(frequency):
    """Apply Tlaloc's frequency rounding and signed-bin wrapping."""

    def c_round(values):
        return np.copysign(np.floor(np.abs(values) + 0.5), values)

    resolution = 2 * SAMPLE_RATE / FFT_SIZE
    quantized = c_round(np.asarray(frequency) / resolution) * resolution
    return c_round(quantized / SAMPLE_RATE * FFT_SIZE).astype(int) % FFT_SIZE


def bin_owners(bins):
    """Return the last row writing each DAC bin, as in Tlaloc's assignment loop."""
    last = {int(value): index for index, value in enumerate(bins)}
    return np.array([last[int(value)] for value in bins])


def emitted_powers(calculate, bins, amplitudes, total):
    """Distribute total power over unique emitted bins and map back to channels."""
    owners = bin_owners(bins)
    active = np.unique(owners)
    powers = calculate(amplitudes[active], total)
    return powers[np.searchsorted(active, owners)]


def waveform_peak(bins, amplitudes, phases):
    """Return the shared complex waveform peak used by Tlaloc's dac_comb."""
    active = np.unique(bin_owners(bins))
    spectrum = np.zeros(FFT_SIZE, dtype=complex)
    spectrum[bins[active]] = amplitudes[active] * np.exp(1j * phases[active])
    return np.max(np.abs(np.fft.ifft(spectrum)))


def load_legacy_ampcor(source):
    """Load the historical recipe calculation independently of corrected v3."""
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DriveFitCommit"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_ampcor"
    )
    namespace = {
        "np": np,
        "logger": SimpleNamespace(debug=lambda *_: None, info=lambda *_: None),
        "pformat_mask": str,
    }
    exec(  # noqa: S102 -- Execute the original read-only recipe method.
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    def calculate(attenuation, flags, amplitudes, **options):
        cfg = SimpleNamespace(
            adrv_max=options["adrv_max"],
            adrv_fill_perc=options["fill_percentile"],
            adrv_ref_perc=options["reference_percentile"],
            adrv_ref_step=options["reference_step"],
        )
        table = Table(
            {
                "adrv_best_adrv": attenuation,
                "flag_adrv": flags,
                "amp_chan_adrv": amplitudes,
            }
        )
        return namespace["compute_ampcor"](SimpleNamespace(config=cfg), table)

    return calculate


def verify_models(calculate, compute_ampcor):
    """Check scale invariance, no-op commit and a demonstrable normalization error."""
    amps = np.array([1.0, 1.0])
    np.testing.assert_allclose(calculate(amps, -12.6), [-12.6 - 10 * np.log10(2)] * 2)
    np.testing.assert_allclose(calculate(amps * 7, -12.6), calculate(amps, -12.6))
    options = {
        "adrv_max": 30,
        "fill_percentile": 50,
        "reference_percentile": 0,
        "reference_step": 0.25,
    }
    unchanged = compute_ampcor([10, 10], [0, 0], amps, **options)
    np.testing.assert_array_equal(unchanged["ampcor"], amps)
    np.testing.assert_equal(unchanged["adrv_ref_total"], 10)
    changed = compute_ampcor([10, 20], [0, 0], amps, **options)
    requested = calculate(amps, -12.6) - [10, 20]
    delivered = calculate(changed["ampcor"], -12.6) - changed["adrv_ref_total"]
    expected = 10 * np.log10(2 / 1.1)
    np.testing.assert_allclose(delivered - requested, expected)
    bins = fft_bins(np.array([100e6, 200e6]))
    phases = np.zeros(2)
    before = waveform_peak(bins, amps, phases)
    after = waveform_peak(bins, changed["ampcor"], phases)
    np.testing.assert_allclose(waveform_peak(bins, amps * 7, phases), before * 7)
    exact_error = 20 * np.log10(before / after)
    np.testing.assert_allclose(exact_error, 20 * np.log10(2 / (1 + np.sqrt(0.1))))
    duplicate_bins = np.array([bins[0], bins[0], bins[1]])
    np.testing.assert_array_equal(bin_owners(duplicate_bins), [1, 1, 2])
    duplicate_amps = np.array([100.0, 1.0, 1.0])
    np.testing.assert_allclose(
        emitted_powers(calculate, duplicate_bins, duplicate_amps, -12.6),
        [-12.6 - 10 * np.log10(2)] * 3,
    )
    np.testing.assert_allclose(
        waveform_peak(duplicate_bins, duplicate_amps, np.zeros(3)),
        before,
    )
    return {
        "fixed_total_error_db": float(expected),
        "known_phase_peak_error_db": float(exact_error),
    }


def audit(args):  # noqa: PLR0915 -- Complete audit and evidence inventory.
    """Compare intended and achieved tone powers without changing the saved commit."""
    args.output.mkdir(parents=True, exist_ok=False)
    calculate = load_power_model(args.power_source)
    counterexample = verify_models(calculate, load_legacy_ampcor(args.recipe))
    with open_drive_fit_product(args.fit) as tree:
        fit = DriveFitStep.view(tree).data.load()
    with open_drive_fit_commit_product(args.commit) as tree:
        commit = DriveFitCommitStep.view(tree).data.load()
    ref_headers = [headers(item["source"]) for item in fit.attrs["inputs"]]
    ref = ref_headers[0]
    target = headers(commit.attrs["target_source"])
    # Total power must be distributed over the FULL comb, never a selected subset.
    np.testing.assert_array_equal(fit.chan, np.arange(len(ref["amp"])))
    np.testing.assert_array_equal(fit.amp_chan, ref["amp"])
    np.testing.assert_array_equal(commit.chan, np.arange(len(target["amp"])))
    for item in ref_headers[1:]:
        np.testing.assert_array_equal(item["amp"], ref["amp"])
        np.testing.assert_array_equal(item["freq"], ref["freq"])
    bins_ref, bins_target = fft_bins(ref["freq"]), fft_bins(target["freq"])
    owners_ref, owners_target = bin_owners(bins_ref), bin_owners(bins_target)
    active = np.unique(owners_target)
    before_rows = calculate(ref["amp"], args.dac_total_dbm)
    after_rows = calculate(commit.amp_best.values, args.dac_total_dbm)
    before = emitted_powers(calculate, bins_ref, ref["amp"], args.dac_total_dbm)
    after = emitted_powers(
        calculate, bins_target, commit.amp_best.values, args.dac_total_dbm
    )
    match = commit.idx_ref.values
    best = before - fit.adrv_best.values
    requested = before[match] - commit.adrv_best.values
    global_atten = commit.attrs["atten_drive_global"]
    delivered = after - global_atten
    error = delivered - requested
    required = after - requested
    # First check the recipe's independent-row assumption, then quantify bins.
    row_required = after_rows - before_rows[match] + commit.adrv_best.values
    np.testing.assert_allclose(row_required, row_required[0], rtol=0, atol=1e-12)
    exact_atten = float(np.median(required))
    step = commit.attrs["config"]["adrv_ref_step"]
    ceiling_atten = float(np.ceil(exact_atten / step) * step)
    accepted = ~commit.amp_flag.values
    clipped = accepted & (commit.adrv_best.values > commit.adrv_best_adrv.values)
    table = Table(
        {
            "chan": commit.chan.values,
            "chan_reference": fit.chan.values[match],
            "dac_bin": bins_target,
            "dac_owner_chan": owners_target,
            "overwritten": owners_target != commit.chan.values,
            "frequency_mhz": commit.f_chan.values / 1e6,
            "accepted": accepted,
            "clipped": clipped,
            "amp_reference": ref["amp"][match],
            "amp_commit": commit.amp_best.values,
            "best_power_dbm": best[match],
            "policy_power_dbm": requested,
            "commit_power_dbm": delivered,
            "commit_minus_policy_db": error,
            "policy_minus_best_db": requested - best[match],
            "commit_minus_best_db": delivered - best[match],
        }
    )
    table.meta.update(
        {
            "power_plane": "drive attenuator output",
            "model": "v2 fixed DAC total power, distributed over Tlaloc unique bins",
            "dac_total_dbm_assumed": args.dac_total_dbm,
        }
    )
    table.write(args.output / "tone_power_audit.ecsv")
    # The code below computes calibration-independent changes in DAC normalization.
    # The base per-tone factor includes the normalized ampcor and actual saved A.
    base = (
        20
        * np.log10(
            commit.amp_best.values[owners_target] / ref["amp"][owners_ref[match]]
        )
        + commit.adrv_best.values
        - global_atten
    )
    base_median = float(np.median(base[active]))
    np.testing.assert_array_equal(target["freq"], ref["freq"])
    rng = np.random.default_rng(args.seed)
    same, regenerated = [], []
    for _ in range(args.trials):
        phases = rng.uniform(0, 2 * np.pi, len(ref["amp"]))
        peak_ref = waveform_peak(bins_ref, ref["amp"], phases)
        peak_same = waveform_peak(bins_target, commit.amp_best.values, phases)
        peak_new = waveform_peak(
            bins_target,
            commit.amp_best.values,
            rng.uniform(0, 2 * np.pi, len(target["amp"])),
        )
        same.append(float(base_median + 20 * np.log10(peak_ref / peak_same)))
        regenerated.append(float(base_median + 20 * np.log10(peak_ref / peak_new)))
    exact_peak_error = None
    if ref["phase"] is not None and target["phase"] is not None:
        if not all(
            item["phase"] is not None and np.array_equal(item["phase"], ref["phase"])
            for item in ref_headers
        ):
            msg = "Reference phases changed during the fit sequence"
            raise ValueError(msg)
        exact_peak_error = float(
            base_median
            + 20
            * np.log10(
                waveform_peak(bins_ref, ref["amp"], ref["phase"])
                / waveform_peak(bins_target, commit.amp_best.values, target["phase"])
            )
        )
    summary = {
        "fit_index": str(args.fit),
        "commit_index": str(args.commit),
        "dac_total_dbm_assumed": args.dac_total_dbm,
        "power_plane": "drive attenuator output; no cryogenic loss applied",
        "channels": len(commit.chan),
        "unique_dac_bins": len(active),
        "overwritten_amplitude_rows": int(np.sum(owners_target != commit.chan.values)),
        "collision_groups": [
            np.where(bins_target == b)[0].tolist()
            for b in np.unique(bins_target)
            if np.sum(bins_target == b) > 1
        ],
        "independent_row_error_db": float(row_required[0] - global_atten),
        "fixed_total_error_range_db": [float(np.min(error)), float(np.max(error))],
        "required_attenuation_range_db": [
            float(np.min(required)),
            float(np.max(required)),
        ],
        "accepted": int(accepted.sum()),
        "filled": int((~accepted).sum()),
        "clipped_accepted": int(clipped.sum()),
        "attenuation_current_db": global_atten,
        "attenuation_required_fixed_total_db": exact_atten,
        "attenuation_ceiling_fixed_total_db": ceiling_atten,
        "fixed_total_error_db": float(np.median(error)),
        "max_policy_underdrive_db": float(np.max((best[match] - requested)[accepted])),
        "reference_phase_headers_present": sum(
            item["phase"] is not None for item in ref_headers
        ),
        "target_phase_header_present": target["phase"] is not None,
        "exact_waveform_error_db": exact_peak_error,
        "phase_sensitivity": {
            "trials": args.trials,
            "seed": args.seed,
            "same_phases_error_db": same,
            "regenerated_phases_error_db": regenerated,
            "meaning": (
                "Median error of controlling DAC-bin rows. "
                "Synthetic sensitivity scenarios; "
                "not measured errors or confidence intervals"
            ),
        },
        "controlled_two_tone_counterexample": counterexample,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.power_source, args.tlaloc_source)
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure = make_figure(table, summary)
    figure.write_html(args.output / "drive_fit_power_audit.html", include_plotlyjs=True)
    print(  # noqa: T201 -- command line report.
        json.dumps(
            {k: v for k, v in summary.items() if k != "phase_sensitivity"}, indent=2
        )
    )
    for name, samples in (("same phases", same), ("regenerated phases", regenerated)):
        print(name, "min/median/max dB", np.percentile(samples, [0, 50, 100]))  # noqa: T201


def make_figure(table, summary):
    """Show power estimates, policy changes, commit error and phase sensitivity."""
    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.14,
        vertical_spacing=0.20,
        subplot_titles=(
            "Estimated best power per tone",
            "Commit power versus intended power",
            "Power error and deliberate underdrive",
            "Missing-phase sensitivity · synthetic trials",
        ),
    )
    for mask, label, color in (
        (table["accepted"] & ~table["clipped"], "Accepted", "#2166ac"),
        (table["clipped"], "Accepted, percentile limited", "#d68b14"),
        (~table["accepted"], "Flagged / filled", "#949494"),
    ):
        common = {
            "mode": "markers",
            "marker": {"size": 5, "color": color},
            "name": label,
            "legendgroup": label,
            "customdata": table["chan"][mask],
            "hovertemplate": (
                "Channel %{customdata}<br>x=%{x:.4f}<br>y=%{y:.4f}"
                "<extra>%{fullData.name}</extra>"
            ),
        }
        fig.add_scatter(
            x=table["frequency_mhz"][mask],
            y=table["best_power_dbm"][mask],
            **common,
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=table["policy_power_dbm"][mask],
            y=table["commit_power_dbm"][mask],
            showlegend=False,
            **common,
            row=1,
            col=2,
        )
        fig.add_scatter(
            x=table["frequency_mhz"][mask],
            y=table["commit_minus_best_db"][mask],
            showlegend=False,
            **common,
            row=2,
            col=1,
        )
    overwritten = table["overwritten"]
    fig.add_scatter(
        x=table["frequency_mhz"][overwritten],
        y=table["commit_minus_best_db"][overwritten],
        mode="markers",
        marker={"symbol": "circle-open", "size": 10, "color": "#b2182b"},
        name="Overwritten DAC row",
        row=2,
        col=1,
    )
    span = [min(table["policy_power_dbm"]), max(table["policy_power_dbm"])]
    fig.add_scatter(
        x=span,
        y=span,
        mode="lines",
        line={"color": "#555", "dash": "dash"},
        name="Exact power",
        showlegend=False,
        row=1,
        col=2,
    )
    fig.add_hline(
        y=summary["fixed_total_error_db"],
        line_dash="dash",
        line_color="#555",
        row=2,
        col=1,
    )
    phase = summary["phase_sensitivity"]
    for name, label, color in (
        ("same_phases_error_db", "Phases retained", "#2166ac"),
        ("regenerated_phases_error_db", "Phases regenerated", "#d68b14"),
    ):
        fig.add_histogram(
            x=phase[name], name=label, opacity=0.65, marker_color=color, row=2, col=2
        )
    fig.add_vline(x=0, line_dash="dash", line_color="#555", row=2, col=2)
    for row, col, x, y in (
        (1, 1, "Frequency (MHz)", "Estimated power (dBm)"),
        (1, 2, "Intended after fill / cutoff (dBm)", "Predicted commit power (dBm)"),
        (2, 1, "Frequency (MHz)", "Commit - original best (dB)"),
        (2, 2, "Median commit - policy target (dB)", "Scenario count"),
    ):
        fig.update_xaxes(title_text=x, row=row, col=col)
        fig.update_yaxes(title_text=y, row=row, col=col)
    fig.update_layout(
        height=1000,
        template="plotly_white",
        barmode="overlay",
        margin={"t": 185, "b": 120, "l": 85, "r": 40},
        legend={"orientation": "h", "x": 0, "y": -0.13},
        title={
            "text": "Drive fit / commit · power audit"
            f"<br><sup>Assumed DAC total: {summary['dac_total_dbm_assumed']:g} dBm"
            " · plane: after drive attenuator"
            f"<br>Current attenuation: {summary['attenuation_current_db']:g} dB"
            f" · median model error: {summary['fixed_total_error_db']:+.3f} dB"
            f" · {summary['unique_dac_bins']} DAC bins / {summary['channels']} rows"
            "<br>Absolute scale is a model estimate; "
            "phase trials are not measured hardware errors.</sup>",
            "x": 0.06,
            "y": 0.98,
        },
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--power-source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--tlaloc-source", type=Path, required=True)
    parser.add_argument("--dac-total-dbm", type=float, required=True)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())
