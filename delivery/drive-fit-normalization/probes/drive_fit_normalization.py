"""Compare baseline and corrected products using the original RoachTonePower.

Run with --before and --after pointing to workflow products.json manifests,
--power-source pointing to v2 roach_tone_power.py and --output to a new folder.
"""

from __future__ import annotations

# ruff: noqa: INP001, T201 -- Standalone scientific evidence probe.
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
from astropy.table import Table
from drive_fit_power_audit import (
    bin_owners,
    emitted_powers,
    fft_bins,
    headers,
    load_power_model,
    waveform_peak,
)

from tolteca_kids.drive_fit import DriveFitStep, open_drive_fit_product
from tolteca_kids.drive_fit_commit import (
    DriveFitCommitStep,
    open_drive_fit_commit_product,
)

mpl.use("Agg")
import matplotlib.pyplot as plt


def compare(args):  # noqa: PLR0915 -- One report per network with independent forward check.
    """Assert unchanged scientific fits and independently predict delivered power."""
    args.output.mkdir(parents=True, exist_ok=True)
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    power = load_power_model(args.power_source)
    records = []
    for network in args.networks:
        interface = f"toltec{network}"
        with (
            open_drive_fit_product(before["fit"], interface=interface) as tree,
            open_drive_fit_product(after["fit"], interface=interface) as tree_new,
        ):
            old_fit = DriveFitStep.view(tree).data.load()
            new_fit = DriveFitStep.view(tree_new).data.load()
            for name in old_fit.data_vars:
                np.testing.assert_array_equal(
                    old_fit[name].values, new_fit[name].values
                )
        with (
            open_drive_fit_commit_product(
                before["commit"], interface=interface
            ) as tree,
            open_drive_fit_commit_product(
                after["commit"], interface=interface
            ) as tree_new,
        ):
            old = DriveFitCommitStep.view(tree).data.load()
            new = DriveFitCommitStep.view(tree_new).data.load()
        for name in (
            "idx_ref",
            "adrv_best",
            "amp_flag",
            "adrv_ref",
            "dist",
            "dist_shifted",
        ):
            np.testing.assert_array_equal(old[name], new[name])
        ref_headers = [headers(item["source"]) for item in old_fit.attrs["inputs"]]
        reference, target = ref_headers[0], headers(new.attrs["target_source"])
        bins_ref, bins_target = fft_bins(reference["freq"]), fft_bins(target["freq"])
        controlling = np.unique(bin_owners(bins_target))
        # Arbitrary 0 dBm DAC total cancels out of every residual below.
        ref_power = emitted_powers(power, bins_ref, reference["amp"], 0)
        desired = ref_power[new.idx_ref.values] - new.adrv_best.values
        old_achieved = (
            emitted_powers(power, bins_target, old.amp_best.values, 0)
            - old.attrs["atten_drive_global"]
        )
        new_achieved = (
            emitted_powers(power, bins_target, new.amp_best.values, 0)
            - new.attrs["atten_drive_global"]
        )
        error_old, error_new = old_achieved - desired, new_achieved - desired
        np.testing.assert_allclose(new.power_policy_db, desired, rtol=0, atol=2e-13)
        np.testing.assert_allclose(
            new.power_commit_db, new_achieved, rtol=0, atol=2e-13
        )
        np.testing.assert_allclose(new.power_error_db, error_new, rtol=0, atol=2e-13)
        step = new.attrs["config"]["adrv_ref_step"]
        tolerance = 2e-13
        assert np.all(error_new >= -tolerance)
        assert np.all(error_new < step + tolerance)
        accepted = ~new.amp_flag.values
        limited = accepted & (new.adrv_best.values > new.adrv_best_adrv.values)
        row = {
            "network": network,
            "channels": len(new.chan),
            "accepted": int(accepted.sum()),
            "filled": int((~accepted).sum()),
            "percentile_limited": int(limited.sum()),
            "fit_arrays_exact": True,
            "matching_policy_exact": True,
            "old_attenuation_db": old.attrs["atten_drive_global"],
            "exact_attenuation_db": new.attrs["atten_drive_exact"],
            "new_attenuation_db": new.attrs["atten_drive_global"],
            "old_model_error_db": float(np.median(error_old)),
            "new_model_error_db": float(np.median(error_new)),
            "max_amplitude_change": float(
                abs(new.amp_best.values - old.amp_best.values).max()
            ),
            "max_emitted_amplitude_change": float(
                abs(
                    new.amp_best.values[controlling] - old.amp_best.values[controlling]
                ).max()
            ),
            "reference_bins": len(np.unique(bins_ref)),
            "target_bins": len(np.unique(bins_target)),
            "reference_phase_headers": sum(
                item["phase"] is not None for item in ref_headers
            ),
            "target_phase_header": target["phase"] is not None,
            "waveform_error_before_db": None,
            "waveform_error_after_db": None,
        }
        if (
            all(item["phase"] is not None for item in ref_headers)
            and target["phase"] is not None
        ):
            for item in ref_headers[1:]:
                np.testing.assert_array_equal(item["phase"], reference["phase"])
            peak_ref = waveform_peak(bins_ref, reference["amp"], reference["phase"])
            for value, key in ((old, "before"), (new, "after")):
                peak = waveform_peak(
                    bins_target, value.amp_best.values, target["phase"]
                )
                delta = (
                    20
                    * np.log10(
                        value.amp_best.values / reference["amp"][value.idx_ref.values]
                    )
                    + value.adrv_best.values
                    - value.attrs["atten_drive_global"]
                    + 20 * np.log10(peak_ref / peak)
                )
                row[f"waveform_error_{key}_db"] = float(np.median(delta))
        records.append(row)
        Table(
            {
                "chan": new.chan.values,
                "policy_power_db": desired,
                "power_before_db": old_achieved,
                "power_after_db": new_achieved,
                "error_before_db": error_old,
                "error_after_db": error_new,
                "amp_before": old.amp_best.values,
                "amp_after": new.amp_best.values,
                "accepted": accepted,
                "percentile_limited": limited,
            }
        ).write(args.output / f"toltec{network}-comparison.ecsv", overwrite=True)
    summary = {
        "before": before,
        "after": after,
        "networks": records,
        "power_source": str(args.power_source),
        "power_source_sha256": hashlib.sha256(
            args.power_source.read_bytes()
        ).hexdigest(),
        "waveform_assumption": (
            "Output retains the target sweep's recorded phases and masks; "
            "calculated prediction, not a measured post-commit sweep."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(records))
    for sign, key, label, color in (
        (-1, "old", "Before", "#a33d35"),
        (1, "new", "Corrected", "#24638b"),
    ):
        axes[0].bar(
            x + sign * 0.18,
            [r[f"{key}_attenuation_db"] for r in records],
            0.36,
            label=label,
            color=color,
        )
        axes[1].bar(
            x + sign * 0.18,
            [r[f"{key}_model_error_db"] for r in records],
            0.36,
            label=label,
            color=color,
        )
    axes[0].set_ylabel("Global drive attenuation (dB)")
    axes[1].set_ylabel("Predicted power - policy target (dB)")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].axhline(0.25, color="gray", lw=0.8, ls="--", label="One attenuation step")
    for ax in axes:
        ax.set_xticks(x, [f"Network {r['network']}" for r in records])
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "normalization-effect.svg")
    fig.savefig(args.output / "normalization-effect.png", dpi=180)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "power-source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--networks", type=int, nargs="+", default=[0, 7, 11])
    compare(parser.parse_args())
