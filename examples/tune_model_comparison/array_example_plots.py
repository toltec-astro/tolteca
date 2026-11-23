"""Report the measured signal/noise examples and saved map quality."""

# ruff: noqa: INP001, PLR2004, RUF001, E501 -- scientific labels and declared bands

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy.table import Table
from compare_maps import compare, read_map, shared_axis
from gap_control_report import policy_figures

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_kids.model_fit import ModelFit
from tolteca_kids.product import open_kids_product
from tolteca_kidsproc.models import KidsSweepGainWithLinTrend

COLORS = ("#606b76", "#397bb3", "#75449b", "#b86b32", "#368770")
LABELS = ("Legacy x (effective)", "V3 x′", "V3 x_opt", "Legacy r (effective)", "V3 r")


def save(fig, output, name):
    """Preserve a full-resolution image and a vector print version."""
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_component(case, j, folder):
    """Compare one declared component/probe coordinate on a common interval."""
    name, component = case["name"], case["components"][j]
    with np.load(folder / f"{name}-{j}-comparison.npz") as d:
        values, time, frequency, power = (
            d[k] for k in ("values", "time", "frequency", "psd")
        )
        with xr.open_dataset(folder / f"{name}-{j}-spectra.nc") as spectra:
            spectra.load()
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5))
        for ax, columns in zip(axes[0], ((0, 1, 2), (3, 4)), strict=True):
            for i in columns:
                ax.plot(
                    time,
                    values[:, i] - np.median(values[:, i]),
                    lw=0.55,
                    color=COLORS[i],
                    label=LABELS[i],
                    alpha=0.85,
                )
            ax.set(
                xlabel="Time from acquisition start (s)",
                ylabel="Change from own median (dimensionless)",
            )
            ax.ticklabel_format(
                axis="y", style="sci", scilimits=(0, 0), useMathText=True
            )
            ax.legend(fontsize=8, frameon=False)
        axes[0, 0].set_title("Recovered detuning")
        axes[0, 1].set_title("Recovered loss")
        ax = axes[1, 0]
        for i in range(5):
            ax.loglog(
                frequency[1:], power[1:, i], color=COLORS[i], label=LABELS[i], lw=1
            )
        ax.axvspan(5, 20, alpha=0.1, color="gray")
        ax.set(
            xlabel="Temporal frequency (Hz)",
            ylabel="PSD (dimensionless² / Hz)",
            title="Same samples, same Welch settings",
        )
        ax.legend(fontsize=8, frameon=False)
        ax = axes[1, 1]
        for i, label, color in (
            (0, "x′ with r", COLORS[1]),
            (1, "x_opt with r", COLORS[2]),
        ):
            ax.semilogx(
                frequency[1:],
                spectra.coherence.values[1:, i, 2],
                label=label,
                color=color,
            )
        ax.set(
            xlabel="Temporal frequency (Hz)",
            ylabel="Magnitude-squared coherence",
            ylim=(0, 1),
            title="Detuning–loss correlation",
        )
        ax.legend(fontsize=8, frameon=False)
        for ax in axes.ravel():
            ax.grid(alpha=0.15)
        fig.suptitle(
            f"152434 · {name} group {case['group']['group_id']} · component {component['resonator_id']}\n"
            f"Network 0 · reference probe {component['coordinate_channel']} · a_nl = {component['a_nl']:.3g}",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        save(fig, folder, f"{name}-{j}-recovery")


def plot_joint_noise(case, folder):
    """Show component correlations and signed loss-to-detuning noise terms."""
    fig, axes = plt.subplots(3, 1, figsize=(6.7, 8.3))
    with np.load(folder / "joint-component-cross.npz") as d:
        f, c = d["frequency"], d["cross"]
        coherence = abs(c[:, 0, 1]) ** 2 / (c[:, 0, 0].real * c[:, 1, 1].real)
    axes[0].semilogx(f[1:], coherence[1:], color="#75449b")
    axes[0].set(
        ylim=(0, 1),
        ylabel="Coherence",
        title="Recovered fractional resonance-frequency changes",
    )
    for j, ax in enumerate(axes[1:]):
        with xr.open_dataset(folder / f"joint-{j}-spectra.nc") as s:
            f, terms = s.f_psd.values, s.linear_terms.values
            power = s.cross_real.values[:, 1, 1]
            band = (f >= 5) & (f <= 20)
            threshold = float(np.mean(power[band])) / 10
            for i, label in enumerate(("D² PSD(x′)", "μ² PSD(r)", "2Dμ Re C(x′, r)")):
                ax.plot(f[1:], terms[1:, i], lw=0.9, label=label, color=COLORS[i + 1])
            ax.plot(
                f[1:], power[1:], color="#303030", lw=0.8, label="Measured PSD(x_opt)"
            )
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=threshold)
            ax.set(
                ylabel="Signed PSD term (Hz⁻¹)",
                title=f"Component {case['components'][j]['resonator_id']}: local noise propagation",
            )
            ax.legend(fontsize=8, ncol=2, frameon=False)
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.set_xlabel("Temporal frequency (Hz)")
    fig.suptitle(
        "Joint solving: correlations and the x′ → x_opt correction", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, folder, "joint-noise")


def plot_sweeps(cases, prepared, legacy_manifest, folder):
    """Use the same selected examples for sweep fits and timestream diagnostics."""
    receipt = json.loads((prepared / "prepared.json").read_text())
    old_source = next(
        r
        for r in json.loads(legacy_manifest.read_text())["model_equivalence"]
        if r["interface"] == "toltec0"
    )
    old = Table.read(old_source["labeled"], format="ascii.ecsv")
    index = DataProdIndex.read(receipt["model_index"])
    with open_kids_product(index, interface="toltec0") as t:
        samples = ModelFit.view(t.toltec_kids.blocks[1].tree).samples.load()
    for case in cases:
        group = case["group"]
        fig, axes = plt.subplots(
            len(group["channels"]),
            2,
            figsize=(7.5, 2.9 * len(group["channels"]) + 0.8),
            squeeze=False,
        )
        for k, channel in enumerate(group["channels"]):
            take = (samples.group_id.values == group["group_id"]) & (
                samples.chan.values == channel
            )
            s = samples.isel(sample=np.flatnonzero(take))
            prediction = KidsSweepGainWithLinTrend.from_table(
                old[old["chan"] == channel]
            ).eval(s.frequency.values)
            f = (s.frequency.values - group["probes"][k]) / 1e3
            for a, (key, values) in enumerate(
                (("I", prediction.real), ("Q", prediction.imag))
            ):
                ax = axes[k, a]
                ax.plot(
                    f, s[key].values / 1e6, ".", ms=2.5, color="black", label="Measured"
                )
                ax.plot(
                    f,
                    np.squeeze(values) / 1e6,
                    color=COLORS[0],
                    label="Legacy tune.txt",
                )
                ax.plot(
                    f,
                    s[f"{key}_model"].values / 1e6,
                    color=COLORS[2],
                    label="V3 full group",
                )
                ax.set(
                    title=f"Channel {channel}",
                    xlabel="Frequency − recorded probe (kHz)",
                    ylabel=f"{key} (recorded IQ / 10⁶)",
                )
                ax.grid(alpha=0.15)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9
        )
        fig.suptitle(
            f"152434: {case['name']} group {group['group_id']} · M={len(group['channels'])}, N={len(group['resonator_ids'])}",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        save(fig, folder, f"{case['name']}-sweep")


def annulus_measurement(a, b):
    """Compare signal RMS and weight uncertainty on identical covered pixels."""
    x, y, aa, bb, _ = compare(a, b)
    radius = np.hypot(x[None, :], y[:, None])
    annulus = (radius >= 20) & (radius <= 40) & np.isfinite(aa) & np.isfinite(bb)
    ax, bx = shared_axis(a["x"], b["x"])
    ay, by = shared_axis(a["y"], b["y"])
    formal = [
        float(np.median(1 / np.sqrt(item["weight"][np.ix_(iy, ix)][annulus])))
        for item, ix, iy in ((a, ax, ay), (b, bx, by))
    ]
    return {
        "pixels": int(annulus.sum()),
        "annulus_rms": [float(np.std(v[annulus], ddof=1)) for v in (aa, bb)],
        "formal_weight_error_median": formal,
    }


def map_quality(output):
    """Keep measured annulus fluctuations separate from inverse-weight errors."""
    if (output / "gap-control.json").exists():
        return policy_figures(output)
    metrics = json.loads((output / "metrics.json").read_text())
    rows = []
    for m in metrics:
        a, b = [read_map(Path(s["path"])) for s in m["sources"][1:]]
        rows.append(
            {
                "array": m["array"],
                **annulus_measurement(a, b),
                "peak": [m["pointing"]["amp"][side] for side in ("left", "right")],
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    for j, (field, title) in enumerate(
        (
            ("annulus_rms", "Measured 20–40 arcsec background RMS"),
            ("formal_weight_error_median", "Formal inverse-weight map uncertainty"),
        )
    ):
        for k, (label, color) in enumerate(
            (("Legacy, common eligibility", COLORS[0]), ("V3 models", COLORS[2]))
        ):
            values = [r[field][k] for r in rows]
            positions = np.arange(3) + (k - 0.5) * 0.36
            axes[j].bar(positions, values, width=0.34, color=color, label=label)
            for pos, value in zip(positions, values, strict=True):
                axes[j].text(
                    pos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9
                )
        axes[j].set(
            xticks=range(3),
            xticklabels=[r["array"] for r in rows],
            ylabel="mJy/beam",
            title=title,
        )
        axes[j].set_ylim(0, axes[j].get_ylim()[1] * 1.13)
        axes[j].grid(axis="y", alpha=0.15)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, output, "map-quality-summary")
    result = {
        "arrays": rows,
        "mapping_speed": {
            "status": "not_validated",
            "reason": "No noise realizations, independent signal-transfer calibration or time-scaling check for these trials. Annulus RMS includes correlated residuals; weights assume the adopted noise model.",
            "reference": "https://doi.org/10.1051/0004-6361/201936220",
        },
    }
    (output / "map-quality-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def build(prepared, legacy_manifest, output):
    """Draw saved examples and make the map-quality summary explicit."""
    folder = output / "timestream-examples"
    summary = json.loads((folder / "summary.json").read_text())
    for case in summary["cases"]:
        if case["status"] != "complete":
            continue
        for j in range(len(case["components"])):
            plot_component(case, j, folder)
        if case["name"] == "joint":
            plot_joint_noise(case, folder)
    plot_sweeps(summary["cases"], prepared, legacy_manifest, folder)
    map_quality(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "legacy_manifest", "output"):
        parser.add_argument(key, type=Path)
    build(**vars(parser.parse_args()))
