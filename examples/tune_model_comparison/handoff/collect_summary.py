#!/usr/bin/env python
"""Collect the reviewed 152434 evidence; never rerun or rescale a reduction."""

# ruff: noqa: E501, T201, PLR2004 -- scientific figure labels and panel geometry
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from array_coordinate_control import common_maps  # noqa: E402
from compare_maps import read_map  # noqa: E402


def collect(base, destination):  # noqa: C901, PLR0915 -- ordered evidence collection
    """Keep compact evidence and source hashes alongside the recipient report."""
    destination.mkdir(parents=True, exist_ok=True)
    report = base / "report"
    joint = json.loads((report / "joint-control.json").read_text())
    quality = json.loads((report / "sweep-model-quality.json").read_text())
    evidence = {
        "observation": 152434,
        "arrays": joint["arrays"],
        "runs": {
            k: {f: v[f] for f in ("n_samples", "policy", "arrays")}
            for k, v in joint["runs"].items()
        },
        "sweep": {k: v["selected_responses"] for k, v in quality["arrays"].items()},
        "apt_sha256": "5f9a99b50829f2d44746b40d16503e4528f2375fc5d71b345bece16216d22831",
        "source_hashes": {},
    }
    for name in (
        "joint-control.json",
        "sweep-model-quality.json",
        "joint-review-checks.json",
    ):
        evidence["source_hashes"][name] = hashlib.sha256(
            (report / name).read_bytes()
        ).hexdigest()
    mpl.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    pos = np.arange(3)
    colors = ("#596875", "#65428a")
    fig, ax = plt.subplots(figsize=(7, 2.7), layout="constrained")
    for k, (key, label) in enumerate(
        (
            ("legacy_median_fraction", "Legacy tune.txt"),
            ("v3_median_fraction", "V3 full group"),
        )
    ):
        bars = ax.bar(
            pos + (k - 0.5) * 0.32,
            [evidence["sweep"][str(i)][key] for i in pos],
            width=0.31,
            label=label,
            color=colors[k],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set(
        xticks=pos,
        xticklabels=[r["array"] for r in joint["arrays"]],
        ylabel="Median normalized IQ residual",
        ylim=(0, 0.28),
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    fig.savefig(destination / "sweep-quality.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(7, 8.2), layout="constrained")
    names = ("legacy-broader", "joint-physical")
    for i, row in enumerate(joint["arrays"]):
        items = []
        for name in names:
            run = Path(joint["runs"][name]["path"])
            [path] = (run / "152434/raw").glob(f"*_{row['array']}_*.fits")
            evidence["source_hashes"][f"{name}/{path.name}"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            items.append(read_map(path))
        aligned, mask, _ = common_maps(items)
        limit = np.percentile(
            np.abs(np.concatenate([m["signal"][mask] for m in aligned])), 99
        )
        for j, (name, item) in enumerate(zip(names, aligned, strict=True)):
            ax = axes[i, j]
            image = ax.pcolormesh(
                item["x"],
                item["y"],
                np.where(mask, item["signal"], np.nan),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="auto",
            )
            n = joint["runs"][name]["arrays"][i]["contributors"]
            title = ("Legacy reference", "V3 full model")[j]
            ax.set(
                xlim=(-40, 40),
                ylim=(-40, 40),
                aspect="equal",
                title=f"{title}\n{n:,} contributors",
            )
            if j == 0:
                ax.set_ylabel(f"{row['array']}\nElevation offset (arcsec)")
            if i == 2:
                ax.set_xlabel("Azimuth offset (arcsec)")
        fig.colorbar(image, ax=axes[i], label="mJy/beam", shrink=0.9)
    fig.savefig(destination / "maps.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    keys = (
        "scalar_eligible",
        "joint_added",
        "fit_unusable",
        "no_unique_component_binding",
    )
    labels = (
        "Scalar route retained",
        "Joint components added",
        "Fit/measurement limits",
        "Identity binding absent",
    )
    bottom = np.zeros(3)
    for key, label, color in zip(
        keys, labels, ("#8fa0ac", "#65428a", "#c49d69", "#d5d5d5"), strict=True
    ):
        values = np.array(
            [r.get(key, r["disposition"].get(key, 0)) for r in joint["arrays"]]
        )
        ax.barh(pos, values, left=bottom, label=label, color=color)
        for j, value in enumerate(values):
            if value > 140:
                ax.text(
                    bottom[j] + value / 2,
                    j,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if key == "joint_added" else "black",
                    fontsize=10,
                )
        bottom += values
    ax.set(
        yticks=pos,
        yticklabels=[r["array"] for r in joint["arrays"]],
        xlabel="Originally beammap-good acquired APT rows",
    )
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.savefig(destination / "population.png", dpi=200)
    plt.close(fig)
    shutil.copy2(
        report / "timestream-examples/joint-sweep.png", destination / "joint-sweep.png"
    )
    (destination / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "evidence")
    args = parser.parse_args()
    collect(args.base, args.output)
