"""Compare legacy x, V3 x_prime and V3 x_opt without refitting any sweep."""

# ruff: noqa: INP001, E501, RUF001, PLR2004 -- scientific plot/report labels

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from build_xprime_control import digest
from compare_maps import read_map, shared_axis
from netCDF4 import Dataset
from run_xprime_control import make_config

ARRAYS = ("a1100", "a1400", "a2000")
LABELS = ("Legacy · common eligibility", "V3 x′", "V3 x_opt")
COLORS = ("#606b76", "#397bb3", "#75449b")


def common_maps(items):
    """Measure all inputs on exactly the same covered pixels; never rescale."""
    x, y = items[0]["x"], items[0]["y"]
    for item in items[1:]:
        ix, _ = shared_axis(x, item["x"])
        iy, _ = shared_axis(y, item["y"])
        x, y = x[ix], y[iy]
    aligned = []
    for item in items:
        _, ix = shared_axis(x, item["x"])
        _, iy = shared_axis(y, item["y"])
        aligned.append(
            dict(
                item,
                x=x,
                y=y,
                **{
                    key: item[key][np.ix_(iy, ix)]
                    for key in ("signal", "weight", "coverage")
                },
            )
        )
    base = aligned[0]
    mask = np.ones_like(base["signal"], dtype=bool)
    for item, original in zip(aligned, items, strict=True):
        assert item["header"]["UNIT"] == base["header"]["UNIT"]
        mask &= np.isfinite(item["signal"]) & np.isfinite(item["weight"])
        mask &= (item["weight"] > 0) & (
            item["coverage"] >= 0.1 * np.nanmax(original["coverage"])
        )
    radius = np.hypot(base["x"][None, :], base["y"][:, None])
    annulus = mask & (radius >= 20) & (radius <= 40)
    if np.count_nonzero(annulus) < 3:
        msg = "Insufficient common annulus coverage"
        raise ValueError(msg)
    return aligned, mask & (radius <= 40), annulus


def read_statistics(folder):
    """Read native identities, contributions, weights and flagged fractions."""
    [path] = folder.glob("*_stats.nc")
    with Dataset(path) as data:
        return {
            k: np.ma.filled(data[k][:], np.nan)
            for k in ("apt_array", "apt_nw", "apt_kids_tone", "weights", "flagged_frac")
        }


def compare_statistics(left, right):
    """Distinguish changing map eligibility from changing final contributions."""
    for key in ("apt_array", "apt_nw", "apt_kids_tone"):
        np.testing.assert_equal(left[key], right[key])
    rows = []
    for a, array in enumerate(ARRAYS):
        take = left["apt_array"] == a
        wa, wb = left["weights"][:, take], right["weights"][:, take]
        fa, fb = left["flagged_frac"][:, take], right["flagged_frac"][:, take]
        used = []
        for w, f in ((wa, fa), (wb, fb)):
            used.append(np.isfinite(w) & (w > 0) & np.isfinite(f) & (f < 1))
        both = used[0] & used[1]
        rows.append(
            {
                "array": array,
                "contributors": [int(u.any(axis=0).sum()) for u in used],
                "contributing_chunk_channels": [int(u.sum()) for u in used],
                "changed_contribution_entries": int(
                    np.count_nonzero(used[0] != used[1])
                ),
                "changed_flagged_fraction_entries": int(
                    np.count_nonzero(
                        ~np.isclose(fa, fb, rtol=0, atol=0, equal_nan=True)
                    )
                ),
                "weights_equal_on_shared_contributions": bool(
                    np.array_equal(wa[both], wb[both])
                ),
                "all_weights_equal": bool(np.array_equal(wa, wb, equal_nan=True)),
            }
        )
    return rows


def save(fig, output, name):
    """Write both review and print figures."""
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def html_section(result):
    """Summarize the measured coordinate control in the main comparison."""

    def figure(name, caption):
        return (
            f'<figure><a href="{name}.pdf"><img src="{name}.png" '
            f'alt="{caption}"></a><figcaption>{caption}</figcaption></figure>'
        )

    rows = result["standard"]
    a1100, a1400, a2000 = [r["rms_mJy_per_beam"] for r in rows]
    text = (
        '<section id="coordinate-control"><h3>1.2. Does the nonlinear coordinate correction explain the map difference?</h3>'
        f"<p><strong>The a1100 excess remains with x′.</strong> Its background RMS is {a1100[1]:.2f} mJy/beam with x′, "
        f"{a1100[2]:.2f} with x<sub>opt</sub>, and {a1100[0]:.2f} in the common-eligibility legacy control. "
        f"For a1400, x′ lowers RMS from {a1400[2]:.2f} to {a1400[1]:.2f}; "
        f"a2000 changes from {a2000[2]:.2f} to {a2000[1]:.2f}. "
        "The x′ → x<sub>opt</sub> correction cannot by itself explain the large a1100 regression.</p>"
        "<p>This is a full-integration map control using the same V3 sweep fits, nonlinear coefficients, model stubs, "
        "APT, raw data and map settings. An isolated native build returns x′ for valid scalar-model samples after "
        "performing the existing x<sub>opt</sub> checks. It preserves the loss values and solver flags, including "
        "rejected samples. It does not set a<sub>nl</sub> or b to zero, refit the sweep, or enable blended-component maps.</p>"
        + figure(
            "xprime-rms",
            "Coordinate control: RMS on identical covered 20–40 arcsec annulus pixels within each comparison. "
            "Left: recorded map settings. Right: both variance-dependent detector cuts disabled in both V3 runs.",
        )
        + "<p><strong>Selection control.</strong> Recorded settings change the contributing channel/chunk entries when the "
        "coordinate changes. The second pair disables the lower and upper inverse-variance cuts in both raw and processed "
        "time chunks. Its saved contribution decisions, flagged fractions and per-channel weights agree exactly. "
        "The RMS difference remains largest in a1400; the smaller a1100 change does not recover the legacy map.</p>"
        "<details><summary>Exact contributor audit and reproducible evidence</summary>"
        "<table><tr><th>Array</th><th>Recorded: contributors x_opt / x′</th><th>Recorded: changed chunk/channel decisions</th>"
        "<th>Cuts disabled: shared contributors / chunk-channel entries</th></tr>"
    )
    for normal, nocut in zip(
        result["contributions"]["standard_opt_to_prime"],
        result["contributions"]["nocut_opt_to_prime"],
        strict=True,
    ):
        text += (
            f"<tr><td>{normal['array']}</td><td>{normal['contributors'][0]} / {normal['contributors'][1]}</td>"
            f"<td>{normal['changed_contribution_entries']}</td>"
            f"<td>{nocut['contributors'][0]} / {nocut['contributing_chunk_channels'][0]}</td></tr>"
        )
    text += (
        "</table><p>Counts are channel/APT rows, not independent optical detector identities. "
        "Equal flagged fractions alone do not establish bitwise equality of final per-sample masks. "
        "PCA modes are recomputed for each coordinate.</p><p>"
        '<a href="xprime-control.json">Metrics, hashes and complete build record</a> · '
        '<a href="../xprime-control/native/xprime-output.patch">Isolated native patch</a>.</p><ul>'
    )
    for directory, label in [
        ("standard", "V3 x′: recorded settings"),
        ("no-variance-cuts-opt", "V3 x_opt: variance cuts disabled"),
        ("no-variance-cuts-prime", "V3 x′: variance cuts disabled"),
    ]:
        text += (
            f'<li>{label}: <a href="../xprime-control/{directory}/redu00/config.yaml">Config</a> · '
            f'<a href="../xprime-control/{directory}/redu00/execution.json">Execution and input hashes</a> · '
            f'<a href="../xprime-control/{directory}/citlali.log">Log</a></li>'
        )
    text += (
        "</ul></details><p><strong>Scientific limit:</strong> these maps retain the existing APT flux calibration "
        "and sensitivity values. Lower RMS is not by itself improved sensitivity or a reason to adopt x′ as the "
        "science coordinate. Signal transfer, the fixed-b assumption, and the remaining a1100 residuals still "
        "need to be assessed. The isolated build is a diagnostic, not a new production mode.</p>"
    )
    for array in ARRAYS:
        text += figure(
            f"xprime-{array}",
            f"{array}: legacy common-eligibility, V3 x′ and V3 x_opt maps with one intensity scale; "
            "the signed difference has its own symmetric scale. Exact shared sky pixels, no interpolation or signal rescaling. "
            "Recorded settings; downstream variance cuts can change contributions.",
        )
    return text + "</section>"


def update_report(output, result):
    """Refresh this section without rerunning the saved model examples."""
    path = output / "index.html"
    text = path.read_text()
    start = text.find('<section id="coordinate-control">')
    if start >= 0:
        end = text.index("</section>", start) + len("</section>")
        text = text[:start] + text[end:]
    marker = '<section id="model-quality">'
    assert text.count(marker) == 1
    text = text.replace(marker, html_section(result) + marker)
    if 'href="#coordinate-control"' not in text:
        text = text.replace(
            '<a href="#model-quality">',
            '<a href="#coordinate-control">x′ map control</a> · <a href="#model-quality">',
        )
    gap = output / "gap-control.json"
    if gap.exists():
        # Import after module initialization: the current report uses common_maps.
        from gap_control_report import reorganize_html  # noqa: PLC0415

        text = reorganize_html(text, json.loads(gap.read_text()))
    path.write_text(text)


def draw_maps(array, items, output):
    """Use one intensity scale plus a signed difference scale per array."""
    items, aperture, annulus = common_maps(items)
    x, y = items[0]["x"], items[0]["y"]
    signals = [np.where(aperture, m["signal"], np.nan) for m in items]
    limits = (min(np.nanmin(s) for s in signals), max(np.nanmax(s) for s in signals))
    dx, dy = np.diff(x)[0], np.diff(y)[0]
    extent = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.8), layout="constrained")
    for ax, signal, title in zip(axes.flat, signals, LABELS, strict=False):
        im = ax.imshow(
            signal,
            origin="lower",
            extent=extent,
            vmin=limits[0],
            vmax=limits[1],
            cmap="viridis",
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="mJy/beam", shrink=0.85)
    diff = signals[1] - signals[2]
    scale = np.nanmax(abs(diff))
    ax = axes[1, 1]
    im = ax.imshow(
        diff, origin="lower", extent=extent, vmin=-scale, vmax=scale, cmap="RdBu_r"
    )
    ax.set_title("x′ − x_opt")
    fig.colorbar(im, ax=ax, label="mJy/beam", shrink=0.85)
    for ax in axes.flat:
        ax.set(
            xlim=(-40, 40),
            ylim=(-40, 40),
            xlabel="Azimuth offset (arcsec)",
            ylabel="Elevation offset (arcsec)",
        )
    fig.suptitle(f"152434 · {array} · same V3 fit, two output coordinates", fontsize=13)
    save(fig, output, f"xprime-{array}")
    return {
        "array": array,
        "common_annulus_pixels": int(annulus.sum()),
        "rms_mJy_per_beam": [
            float(np.std(m["signal"][annulus], ddof=1)) for m in items
        ],
        "peak_mJy_per_beam": [float(m["header"]["POINTING.AMP"]) for m in items],
        "fwhm_arcsec": [
            [float(m["header"][f"POINTING.{k}"]) for k in ("A_FWHM", "B_FWHM")]
            for m in items
        ],
        "sources": [
            {"path": str(m["path"]), "sha256": digest(m["path"])} for m in items
        ],
    }


def build(base, output):
    """Audit settings and make the coordinate experiment part of the main report."""
    base, output = base.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials = {
        "legacy": base / "prepared/legacy-common/redu01",
        "prime": base / "xprime-control/standard/redu00",
        "opt": base / "prepared/python-full/redu01",
        "nocut_prime": base / "xprime-control/no-variance-cuts-prime/redu00",
        "nocut_opt": base / "xprime-control/no-variance-cuts-opt/redu00",
    }
    configs = {}
    for name, path in trials.items():
        c = yaml.safe_load((path / "config.yaml").read_text())
        c["runtime"].pop("output_dir")
        configs[name] = c
    assert configs["prime"] == configs["opt"]
    assert configs["nocut_prime"] == configs["nocut_opt"]
    expected = make_config(configs["opt"], base, variance_cuts=False)
    expected["runtime"].pop("output_dir")
    assert configs["nocut_opt"] == expected
    receipts = {
        name: json.loads((trials[name] / "execution.json").read_text())
        for name in ("prime", "opt", "nocut_prime", "nocut_opt")
    }
    assert all(r["exit_code"] == 0 for r in receipts.values())
    assert receipts["opt"]["binary_sha256"] == receipts["nocut_opt"]["binary_sha256"]
    assert (
        receipts["prime"]["binary_sha256"] == receipts["nocut_prime"]["binary_sha256"]
    )
    assert (
        receipts["prime"]["input_hashes"]
        == receipts["nocut_prime"]["input_hashes"]
        == receipts["nocut_opt"]["input_hashes"]
    )

    def maps(name, array):
        [path] = (trials[name] / "152434/raw").glob(f"*_{array}_*.fits")
        return read_map(path)

    rows = []
    nocut: list[dict[str, Any]] = []
    for array in ARRAYS:
        rows.append(
            draw_maps(
                array, [maps(n, array) for n in ("legacy", "prime", "opt")], output
            )
        )
        items = [maps(n, array) for n in ("nocut_prime", "nocut_opt")]
        items, _, annulus = common_maps(items)
        nocut.append(
            {
                "array": array,
                "common_annulus_pixels": int(annulus.sum()),
                "rms_mJy_per_beam": [
                    float(np.std(m["signal"][annulus], ddof=1)) for m in items
                ],
                "coverage_exactly_equal": bool(
                    np.array_equal(items[0]["coverage"], items[1]["coverage"])
                ),
                "weight_maps_exactly_equal": bool(
                    np.array_equal(items[0]["weight"], items[1]["weight"])
                ),
                "sources": [
                    {"path": str(m["path"]), "sha256": digest(m["path"])} for m in items
                ],
            }
        )
    stats = {n: read_statistics(p / "152434/raw") for n, p in trials.items()}
    comparisons = {
        label: compare_statistics(stats[left], stats[right])
        for label, left, right in (
            ("standard_opt_to_prime", "opt", "prime"),
            ("nocut_opt_to_prime", "nocut_opt", "nocut_prime"),
        )
    }
    assert all(
        r["changed_contribution_entries"] == 0
        and r["changed_flagged_fraction_entries"] == 0
        and r["all_weights_equal"]
        for r in comparisons["nocut_opt_to_prime"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for j, (data, labels, colors, title) in enumerate(
        (
            (rows, LABELS, COLORS, "Recorded map settings"),
            (
                nocut,
                ("V3 x′", "V3 x_opt"),
                COLORS[1:],
                "Variance-dependent cuts disabled",
            ),
        )
    ):
        width = 0.8 / len(labels)
        for k, (label, color) in enumerate(zip(labels, colors, strict=True)):
            pos = np.arange(3) + (k - (len(labels) - 1) / 2) * width
            values = [r["rms_mJy_per_beam"][k] for r in data]
            axes[j].bar(pos, values, width=width * 0.95, color=color, label=label)
            for x, v in zip(pos, values, strict=True):
                axes[j].text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        axes[j].set(
            xticks=range(3),
            xticklabels=ARRAYS,
            ylabel="20–40 arcsec RMS (mJy/beam)",
            title=title,
            ylim=(0, 165),
        )
        axes[j].legend(fontsize=8, frameon=False, loc="upper right")
        axes[j].grid(axis="y", alpha=0.15)
    save(fig, output, "xprime-rms")
    result = {
        "labels": list(LABELS),
        "standard": rows,
        "no_variance_cuts_labels": ["V3 x_prime", "V3 x_opt"],
        "no_variance_cuts": nocut,
        "contributions": comparisons,
        "trials": {k: str(p) for k, p in trials.items()},
        "execution_audit": {
            "new_trial_input_hashes_identical": True,
            "input_artifact_count": len(receipts["prime"]["input_hashes"]),
            "baseline_binary_matches_existing_xopt_trial": True,
            "records": {
                name: {
                    "path": str(trials[name] / "execution.json"),
                    "sha256": digest(trials[name] / "execution.json"),
                }
                for name in receipts
            },
        },
        "native_build": json.loads(
            (base / "xprime-control/native/build.json").read_text()
        ),
        "interpretation": {
            "a1100": "x_prime does not recover the cleaner legacy map. The x_prime-to-x_opt correction is not the sole explanation of the main a1100 excess.",
            "a1400": "x_prime has substantially smaller annulus RMS, including when variance cuts are disabled and saved contributions/weights agree.",
            "limits": "No refit, changed nonlinear coefficient, recalibrated APT, native joint output or validated sensitivity claim. PCA modes are recomputed for each coordinate. Bitwise equality of per-sample final masks is not inferred merely from equal flagged fractions.",
        },
    }
    (output / "xprime-control.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--update-report", action="store_true")
    args = parser.parse_args()
    result = build(args.base, args.output)
    if args.update_report:
        update_report(args.output, result)
