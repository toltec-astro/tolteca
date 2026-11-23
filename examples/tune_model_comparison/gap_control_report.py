"""Render the gap correction and sample-acceptance control from saved evidence."""

# ruff: noqa: INP001, E501, RUF001 -- scientific report and figures

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from array_coordinate_control import ARRAYS, common_maps, save
from compare_maps import read_map
from joint_map_report import integrate_html

COLORS = ("#606b76", "#75449b", "#237d98")
DISPLAY_ORDER = ("legacy", "v3-numerical", "v3-physical")
DISPLAY_LABELS = ("Legacy reference", "V3 numerical", "V3 physical")
DISPLAY_COLORS = (COLORS[0], COLORS[2], COLORS[1])


def policy_summary(result):
    """Compare acceptance policies by run identity, not positional legend order."""
    rows = []
    for measured in result["maps"]:
        array = measured["array"]
        rms = dict(zip(result["order"], measured["rms_mJy_beam"], strict=True))
        counts = {
            name: next(r for r in result["runs"][name]["arrays"] if r["array"] == array)
            for name in DISPLAY_ORDER
        }
        numerical = counts["v3-numerical"]["accepted_channel_samples"]
        physical = counts["v3-physical"]["accepted_channel_samples"]
        rows.append(
            {
                "array": array,
                "rms_mJy_beam": rms,
                "counts": counts,
                "physical_to_numerical_rms": rms["v3-physical"] / rms["v3-numerical"],
                "physical_retained_fraction": physical / numerical,
                "additional_rejected_channel_samples": numerical - physical,
                "common_annulus_pixels": measured["common_annulus_pixels"],
            }
        )
    return {
        "source": "gap-control.json",
        "order": list(DISPLAY_ORDER),
        "arrays": rows,
        "mapping_speed": {
            "status": "not_validated",
            "reason": "Annulus fluctuations include residual sky/source signal. Matched astronomical signal transfer and noise-only sensitivity have not been established.",
        },
    }


def policy_figures(output):
    """Make the current summary exclusively from the corrected three-run audit."""
    path = output / "gap-control.json"
    summary = policy_summary(json.loads(path.read_text()))
    summary["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    old = output / "map-quality-summary.json"
    if old.exists() and json.loads(old.read_text()).get("source") != "gap-control.json":
        for suffix in ("json", "png", "pdf"):
            source = output / f"map-quality-summary.{suffix}"
            if source.exists():
                shutil.copy2(
                    source, output / f"historical-map-quality-summary.{suffix}"
                )
    rows = summary["arrays"]
    pos = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for k, (name, label, color) in enumerate(
        zip(DISPLAY_ORDER, DISPLAY_LABELS, DISPLAY_COLORS, strict=True)
    ):
        values = [r["rms_mJy_beam"][name] for r in rows]
        bars = axes[0].bar(
            pos + (k - 1) * 0.26, values, width=0.24, label=label, color=color
        )
        axes[0].bar_label(bars, fmt="%.2f", fontsize=10, rotation=90, padding=4)
    axes[0].set(
        title="Corrected maps: background RMS",
        ylabel="mJy/beam",
        ylim=(0, 1.3 * max(v for r in rows for v in r["rms_mJy_beam"].values())),
    )
    axes[0].legend(frameon=False, fontsize=11)
    excluded = [100 * (1 - r["physical_retained_fraction"]) for r in rows]
    bars = axes[1].bar(pos, excluded, width=0.5, color=COLORS[1])
    axes[1].bar_label(
        bars,
        labels=[
            f"{v:.3f}%\n{r['additional_rejected_channel_samples']:,} samples"
            for v, r in zip(excluded, rows, strict=True)
        ],
        fontsize=10,
    )
    axes[1].set(
        title="Additional rejection by physical acceptance",
        ylabel="% of V3 numerical channel-samples",
        ylim=(0, max(0.1, *excluded) * 1.4),
    )
    for ax in axes:
        ax.set(xticks=pos, xticklabels=[r["array"] for r in rows])
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.title.set_size(12)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    save(fig, output, "map-quality-summary")
    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    for k, (name, label, color) in enumerate(
        zip(DISPLAY_ORDER, DISPLAY_LABELS, DISPLAY_COLORS, strict=True)
    ):
        bars = ax.bar(
            pos + (k - 1) * 0.26,
            [r["counts"][name]["contributing_channels"] for r in rows],
            width=0.24,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.0f", fontsize=9)
    ax.set(
        xticks=pos,
        xticklabels=[
            f"{r['array']}\n{r['counts']['v3-numerical']['intended_channels']:,} eligible"
            for r in rows
        ],
        ylabel="Contributing channel/APT rows",
        ylim=(0, 2400),
        title="Current runs: one eligibility set, two sample policies",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, output, "gap-populations")
    old.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def performance_section(result):
    """Explain the policy comparison and its denominators before historical plots."""
    summary = policy_summary(result)
    rows = "".join(
        f"<tr><td>{r['array']}</td>"
        + "".join(f"<td>{r['rms_mJy_beam'][n]:.2f}</td>" for n in DISPLAY_ORDER)
        + f"<td>{100 * (1 - r['physical_to_numerical_rms']):.2f}%</td>"
        + f"<td>{100 * r['physical_retained_fraction']:.3f}%</td></tr>"
        for r in summary["arrays"]
    )
    return (
        '<section id="map-quality"><h2>1. Current performance: numerical versus physical acceptance</h2>'
        "<p>These are two sample-acceptance policies for the same fitted nonlinear V3 model and optical-detuning output. "
        "They are not numerical versus physical models. Numerical acceptance requires a finite, solvable response; "
        "physical acceptance additionally rejects domain/stability warnings. Both keep the solver diagnostic flags.</p>"
        "<p>All three runs use the corrected filtering/PCA implementation, the same common-eligibility APT, "
        "and identical per-channel weights. Legacy and V3 numerical also have identical sample masks, map weights "
        "and coverage. Physical acceptance changes coverage. “Full intended population” means all 3,096 "
        "initially eligible channel/APT rows, not all 4,646 beammap-good channels; joint component-to-beam "
        "identity gaps and unusable models remain excluded.</p>"
        '<figure><img src="gap-populations.png" alt="Current contributing channel counts">'
        "<figcaption>Figure 2. Current acceptance accounting; counts are acquired channel/APT rows, not independent resonators.</figcaption></figure>"
        '<figure><a href="map-quality-summary.pdf"><img src="map-quality-summary.png" alt="Numerical and physical acceptance RMS and sample cost"></a>'
        "<figcaption>Figure 3. Corrected maps on the same covered 20–40 arcsec annulus: background RMS and "
        "additional channel-samples rejected by physical acceptance. The sample percentage uses V3 numerical accepted samples as its denominator.</figcaption></figure>"
        "<table><thead><tr><th>Array</th><th>Legacy RMS</th><th>V3 numerical RMS</th><th>V3 physical RMS</th>"
        "<th>RMS decrease, physical vs numerical</th><th>Samples retained, physical/numerical</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<p>RMS is in mJy/beam. Physical acceptance lowers this metric most in a1100 and a2000; "
        "its effect in a1400 is small. Against the matched legacy reference, the earlier large a1100 excess "
        "is absent; V3 numerical remains higher in a1400 and slightly higher in a2000. "
        "The policy comparison changes both accepted samples and the estimated cleaning basis.</p>"
        "<p><strong>These are map-fluctuation measurements, not validated sensitivity or mapping speed.</strong> "
        "The annulus contains correlated residuals and possible source wings. Lower RMS does not establish "
        "equal astronomical signal transfer. A policy decision needs that validation as well as the explicit "
        "sample cost. No signal rescaling or new pointing calibration is adopted.</p>"
        '<p><a href="map-quality-summary.json">Current summary and exact denominators</a> · '
        '<a href="gap-control.json">Native audit and source hashes</a>.</p></section>'
    )


def headline(result):
    """Lead with the new full-population result, keeping the earlier control below."""
    rms = dict(zip(result["order"], result["maps"][0]["rms_mJy_beam"], strict=True))
    return (
        "<section id='gap-current-finding'><h2>Current result: gap correction and explicit acceptance</h2>"
        "<p>This review separates the current corrected maps, the fitted-model evidence, and the earlier "
        "investigation. The native correction fills gaps before filtering, preserves the map rejection masks, "
        "and fits PCA mode amplitudes using only valid samples. All three runs use the original common APT "
        "and byte-identical per-channel weights; no always-valid subset is selected. "
        "The legacy/V3 numerical pair also has identical sample masks, map weights and coverage: "
        "1,902 contributing a1100 channels after the same 33 nearby-tone exclusions.</p>"
        f"<p><strong>a1100 annulus RMS:</strong> legacy {rms['legacy']:.2f}, V3 numerical {rms['v3-numerical']:.2f}, "
        f"V3 physical {rms['v3-physical']:.2f} mJy/beam. Numerical acceptance retains finite physical-domain/stability "
        "violations and their diagnostic flags; hard failures remain excluded. These RMS values are not "
        "validated sensitivity. <a href='#gap-control'>See the policy, maps and acceptance accounting</a>.</p></section>"
    )


def html_section(result):
    """Separate computational validity, model diagnostics and map eligibility."""
    populations = "".join(
        f"<tr><td>{label}</td><td>{r['array']}</td><td>{r['intended_channels']}</td>"
        f"<td>{r['contributing_channels']}</td><td>{r['partially_flagged_channels']}</td>"
        f"<td>{r['rejected_channel_samples']}</td></tr>"
        for name, label in zip(DISPLAY_ORDER, DISPLAY_LABELS, strict=True)
        for r in result["runs"][name]["arrays"]
    )
    return (
        "<section id='gap-control'><h3>1.1 Processing, maps and accepted samples</h3>"
        "<p>Filtering needs continuous numerical input across rejected samples while preserving their map "
        "exclusion. The shared gap filler follows the despiker construction: endpoint "
        "baseline, observed network fluctuations and reproducible residual noise. It does not erase flags "
        "or turn reconstructed values into data. It also repairs endpoint and fully missing-channel handling.</p>"
        "<p>Every FIR output whose support touches a gap stays excluded from maps. Covariance uses valid "
        "overlaps; mode amplitudes use valid-sample least squares with the missing-row Gram correction. "
        "Insufficient mode support rejects that network's time sample. Tests change filled values by many "
        "orders of magnitude without changing valid cleaned output, and exercise the actual map accumulator.</p>"
        "<pre>kids:\n  solver:\n    sample_acceptance: numerical  # physical remains the default</pre>"
        "<p><code>physical</code> applies the domain/stability checks. <code>numerical</code> keeps finite "
        "algebraic results carrying only those warnings, so the model comparison does not impose new "
        "physical selection cuts on V3. Both retain identical solver diagnostic bits. Invalid models, "
        "singular/nonfinite responses and failed joint recovery are never made valid by this option. "
        "It does not change fitting, the nonlinear correction, or joint branch tracking. "
        "Filled samples remain excluded under either policy; retaining a finite solved value is different "
        "from accepting a replacement value in a gap.</p>"
        "<p>Map RMS in mJy/beam on identical covered 20–40 arcsec annulus pixels. Per-channel weights are "
        "identical. The numerical pair has identical sample masks, map weights and coverage; "
        "the physical policy changes coverage. No signal rescaling.</p>"
        "<figure><a href='gap-maps.pdf'><img src='gap-maps.png' alt='Full intended population maps with corrected gaps'></a>"
        "<figcaption>One signal scale per array, shared across the three policies.</figcaption></figure>"
        f"<details><summary>Full intended population and accepted samples</summary><table><thead><tr>"
        "<th>Run</th><th>Array</th><th>Intended channels</th><th>Contributing</th><th>Partly rejected</th>"
        f"<th>Rejected samples among intended channels</th></tr></thead><tbody>{populations}</tbody></table></details>"
        "<p>Whole-channel APT/nearby-tone exclusions remain in force. This does not recover missing model-to-APT "
        "component identities or prove that every finite modeled response is scientifically suitable.</p>"
        "<figure><a href='gap-trajectories.pdf'><img src='gap-trajectories.png' alt='Corrected a1100 network mean timestreams'></a>"
        "<figcaption>After PCA, each network mean uses the same APT weights and that run's accepted samples. "
        "Differences include sample acceptance; filled samples receive no weight in these means.</figcaption></figure>"
        "<p><a href='gap-control.json'>Checks, exact metrics and hashed outputs</a> · "
        "<a href='../gap-control/legacy/redu00/config.yaml'>Legacy config</a> · "
        "<a href='../gap-control/v3-physical/redu00/config.yaml'>V3 physical config</a> · "
        "<a href='../gap-control/v3-numerical/redu00/config.yaml'>V3 numerical config</a>.</p></section>"
    )


def reorganize_html(text, result):
    """Keep current findings and older investigation in separate, stable sections."""

    def section(name):
        match = re.search(
            rf"<section id=[\'\"]{name}[\'\"]>.*?</section>", text, re.DOTALL
        )
        if match is None:
            message = f"Missing report section {name}"
            raise ValueError(message)
        return match[0]

    model = section("model-quality")
    examples = section("timestream-examples")
    arrays = []
    for name in ARRAYS:
        body = section(name)
        body = re.sub(
            r"<p><strong>Historical capture, before the gap correction.</strong>.*?</p>",
            "",
            body,
        )
        body = re.sub(
            r"<h[23]>.*?</h[23]>",
            f"<h3>{name}: original maps and contributor breakdown</h3>",
            body,
            count=1,
        )
        arrays.append(
            body.replace(
                "</h3>",
                "</h3><p><strong>Historical capture, before the gap correction.</strong> These counts, maps and native pointing fits do not describe the current policy runs.</p>",
                1,
            )
        )
    if 'id="historical-overview"' in text:
        overview = section("historical-overview")
        provenance = section("historical-provenance")
    else:
        overview = text.split('<h2 id="overview">', 1)[1].split(
            '<section id="map-quality">', 1
        )[0]
        overview = (
            '<section id="historical-overview"><h3>Original APT and contribution accounting</h3>'
            + overview.split("</h2>", 1)[1]
            + "</section>"
        )
        provenance = (
            '<section id="historical-provenance"><h3>Original execution records</h3>'
            + text.split('<h2 id="provenance">', 1)[1]
            .split("</h2>", 1)[1]
            .removesuffix("</html>")
            + "</section>"
        )
    coordinate = section("coordinate-control")
    coordinate = re.sub(
        r"<p>The optical-coordinate versus x′ experiment below.*?</p>", "", coordinate
    )
    coordinate = re.sub(
        r"<h3>.*?</h3>",
        "<h3>B. Coordinate experiment before the gap correction</h3><p>The optical-coordinate versus x′ experiment below retains the old sample handling. It is not the current numerical/physical acceptance comparison.</p>",
        coordinate,
        count=1,
    )
    pca = section("pca-trace")
    pca = re.sub(
        r"<p>This diagnostic preceded the corrected full-population runs.*?</p>",
        "",
        pca,
    )
    pca = re.sub(
        r"<h3>.*?</h3>",
        "<h3>C. Diagnosing the old PCA projection</h3><p>This diagnostic preceded the corrected full-population runs. The 1,855-channel selection is retained only to explain the failure mechanism.</p>",
        pca,
        count=1,
    )
    pca = pca.replace(
        "the production implementation has not been changed.",
        "that diagnostic did not itself change native processing. The current comparison now uses the separately tested native correction.",
    )
    pca = pca.replace(
        "<strong>What remains:</strong> implement and test a flag-aware native projection, including rank-deficient or poorly constrained mode amplitudes, and rerun the broader population.",
        "<strong>Outcome:</strong> the flag-aware native projection, rank-deficient handling and broader-population rerun are now implemented and tested in the current comparison above.",
    )
    prefix = text.split("<nav>", 1)[0]
    current = (
        '<nav><a href="#map-quality">Current policy comparison</a> · <a href="#gap-control">Maps and processing</a> · '
        '<a href="#model-quality">Fit quality</a> · <a href="#timestream-examples">Solved examples</a> · '
        '<a href="#remaining">Remaining validation</a> · <a href="#history">Earlier investigation</a></nav>'
        + headline(result)
        + performance_section(result)
        + html_section(result)
        + model
        + examples
        + '<section id="remaining"><h2>3. Interpretation, reproducibility and remaining validation</h2>'
        "<p>The corrected runs remove the earlier large a1100 map excess. The numerical/physical comparison "
        "measures the effect of sample acceptance at fixed fitting and per-channel weighting. It does not "
        "demonstrate improved sensitivity from the V3 model or establish an optimal acceptance policy.</p>"
        "<p>The Python workflow remains the detailed-model reference. Measured Python/native parity in the "
        "145609 replay is still unresolved (14 affine and 16 delay groups out of 473); transport equality "
        "does not resolve that solving discrepancy. Multiplet component-to-beam/APT identities, independent "
        "signal transfer, and noise depth versus exposure remain needed before a scientific release. "
        "The two solved examples below the fit audit do not replace the full-network parity check.</p>"
        "<p>The current build, configs, source hashes and measured checks are linked in "
        '<a href="gap-control.json">gap-control.json</a>. Run the maintained '
        "<code>gap_control_report.py</code> recipe to regenerate this summary from those saved results; "
        "no model fitting or reduction is repeated by rendering. The older 22 November wheel bundle "
        "does not include the later native corrections.</p></section>"
        '<section id="history"><h2>Earlier investigation — superseded map comparisons</h2>'
        "<p>The following experiments explain how the gap/PCA problem was found. Their engines, cuts and "
        "sample populations differ from the current three-run comparison. Their map excesses, fitted "
        "offsets and contributor counts are historical results, not the current performance summary.</p></section>"
        "<h3>A. Original maps, populations and provenance</h3>"
        + overview
        + "".join(arrays)
        + provenance
        + coordinate
        + pca
    )
    # Number captions once in reading order; prior builders used conflicting numbers.
    number = 1

    def caption(match):
        nonlocal number
        number += 1
        content = re.sub(r"^Figure \d+\.\s*", "", match[1])
        return f"<figcaption>Figure {number}. {content}</figcaption>"

    current = re.sub(
        r"<figcaption>(.*?)</figcaption>", caption, current, flags=re.DOTALL
    )
    current = current.replace(
        "; Figure 3 separates them.", "; the model-quality section separates them."
    )
    combined = prefix + current + "</html>"
    if '<section id="joint-maps">' in text:
        combined = integrate_html(combined, section("joint-maps"))
    return combined


def render(base):
    """Build plots and update the existing review without rerunning reductions."""
    output = base / "report"
    result = json.loads((output / "gap-control.json").read_text())
    policy_figures(output)
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for a, array in enumerate(ARRAYS):
        items = []
        for name in DISPLAY_ORDER:
            [path] = (base / "gap-control" / name / "redu00/152434/raw").glob(
                f"*_{array}_*.fits"
            )
            items.append(read_map(path))
        aligned, mask, _ = common_maps(items)
        limit = np.percentile(
            np.concatenate([np.abs(m["signal"][mask]) for m in aligned]), 99
        )
        for k, m in enumerate(aligned):
            image = axes[a, k].pcolormesh(
                m["x"],
                m["y"],
                np.where(mask, m["signal"], np.nan),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="auto",
                rasterized=True,
            )
            axes[a, k].set(
                xlim=(-40, 40),
                ylim=(-40, 40),
                aspect="equal",
                xlabel="Azimuth offset (arcsec)",
                ylabel="Elevation offset (arcsec)",
                title=f"{array} · {DISPLAY_LABELS[k]}\nRMS {result['maps'][a]['rms_mJy_beam'][result['order'].index(DISPLAY_ORDER[k])]:.2f} mJy/beam",
            )
        fig.colorbar(image, ax=axes[a], label="mJy/beam", shrink=0.8)
    save(fig, output, "gap-maps")
    saved = np.load(output / "gap-trajectories.npz")
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    time = saved["time"] - saved["time"][0]
    for nw, ax in enumerate(axes.flat):
        for name, label, color in zip(
            DISPLAY_ORDER, DISPLAY_LABELS, DISPLAY_COLORS, strict=True
        ):
            ax.plot(time, saved[name][:, nw, 1], label=label, color=color, lw=0.7)
        ax.set(
            title=f"Network {nw} · after PCA",
            xlabel="Time from first saved sample (s)",
            ylabel="Weighted mean (mJy/beam)",
        )
    axes[0, 0].legend(fontsize=8)
    save(fig, output, "gap-trajectories")
    path = output / "index.html"
    path.write_text(reorganize_html(path.read_text(), result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    render(parser.parse_args().base.resolve())
