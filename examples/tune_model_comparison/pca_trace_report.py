"""Render the matched 152434 PCA trace and update the comparison summary."""

# ruff: noqa: INP001, E501, RUF001, PLR2004 -- scientific plots and report prose

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from array_coordinate_control import common_maps, save
from astropy.table import Table
from compare_maps import read_map

COLORS = ("#606b76", "#75449b", "#397bb3", "#b45a35")


def headline(result):
    """Give the latest matched result precedence over earlier map controls."""
    legacy, v3 = result["matched_map_rms_mJy_beam"]
    return (
        f'<p id="pca-current-finding"><strong>Updated a1100 finding:</strong> '
        f"with the same {result['n_channels']:,} always-valid channels, sample masks and weights, "
        f"the V3 map RMS is {v3:.2f} versus legacy {legacy:.2f} mJy/beam. "
        "A native PCA projection bug lets stored flagged values contaminate valid channels in the broader V3 run. "
        '<a href="#pca-trace">See the before/after-PCA trace and diagnostic replay</a>.</p>'
    )


def html_section(result, leakage):
    """Describe the measured failure mechanism without claiming sensitivity gains."""

    def fig(name, caption):
        return f'<figure><a href="{name}.pdf"><img src="{name}.png" alt="{caption}"></a><figcaption>{caption}</figcaption></figure>'

    text = (
        '<section id="pca-trace"><h3>1.3 Trace a1100 before and after PCA</h3>'
        "<p><strong>The original excess is not evidence that the better sweep fit gives a worse map.</strong> "
        "This control uses v2 tune.txt coefficients and V3 x_opt in the same pinned Citlali executable. "
        "It selects the intersection of channels that are finite and unflagged at every saved pre- and post-PCA sample. "
        "No fit residual, nonlinearity or noise-quality cut is added. The result is 1,855 of the 1,935 "
        "initially eligible a1100 channels, with 3,628 common samples in 12 chunks. Every retained sample is "
        "unflagged; detector weights, pointing coordinates, timestamps, map weights and coverage agree exactly.</p>"
        + fig(
            "pca-matched-maps",
            "Matched a1100 maps: native Jinc mapping, common detector population, exact equal weights and coverage. "
            "One signal scale; difference has a symmetric scale. No signal rescaling.",
        )
        + "<p><strong>Confirmed implementation failure:</strong> the covariance stage masks flagged samples, "
        "but the projection computes the mode amplitudes from the entire stored signal matrix. Nonzero stored "
        "values in partially flagged probes therefore enter the mode amplitudes and are subtracted from "
        "otherwise valid probes. The legacy capture has no partially flagged probes in these chunks, "
        "whereas the V3 capture does. This is a mask-handling inconsistency, not evidence of worse sweep fitting.</p>"
        "<p>The replay keeps the original saved PCA basis and the same 1,855 valid target channels and weights. "
        "Zeroing flagged contributions removes their direct leakage. Estimating the five amplitudes by least "
        "squares over valid probes also accounts for the missing basis rows. Both are diagnostic replays; "
        "the production implementation has not been changed.</p>"
        + fig(
            "pca-flag-leakage",
            "Flagged-value leakage on unchanged, always-valid target channels. Left: network-weighted means "
            "for networks 1 and 4. Right: weighted per-channel residual RMS for all six networks. "
            "The least-squares replay uses the captured basis and valid probes only.",
        )
        + "<p>The saved native outputs are reproduced from the saved eigenvectors to "
        f"{max(result['pca_replay_max_abs_mJy_beam'], leakage['native_replay_max_abs_mJy_beam']):.2g} mJy/beam maximum absolute error. "
        "The valid target channels have identical pre-PCA samples in the broad and strict captures; their "
        "post-PCA change is introduced by the treatment of other probes and the changed cleaning basis.</p>"
        + fig(
            "pca-stage-spectra",
            "Before and after native PCA on the strict common population, using identical map weights. "
            "Hann Welch spectra, 256 samples per segment, 128 overlap, calculated within each native chunk. "
            "These are fluctuation spectra including sky/source signal, not noise-only sensitivity.",
        )
        + fig(
            "pca-stage-trajectories",
            "The complete a1100 array's weighted mean timestream before and after native PCA on the same 1,855 channels. Chunk means are removed before the pre-PCA trace, matching the native operation. Channel weights are held equal, and network means are combined using their total channel weights.",
        )
        + fig(
            "pca-transfer",
            "Model-dependent signal transfer is already visible before PCA. "
            "Left: median and 10–90% intervals of per-channel/chunk regression gain. "
            "Right: weighted fluctuation RMS before and after each model’s native PCA. No regression gain is applied to a map.",
        )
        + "<p><strong>What remains:</strong> implement and test a flag-aware native projection, including "
        "rank-deficient or poorly constrained mode amplitudes, and rerun the broader population. "
        "The strict subset is a diagnostic control, not a proposed permanent detector exclusion. "
        "The APT flux factors remain fixed, so a lower annulus RMS does not establish improved "
        "astronomical signal transfer, sensitivity or mapping speed.</p>"
        "<details><summary>Reproduction and exact selection</summary><p>Per-network retained counts: "
        "357, 276, 292, 329, 297 and 304 for networks 0–5. Thirty-three channels fail the all-sample "
        "validity condition in legacy; eighty do in V3. All thirty-three legacy exclusions are also "
        "excluded in V3. Thus 47 additional channels account for the difference in this condition.</p>"
        "<p>The native RTC output is already filtered, downsampled by two and flux-calibrated; it is "
        "immediately before PCA, not raw I/Q. PTC is after mean subtraction, five-mode network PCA and weighting. "
        "Variance cuts are disabled in both. Exported chunk-start indices assume equal chunk lengths; "
        "we reconstruct boundaries from raw inner ranges and decimation, then verify the saved PCA output. "
        "This corrects diagnostic indexing without altering the native reduction.</p>"
        '<p><a href="pca-trace.json">Matched-stage audit and basis exchange</a> · '
        '<a href="pca-flag-projection.json">Flagged-projection replay</a> · '
        '<a href="pca-channel-chunks.ecsv">Every retained channel/chunk</a> · '
        '<a href="pca-spectra.npz">Spectra and network trajectories</a> · '
        '<a href="../pca-control/channel-selection.ecsv">Complete selection ledger</a> · '
        '<a href="../pca-control/matched-apt.ecsv">Shared APT</a></p><ul>'
    )
    for name in ("legacy", "v3"):
        text += (
            f'<li>{name}: <a href="../pca-control/matched/{name}/redu00/config.yaml">Matched config</a> · '
            f'<a href="../pca-control/matched/{name}/redu00/execution.json">Execution receipt</a> · '
            f'<a href="../pca-control/capture/{name}/redu00/config.yaml">Capture config</a></li>'
        )
    return text + "</ul></details></section>"


def figures(output):  # noqa: C901, PLR0915 -- sequential scientific figures
    """Plot population-matched maps, stage spectra, and projection leakage."""
    result = json.loads((output / "pca-trace.json").read_text())
    leakage = json.loads((output / "pca-flag-projection.json").read_text())
    with np.load(output / "pca-spectra.npz") as data:
        frequency, psd, time, native_means = [
            data[k] for k in ("frequency", "psd", "time", "network_means")
        ]
    with np.load(output / "pca-flag-trajectories.npz") as data:
        means = data["network_means"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), layout="constrained")
    for nw, ax in enumerate(axes.flat):
        for k, label in enumerate(result["stage_order"]):
            ax.loglog(
                frequency[1:],
                np.sqrt(psd[nw, k, 1:]),
                label=label,
                color=COLORS[k % 2],
                ls="--" if k < 2 else "-",
                lw=1.1,
            )
        ax.set(
            title=f"Network {nw} · {result['networks'][nw]['channels']} channels",
            xlabel="Temporal frequency (Hz)",
            ylabel="ASD (mJy/beam / √Hz)",
            xlim=(0.2, 25),
        )
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "152434 a1100 · exact same samples and weights before and after native PCA"
    )
    save(fig, output, "pca-stage-spectra")
    array_means = np.empty((len(time), 4))
    for chunk, (start, stop) in enumerate(result["chunk_bounds"]):
        weights = [r["sum_weights"] for r in result["chunks"] if r["chunk"] == chunk]
        array_means[start:stop] = np.average(
            native_means[start:stop], weights=weights, axis=1
        )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for phase, ax in enumerate(axes):
        for model, label in enumerate(("Legacy", "V3 x_opt")):
            ax.plot(
                time,
                array_means[:, phase * 2 + model],
                color=COLORS[model],
                label=label,
                lw=0.7,
            )
        ax.set(
            title=("Before PCA · chunk-centered", "After native PCA")[phase],
            xlabel="Time from first saved sample (s)",
            ylabel="Array-weighted mean (mJy/beam)",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    save(fig, output, "pca-stage-trajectories")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    table = Table.read(output / "pca-channel-chunks.ecsv")
    quantiles = np.array(
        [
            np.percentile(table["gain_v3_over_legacy"][table["nw"] == n], [10, 50, 90])
            for n in range(6)
        ]
    )
    axes[0].errorbar(
        range(6),
        quantiles[:, 1],
        yerr=(quantiles[:, 1] - quantiles[:, 0], quantiles[:, 2] - quantiles[:, 1]),
        fmt="o",
        capsize=4,
        color=COLORS[1],
    )
    axes[0].axhline(1, color="gray", ls=":")
    axes[0].set(
        xlabel="Network",
        ylabel="Pre-PCA V3 / legacy regression gain",
        title="Same raw data, different response scale",
    )
    values = np.array([r["rms_mJy_beam"] for r in result["networks"]])
    for k, label in enumerate(result["stage_order"]):
        axes[1].semilogy(
            range(6),
            values[:, k],
            label=label,
            color=COLORS[k % 2],
            ls="--" if k < 2 else "-",
            marker="o",
        )
    axes[1].set(
        xlabel="Network",
        ylabel="Weighted channel RMS (mJy/beam)",
        title="Strict common population",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    save(fig, output, "pca-transfer")
    fig = plt.figure(figsize=(11, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    for j, nw in enumerate((1, 4)):
        ax = fig.add_subplot(grid[j, 0])
        for model, k, label, color in (
            (1, 0, "V3 native projection", COLORS[3]),
            (1, 2, "V3 valid-probe LS", COLORS[2]),
            (0, 0, "Legacy native projection", COLORS[0]),
        ):
            ax.plot(time, means[:, nw, model, k], label=label, color=color, lw=0.65)
        ax.set(
            title=f"Network {nw} · same valid target channels",
            xlabel="Time from first saved sample (s)",
            ylabel="Weighted mean (mJy/beam)",
        )
        ax.grid(alpha=0.15)
        if j == 0:
            ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[:, 1])
    positions = np.arange(6)
    for i, (model, k, label, color) in enumerate(
        (
            ("legacy", 0, "Legacy native", COLORS[0]),
            ("v3", 0, "V3 native", COLORS[3]),
            ("v3", 2, "V3 valid-probe LS", COLORS[2]),
            ("v3", 3, "V3 strict population", COLORS[1]),
        )
    ):
        values = [r["rms_mJy_beam"][k] for r in leakage["rows"] if r["model"] == model]
        ax.bar(
            positions + (i - 1.5) * 0.2, values, width=0.19, label=label, color=color
        )
    ax.set(
        xticks=positions,
        xlabel="Network",
        ylabel="Weighted channel RMS (mJy/beam)",
        title="Flagged values contaminate valid targets",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.15)
    save(fig, output, "pca-flag-leakage")
    items = [read_map(Path(r["path"])) for r in result["matched_map_sources"]]
    items, mask, _ = common_maps(items)
    images = [np.where(mask, m["signal"], np.nan) for m in items]
    images.append(images[1] - images[0])
    vmin = min(np.nanmin(a) for a in images[:2])
    vmax = max(np.nanmax(a) for a in images[:2])
    x, y = items[0]["x"], items[0]["y"]
    dx, dy = np.diff(x)[0], np.diff(y)[0]
    extent = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    labels = [
        f"Legacy · RMS {result['matched_map_rms_mJy_beam'][0]:.2f}",
        f"V3 x_opt · RMS {result['matched_map_rms_mJy_beam'][1]:.2f}",
        "V3 − legacy",
    ]
    for k, ax in enumerate(axes):
        limit = np.nanmax(abs(images[2]))
        im = ax.imshow(
            images[k],
            origin="lower",
            extent=extent,
            cmap="RdBu_r" if k == 2 else "viridis",
            vmin=-limit if k == 2 else vmin,
            vmax=limit if k == 2 else vmax,
        )
        ax.set(
            title=labels[k],
            xlim=(-40, 40),
            ylim=(-40, 40),
            xlabel="Azimuth offset (arcsec)",
            ylabel="Elevation offset (arcsec)",
        )
        fig.colorbar(im, ax=ax, label="mJy/beam", shrink=0.8)
    fig.suptitle(
        "152434 a1100 · 1,855 identical channels · equal sample masks and weights"
    )
    save(fig, output, "pca-matched-maps")
    return result, leakage


def update_report(output, result, leakage):
    """Refresh the maintained report section while preserving earlier evidence."""
    path = output / "index.html"
    text = path.read_text()
    for marker, endtag in (
        ('<section id="pca-trace">', "</section>"),
        ('<p id="pca-current-finding">', "</p>"),
    ):
        start = text.find(marker)
        if start >= 0:
            end = text.index(endtag, start) + len(endtag)
            text = text[:start] + text[end:]
    marker = '<section id="model-quality">'
    assert text.count(marker) == 1
    text = text.replace(marker, html_section(result, leakage) + marker)
    text = text.replace("</nav>", "</nav>" + headline(result), 1)
    if 'href="#pca-trace">PCA trace' not in text:
        text = text.replace(
            '<a href="#model-quality">',
            '<a href="#pca-trace">PCA trace</a> · <a href="#model-quality">',
            1,
        )
    gap = output / "gap-control.json"
    if gap.exists():
        from gap_control_report import reorganize_html  # noqa: PLC0415

        text = reorganize_html(text, json.loads(gap.read_text()))
    path.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result, leakage = figures(args.output)
    update_report(args.output, result, leakage)
