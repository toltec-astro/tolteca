#!/usr/bin/env python
"""Report full acquired-array maps, coverage and matched-model controls."""

# ruff: noqa: PLR2004 -- plot layout and diagnostic aperture constants
# ruff: noqa: E501, RUF001 -- embedded scientific report prose and typography

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import matplotlib as mpl

mpl.use("Agg")
import array_diagnostic_plots as diagnostic_plots
import array_example_plots as example_plots
import matplotlib.pyplot as plt
import numpy as np
from array_coordinate_control import html_section as coordinate_section
from array_populations import (
    GROUP_KINDS,
    NONLINEARITY_BINS,
    POPULATIONS,
    apt_audit,
    model_details,
    population_audit,
)
from array_timestream_examples import build as solve_examples
from astropy.table import Table
from compare_maps import compare, read_map
from gap_control_report import headline as gap_headline
from gap_control_report import html_section as gap_section
from gap_control_report import reorganize_html
from netCDF4 import Dataset
from pca_trace_report import headline as pca_headline
from pca_trace_report import html_section as pca_section
from sweep_model_comparison import build as compare_sweeps

ARRAYS = ("a1100", "a1400", "a2000")


def maps(folder, array):
    """Require one exact intensity map for an array."""
    found = list(folder.glob(f"*_{array}_*.fits"))
    if len(found) != 1:
        msg = f"Expected one {array} map in {folder}"
        raise ValueError(msg)
    return read_map(found[0])


def statistics(folder, *, channel_ledger=None):
    """Count contributing detector/chunk pairs from saved native statistics."""
    [path] = folder.glob("*_stats.nc")
    with Dataset(path) as data:
        weights = np.ma.filled(data["weights"][:], np.nan)
        flagged = np.ma.filled(data["flagged_frac"][:], np.nan)
        arrays = np.asarray(data["apt_array"][:], dtype=int)
        networks = np.asarray(data["apt_nw"][:], dtype=int)
        used = (
            np.isfinite(weights) & (weights > 0) & np.isfinite(flagged) & (flagged < 1)
        )
        if "apt_kids_tone" in data.variables:
            channels = np.asarray(data["apt_kids_tone"][:], dtype=int)
        else:
            # The historical engine saves APT UID, but omits its extra columns.
            # Recover channel labels by exact identity, never by row position.
            if channel_ledger is None:
                msg = "Historical statistics require the source APT channel ledger"
                raise ValueError(msg)
            identified = channel_ledger[channel_ledger["apt_flag"] == 0]
            lookup = {
                (int(row["nw"]), int(row["apt_uid"])): int(row["channel"])
                for row in identified
            }
            if len(lookup) != len(identified):
                msg = "Duplicate network/UID identity in the channel ledger"
                raise ValueError(msg)
            uids = np.asarray(data["apt_uid"][:])
            if not np.isfinite(uids).all() or not (uids == uids.astype(np.int64)).all():
                msg = "Historical statistics contain invalid APT identities"
                raise ValueError(msg)
            # Old APTs reuse UID=0 for excluded unmatched rows. They do not
            # contribute; resolve only actually contributing, identified rows.
            channels = np.full(len(uids), -1, dtype=int)
            for i in np.flatnonzero(used.any(axis=0)):
                channels[i] = lookup[int(networks[i]), int(uids[i])]
        rows = []
        for nw in np.unique(networks):
            selected = networks == nw
            values = used[:, selected]
            rows.append(
                {
                    "nw": int(nw),
                    "array": int(arrays[selected][0]),
                    "channels": int(selected.sum()),
                    "contributing_channels": int(values.any(axis=0).sum()),
                    "contributing_in_all_chunks": int(values.all(axis=0).sum()),
                    "contributing_channel_ids": channels[selected][
                        values.any(axis=0)
                    ].tolist(),
                    "contributing_chunk_ids": [
                        np.flatnonzero(values[:, i]).tolist()
                        for i in np.flatnonzero(values.any(axis=0))
                    ],
                    "contributing_channel_chunks": int(values.sum()),
                    "available_channel_chunks": int(values.size),
                    "chunks": len(weights),
                }
            )
    return rows


def validate_contributions(channels, run_stats):
    """Require every reported map contributor to belong to its allowed set."""
    for name, rows in run_stats.items():
        for row in rows:
            ledger = channels[channels["nw"] == row["nw"]]
            allowed = (
                ledger["apt_flag"] == 0 if name == "legacy" else ledger["eligible"]
            )
            unexpected = set(row["contributing_channel_ids"]) - set(
                ledger["channel"][allowed]
            )
            if unexpected:
                msg = f"{name} network {row['nw']} used ineligible channels: {sorted(unexpected)}"
                raise ValueError(msg)


def background(x, y, a, b):
    """Describe map-background structure on an identical covered annulus."""
    inner, outer = 20.0, 40.0
    radius = np.hypot(x[None, :], y[:, None])
    keep = (radius >= inner) & (radius <= outer) & np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        msg = "Insufficient common background pixels"
        raise ValueError(msg)
    return {
        "pixels": int(keep.sum()),
        "inner_arcsec": inner,
        "outer_arcsec": outer,
        "legacy_rms": float(np.std(a[keep], ddof=1)),
        "python_rms": float(np.std(b[keep], ddof=1)),
        "legacy_median": float(np.median(a[keep])),
        "python_median": float(np.median(b[keep])),
        "interpretation": "map-background fluctuations, including correlated residuals; not a noise-only sensitivity or mapping-speed estimate",
    }


def draw(array, legacy, common, physical, output):
    """Draw unchanged maps and residuals with a shared signal color scale."""
    x, y, a, b, metrics = compare(common, physical)
    metrics["background"] = background(x, y, a, b)
    metrics["full_legacy_comparison"] = compare(legacy, physical)[-1]
    metrics["array"] = array
    metrics["sources"] = [
        {
            "path": str(m["path"].resolve()),
            "sha256": hashlib.sha256(m["path"].read_bytes()).hexdigest(),
        }
        for m in (legacy, common, physical)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    axes = axes.ravel()
    central = (abs(y[:, None]) <= 40) & (abs(x[None, :]) <= 40)
    vmin, vmax = signal_limits(legacy, common, physical)
    delta = b - a
    limit = float(np.nanmax(np.abs(delta[central])))
    for ax, data, title in zip(
        axes[:3],
        (legacy, common, physical),
        (
            "Legacy · original APT",
            "Legacy models · common APT",
            "Python models · original APT",
        ),
        strict=True,
    ):
        ix, iy = np.argsort(data["x"]), np.argsort(data["y"])
        xx, yy = data["x"][ix], data["y"][iy]
        signal = data["signal"][np.ix_(iy, ix)].copy()
        cov = data["coverage"][np.ix_(iy, ix)]
        weight = data["weight"][np.ix_(iy, ix)]
        signal[(cov < 0.1 * np.nanmax(cov)) | ~(weight > 0)] = np.nan
        im = ax.imshow(
            signal,
            origin="lower",
            extent=pixel_extent(xx, yy),
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.75, label="mJy/beam")
    im = axes[3].imshow(
        delta,
        origin="lower",
        extent=pixel_extent(x, y),
        vmin=-limit,
        vmax=limit,
        cmap="RdBu_r",
    )
    axes[3].set_title("Python − common legacy", fontsize=11)
    fig.colorbar(im, ax=axes[3], shrink=0.75, label="mJy/beam")
    for ax in axes:
        ax.set(xlim=(-40, 40), ylim=(-40, 40), xlabel="Azimuth offset (arcsec)")
    for ax in axes[::2]:
        ax.set_ylabel("Elevation offset (arcsec)")
    fig.suptitle(
        f"152434 · {array} · all available networks · original science settings"
    )
    fig.savefig(output / f"{array}.png", dpi=170)
    fig.savefig(output / f"{array}.pdf")
    plt.close(fig)
    return metrics


def signal_limits(*items):
    """Retain negative structure using one linear scale for all signal panels."""
    values = []
    for data in items:
        central = (abs(data["y"][:, None]) <= 40) & (abs(data["x"][None, :]) <= 40)
        keep = central & (data["coverage"] >= 0.1 * np.nanmax(data["coverage"]))
        keep &= (data["weight"] > 0) & np.isfinite(data["signal"])
        values.extend(data["signal"][keep])
    return float(np.min(values)), float(np.max(values))


def pixel_extent(x, y):
    """Find image boundaries from actual pixel centers without resampling."""
    dx, dy = (x[1] - x[0]) / 2, (y[1] - y[0]) / 2
    return x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy


def table(headers, rows):
    """Render plain scientific tables with escaped values."""

    def cell(value):
        return html.escape(str(value))

    return (
        "<table><thead><tr>"
        + "".join(f"<th>{cell(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in r) + "</tr>" for r in rows
        )
        + "</tbody></table>"
    )


def file_link(path, label, output):
    """Link a saved artifact without depending on a particular web server."""
    return f'<a href="{quote(os.path.relpath(path, output), safe="/")}">{html.escape(label)}</a>'


def control_comparisons(result, array, legacy_current, legacy, common):
    """Keep engine, eligibility and model changes separately reviewable."""
    controls = ""
    if legacy_current is not None:
        current = maps(legacy_current, array)
        result["engine_control"] = compare(maps(legacy, array), current)[-1]
        result["coverage_control"] = compare(current, maps(common, array))[-1]
        control_rows = []
        for label, values in (
            (
                "Engine only; original APT and legacy coefficients",
                result["engine_control"],
            ),
            (
                "Eligibility only; same v3 engine and legacy coefficients",
                result["coverage_control"],
            ),
            (
                "Model and subsequent cuts; same v3 engine and initial eligibility",
                result,
            ),
        ):
            amp = values["pointing"]["amp"]
            control_rows.append(
                [
                    label,
                    f"{amp['right'] / amp['left']:.4f}",
                    f"{values['signal_correlation']:.5f}",
                    f"{values['difference_rms']:.3f}",
                ]
            )
        controls = (
            "<h3>Separate comparison controls</h3><p>Each row changes the stated part of the reduction. These summary metrics are not additive; cleaning and weights can respond to the changed data and detector set.</p>"
            + table(
                [
                    "Controlled change",
                    "Peak ratio",
                    "Correlation",
                    "Difference RMS (mJy/beam)",
                ],
                control_rows,
            )
        )
    return controls


def model_composition(components, groups, array):
    """Separate fit composition, routing and instantaneous measurement counts."""
    matrix_rows = []
    routed = groups[
        (groups["array"] == array) & (groups["block"] == 1) & groups["routed"]
    ]
    undersampled = routed[routed["n_readout_channels"] < routed["n_resonators"]]
    coverage = {
        "routed_groups": len(routed),
        "groups_with_fewer_probes_than_components": len(undersampled),
        "components_in_these_groups": int(sum(undersampled["n_resonators"])),
    }
    final = components[(components["array"] == array) & (components["block"] == 1)]
    for label, selected in (
        ("Singlet components", final[final["n_resonators"] == 1]),
        ("Multiplet components", final[final["n_resonators"] > 1]),
    ):
        converged = selected["fit_status"] == "optimizer_converged"
        a = np.asarray(selected["a_nl"])[converged]
        a = a[np.isfinite(a)]
        quantiles = (
            "/".join(f"{v:.3f}" for v in np.percentile(a, [10, 50, 90]))
            if len(a)
            else "—"
        )
        matrix_rows.append(
            [
                label,
                len(selected),
                int(sum(converged)),
                int(sum(selected["routed"])),
                quantiles,
            ]
        )
    return matrix_rows, coverage


def population_tables(summary):
    """Explain which rows actually entered the maps, with explicit denominators."""
    populations = summary["populations"]
    composition, nonlinearity, quality, metrics = [], [], [], []
    for name, label in POPULATIONS.items():
        pop = populations[name]
        composition.append(
            [label, pop["channels"], pop["apt_bad"]]
            + [pop["group_kind"].get(k, 0) for k in GROUP_KINDS]
            + [pop["channel_kind"].get("blended", 0)]
        )
        nonlinearity.append(
            [label]
            + [pop["a_nl_bin"].get(k, 0) for k in NONLINEARITY_BINS]
            + [pop["a_nl_bin"].get(k, 0) for k in ("blended", "unresolved")]
        )
        quality.append(
            [label]
            + [
                pop["quality_state"].get(k, 0)
                for k in ("no_flags", "warnings_only", "joint_required", "unusable")
            ]
        )
        metrics.append(
            [label, pop["unique_selected_groups"]]
            + [
                "/".join(f"{v:.4g}" for v in m["median_p90"])
                + f" (n={m['groups_with_finite_value']})"
                for m in pop["group_metrics"].values()
            ]
        )
    flags = sorted({f for p in populations.values() for f in p["flag_bits"]})
    overlap = summary["common_python_overlap"]
    return (
        "<h3>Who actually contributed?</h3>"
        "<p>Each count below is an acquired channel/APT row contributing in at least one chunk. The same final-block Python fit supplies diagnostic labels for every column, including the legacy population. This does not imply that legacy solving used the Python model. Group multiplicity counts all resonators in the selected group; the last column counts probes coupled to several components.</p>"
        + table(
            [
                "Population",
                "Channels",
                "APT bad used",
                *GROUP_KINDS.values(),
                "Blended probes (overlapping count)",
            ],
            composition,
        )
        + "<p>A multi-resonator group can contain a probe coupled to only one component. Such a probe retains its independent scalar calibration. Several probes can also measure one resonator: these channel counts must not be read as independent detector counts. No new component-to-beam identity was established by this comparison.</p>"
        + "<h4>Nonlinearity of the contributing channel population</h4><p>The bins use the Python fitted component's a_nl where the channel has a single-component scalar model. Blended channels have several a_nl values and remain a separate category; they are not assigned their largest or first value. Legacy gainlintrend asymmetry A is not this physical a_nl. All Python fits use the nonlinear model; these bins are descriptive, not underdrive/overdrive acceptance cuts.</p>"
        + table(
            ["Population", *NONLINEARITY_BINS, "Blended", "Unresolved"],
            nonlinearity,
        )
        + "<h4>Fit quality of the contributing population</h4><p>The four states partition each population: unusable takes precedence over joint-required, then warning-only, then zero flags. These are Python diagnostic states for the same channels. A zero flag is not scientific acceptance. This run disables numerical residual and conditioning cuts; warnings, including parameter bounds and initialization fallback, can remain in maps.</p>"
        + table(
            ["Population", "No flags", "Warnings only", "Joint required", "Unusable"],
            quality,
        )
        + "<details><summary>Individual flag counts (bits can overlap)</summary>"
        + table(
            ["Python fit flag", *POPULATIONS.values()],
            [
                [f] + [populations[n]["flag_bits"].get(f, 0) for n in POPULATIONS]
                for f in flags
            ],
        )
        + "</details>"
        + table(
            [
                "Population",
                "Unique selected groups",
                "Residual fraction: median/p90",
                "Reduced chi²: median/p90",
            ],
            metrics,
        )
        + "<p>Fit metrics count each selected group once, even if several contributing channels reference it. Residual fraction is sqrt(Σ|IQ−model|²/Σ|IQ|²) over fit samples; its denominator includes the IQ baseline, not just the resonance excursion. Reduced chi² assumes independent repeat means and IQ quadratures. Neither quantity is a calibrated optical-performance test.</p>"
        + f"<p><strong>Effective-population overlap:</strong> {overlap['both']} channels contribute in both the common-eligibility legacy and Python maps; {overlap['common_only']} contribute only in the former and {overlap['python_only']} only in the latter. Identical initial eligibility does not ensure identical contributing channels, valid samples or weights.</p>"
    )


def figure(name, caption):
    """Link a scientific figure to its full-resolution and vector versions."""
    return (
        f'<figure><a href="{name}.png"><img src="{name}.png" alt="{html.escape(caption)}"></a>'
        f'<figcaption>{html.escape(caption)} <a href="{name}.pdf">PDF</a>.</figcaption></figure>'
    )


def sweep_quality_section(quality):
    """Summarize actual paired fit evidence before interpreting map differences."""
    values = list(quality["arrays"].values())
    good = sum(r["apt_good"] for r in values)
    fitted = sum(r["has_fitted_response"] for r in values)
    converged = sum(r["has_converged_response"] for r in values)
    return (
        '<section id="model-quality"><h2>2. Model quality: legacy tune.txt versus v3 group fits</h2>'
        f"<p><strong>The map gate is not a fit-success count.</strong> Of {good:,} beammap-good channels, {fitted:,} ({fitted / good:.1%}) have a fitted response and {converged:,} ({converged / good:.1%}) have at least one optimizer-converged response. Only 3,096 entered the initial scalar-map eligibility set. The difference includes joint-output handling and instantaneous solving constraints.</p>"
        + figure(
            "model-coverage",
            "Figure 2. Every count is restricted to the original beammap-good channels. A fitted or converged response does not by itself certify scientific adequacy.",
        )
        + figure(
            "model-exclusions",
            "Figure 3. Mutually exclusive reasons for the APT-good to scalar-map gap. Missing fits and probe-window gaps are not evidence that the APT associations are wrong.",
        )
        + "<p>Of the 410 rows previously labeled “fit unusable,” 151 have no fitted response, 104 have a selected optimizer failure, 28 have their probe outside the fitted windows, and 127 have a converged fit but insufficient simultaneous measurements for independent component solving. The separate 1,140 joint-required rows retain fitted models; this comparison has not supplied accepted component-to-beam bindings. These exclusions must not all be reported as failed sweep models.</p>"
        + "<p><strong>Direct sweep test.</strong> Evaluate the original legacy tune.txt equation and the saved full v3 M-channel/N-resonator prediction at the identical data-valid measured frequencies in each saved response window. The v3 model includes a_nl and the complex affine baseline. Recorded IQ accumulation is identical; measured samples are checked against the saved sweep. No refit, gain adjustment, frequency shift or acceptance cut is applied.</p>"
        + figure(
            "sweep-model-quality",
            "Figure 4. APT-good channels with a selected fitted response. Points below the diagonal favor v3. Circles/triangles distinguish one/multiple resonators; colors distinguish downstream map status. The CDFs use the same paired population.",
        )
        + "<p>The denominator is the RMS measured complex-IQ excursion about its mean within the same window, rather than the much larger IQ baseline. This is an in-sample description test: the more flexible v3 model has more parameters. It is not held-out validation, an optical-sensitivity measurement, or proof of unique timestream solving. Channels without a selected fit remain in the coverage ledger rather than disappearing from the denominator.</p>"
        + "<p>The full paired population includes map-ineligible channels. The downloadable response ledger also retains every candidate group/channel association; it does not choose the best residual among competing fits. Remaining scientific work is to validate the response-to-signal transfer and group-to-APT mapping, while investigating the smaller missing/failed-fit population.</p>"
        + '<p><a href="sweep-model-quality.json">Summary and original tune.txt hashes</a> · <a href="sweep-model-responses.ecsv">Every paired response</a> · <a href="sweep-model-channels.ecsv">All channels and exclusion causes</a> · <a href="sweep-example-selection.json">Representative-case selection</a>.</p></section>'
    )


def timestream_section(summary):
    """Connect selected sweep fits to measured, pre-cleaning signal recovery."""
    parts = [
        '<section id="timestream-examples"><h3>2.1 Follow a singlet and a joint model into the timestream</h3>',
        "<p>These are measured 152434 network-0 samples, calibrated by Tune block 1. Selection precedes solving: all member channels must be APT-good, have converged sweep fits, no group warning and excursion-normalized residual below 0.05. Within the M=N=1 and M=N=2 pools, choose the group nearest the median maximum channel residual. This chooses adequate examples without selecting for low timestream noise. APT-good here describes acquired channel rows; it does not establish a new optical identity for each joint component.</p>",
        "<p>The maintained Python fitted-group solver uses all channels simultaneously and retains the complete affine/nonlinear fit. The singlet has two unknown states and the joint group four. Both examples recover all 7,864 samples without flagged failures. This is numerical validity on these data, not an independent test of physical uniqueness. Joint signals below are diagnostics: their component/APT bindings have not been supplied to the saved map trial.</p>",
        "<p>For each component, x′ and x_opt are evaluated at the recorded probe nearest its fitted resonance. The displayed legacy x and r are effective channel coordinates at that probe, not a separated legacy detector solution. Its tune.txt f_in is 6,457.47 Hz below the recorded probe; the comparison preserves that legacy as-used convention and records the offset. Every time trace subtracts only its own median; no amplitude matching is applied.</p>",
        "<p>The same continuous samples enter one-sided Welch spectra (1,024-sample Hann windows, 50% overlap, constant detrending). The shaded 5–20 Hz interval summarizes measured fluctuations before decorrelation, telescope processing and flux calibration. It is not a noise-only NEFD or mapping-speed measurement. The x_opt coordinate and D, μ noise propagation retain the fitted fixed-b model assumption; they are not an independent measurement of instantaneous nonlinearity.</p>",
    ]
    for case in summary["cases"]:
        name, group = case["name"], case["group"]
        parts.append(
            f"<h4>{name.title()} group {group['group_id']}: channels {', '.join(map(str, group['channels']))}</h4>"
        )
        parts.append(
            figure(
                f"timestream-examples/{name}-sweep",
                "Measured sweep and both model predictions for exactly the group followed below. Full joint responses include both resonators at each probe.",
            )
        )
        if case["status"] != "complete":
            parts.append(
                f"<p>No spectral comparison: {html.escape(case['status'])}.</p>"
            )
            continue
        for j, component in enumerate(case["components"]):
            asd = component["band_asd"]
            parts.append(
                figure(
                    f"timestream-examples/{name}-{j}-recovery",
                    f"Component {component['resonator_id']}, reference channel {component['coordinate_channel']}. Recovered detuning/loss, measured PSD and coherence use the same samples. 5–20 Hz ASD: legacy x {asd[0]:.3g}, x′ {asd[1]:.3g}, x_opt {asd[2]:.3g} Hz⁻½; x_opt/x′ = {asd[2] / asd[1]:.3f}. These coordinates have different signal transfer; lower PSD alone does not imply better optical sensitivity.",
                )
            )
            parts.append(
                f'<p><a href="timestream-examples/{name}-{j}-comparison.npz">Time series and spectral matrix</a> · '
                f'<a href="timestream-examples/{name}-{j}-spectra.nc">Signed noise terms</a> · '
                f'<a href="timestream-examples/{name}-solved.npz">Complete group output and flags</a>.</p>'
            )
        if name == "joint":
            parts.append(
                figure(
                    "timestream-examples/joint-noise",
                    "Recovered component correlations and the signed terms D² PSD(x′), μ² PSD(r), 2Dμ Re C(x′,r). Correlated terms can cancel or reinforce; the r contribution must not be treated as independent noise.",
                )
            )
    parts.append(
        '<p><a href="timestream-examples/selection.json">Frozen selection and full group models</a> · <a href="timestream-examples/summary.json">Flags, ASD, frequency conventions and input hashes</a>. Each recovery figure has companion comparison.npz and spectra.nc files in the same directory; solved.npz retains every sample, including flags.</p></section>'
    )
    return "".join(parts)


def map_quality_section():
    """Place the map-quality result and its limits alongside the population summary."""
    return (
        '<section id="map-quality"><h3>1.1 Maps, RMS and sensitivity status</h3>'
        + figure(
            "full-comparison/maps",
            "Full-APT legacy and V3 maps for all three arrays. This comparison changes model eligibility as well as the model. The array sections below also show the common-eligibility legacy control and signed difference maps.",
        )
        + figure(
            "map-quality-summary",
            "Common-eligibility legacy versus V3: measured background RMS and formal median 1/√weight on identical finite, covered 20–40 arcsec annulus pixels. Flux scaling and sky coordinates are unchanged.",
        )
        + '<p><strong>Mapping speed is not validated for these trials.</strong> The saved FITS contain signal, weight and coverage but no noise realizations. Annulus RMS includes correlated residuals and possible source wings; inverse-weight errors assume the adopted noise model. Neither establishes noise-only sensitivity or retained astronomical signal. A mapping-speed comparison needs matched signal transfer and noise depth versus exposure and area; it cannot be inferred from raw detector PSD or contributing-channel count. See <a href="https://doi.org/10.1051/0004-6361/201936220">Perotto et al. (2020), §10.1</a> for the NEFD/mapping-speed convention.</p>'
        + '<p>The existing maps show larger V3 background fluctuations, particularly in a1100 and a1400. The a1400 pointing width reaches its upper fit bound. These are comparison products, not accepted pointing calibration. <a href="map-quality-summary.json">Exact RMS, formal errors and validation status</a>.</p></section>'
    )


def report(  # noqa: PLR0913, PLR0915 -- explicit inputs and sequential report sections
    prepared, legacy, common, physical, output, *, legacy_manifest, legacy_current=None
):
    """Write per-array figures, numerical tables and complete coverage ledgers."""
    output.mkdir(parents=True, exist_ok=True)
    coordinate_path = output / "xprime-control.json"
    coordinate_control = (
        json.loads(coordinate_path.read_text()) if coordinate_path.exists() else None
    )
    gap_path = output / "gap-control.json"
    gap_result = json.loads(gap_path.read_text()) if gap_path.exists() else None
    pca_path = output / "pca-trace.json"
    pca_result = json.loads(pca_path.read_text()) if pca_path.exists() else None
    pca_leakage = (
        json.loads((output / "pca-flag-projection.json").read_text())
        if pca_result
        else None
    )
    receipt = json.loads((prepared / "prepared.json").read_text())
    channels = Table.read(prepared / "channel-coverage.ecsv")
    groups = Table.read(prepared / "group-coverage.ecsv")
    plans = Table.read(prepared / "plan-coverage.ecsv")
    components = Table.read(prepared / "component-coverage.ecsv")
    run_stats = {
        name: statistics(folder, channel_ledger=channels)
        for name, folder in (
            ("legacy", legacy),
            ("common", common),
            ("python", physical),
        )
    }
    validate_contributions(channels, run_stats)
    apts = apt_audit({"legacy": legacy, "common": common, "python": physical}, channels)
    details, model_sources = model_details(receipt)
    population_rows, populations = population_audit(channels, details, run_stats)
    population_rows.meta.update(
        count_unit="acquired channel/APT row, not unique physical detector",
        diagnostic_model="selected Python fit for all populations, including legacy",
        contribution="finite positive saved weight and finite flagged_frac < 1 in at least one chunk",
    )
    population_rows.write(output / "population-ledger.ecsv", overwrite=True)
    (output / "populations.json").write_text(
        json.dumps(
            {"apt": apts, "model_sources": model_sources, "arrays": populations},
            indent=2,
        )
    )
    quality = compare_sweeps(prepared, legacy_manifest, output)
    diagnostic_plots.populations({"arrays": populations}, output)
    diagnostic_plots.coverage(quality, output)
    responses = Table.read(output / "sweep-model-responses.ecsv")
    diagnostic_plots.quality_plots(quality, responses, output)
    diagnostic_plots.examples(prepared, legacy_manifest, responses, output)
    results, sections, overview = [], [], []
    for number, array in enumerate(ARRAYS):
        result = draw(
            array,
            maps(legacy, array),
            maps(common, array),
            maps(physical, array),
            output,
        )
        subset = channels[channels["array"] == number]
        statuses = {
            str(s): int((subset["status"] == s).sum())
            for s in np.unique(subset["status"])
        }
        counts = {}
        for name, rows in run_stats.items():
            counts[name] = sum(
                r["contributing_channels"] for r in rows if r["array"] == number
            )
        result["coverage"] = {
            "channels": len(subset),
            "status": statuses,
            "contributing_channels": counts,
            "contributing_in_all_chunks": {
                name: sum(
                    r["contributing_in_all_chunks"]
                    for r in rows
                    if r["array"] == number
                )
                for name, rows in run_stats.items()
            },
            "mean_contributing_channels_per_chunk": {
                name: sum(
                    r["contributing_channel_chunks"] / r["chunks"]
                    for r in rows
                    if r["array"] == number
                )
                for name, rows in run_stats.items()
            },
        }
        result["population_audit"] = populations[str(number)]
        results.append(result)
        controls = control_comparisons(result, array, legacy_current, legacy, common)
        p = result["pointing"]
        full_p = result["full_legacy_comparison"]["pointing"]
        overview.append(
            [
                array,
                len(subset),
                int(np.sum(subset["apt_flag"] == 0)),
                int(np.sum(subset["eligible"])),
                counts["legacy"],
                counts["common"],
                counts["python"],
                f"{p['amp']['right'] / p['amp']['left']:.3f}",
                f"{result['signal_correlation']:.3f}",
                f"{result['difference_rms']:.2f}",
            ]
        )
        rows = []
        for key, name in [
            ("amp", "Fitted peak (mJy/beam)"),
            ("x_t", "Azimuth offset (arcsec)"),
            ("y_t", "Elevation offset (arcsec)"),
            ("a_fwhm", "Width a (arcsec)"),
            ("b_fwhm", "Width b (arcsec)"),
        ]:
            rows.append(
                [
                    name,
                    f"{full_p[key]['left']:.3f}",
                    f"{p[key]['left']:.3f}",
                    f"{p[key]['right']:.3f}",
                    f"{p[key]['delta']:+.3f}",
                ]
            )
        bg = result["background"]
        rows.append(
            [
                "20–40 arcsec background RMS (mJy/beam)",
                "—",
                f"{bg['legacy_rms']:.3f}",
                f"{bg['python_rms']:.3f}",
                "descriptive",
            ]
        )
        network_rows = []
        for n in receipt["networks"]:
            if n["array"] != number:
                continue
            nw = int(n["interface"].removeprefix("toltec"))
            used = {
                name: next(r["contributing_channels"] for r in records if r["nw"] == nw)
                for name, records in run_stats.items()
            }
            network_rows.append(
                [
                    nw,
                    n["channels"],
                    n["status"].get("original_apt_excluded", 0),
                    n["status"].get("fit_unusable", 0),
                    n["status"].get("joint_apt_binding_missing", 0),
                    n["eligible"],
                    used["common"],
                    used["python"],
                ]
            )
        block_rows = []
        for block in (0, 1):
            selected = groups[(groups["array"] == number) & (groups["block"] == block)]
            planned = plans[(plans["array"] == number) & (plans["block"] == block)]
            singlet = (selected["n_channels"] == 1) & (selected["n_resonators"] == 1)
            block_rows.append(
                [
                    block,
                    len(planned),
                    int(sum(planned["status"] != "eligible")),
                    len(selected),
                    int(sum(selected["fit_status"] == "optimizer_converged")),
                    int(sum(selected["n_resonators"])),
                    int(sum(selected["routed"])),
                    int(sum(singlet)),
                    int(sum(~singlet)),
                ]
            )
        matrix_rows, result["group_readout_coverage"] = model_composition(
            components, groups, number
        )
        sections.append(
            f'<section id="{array}"><h2>{number + 3}. {array}</h2>'
            f'<figure><a href="{array}.png"><img src="{array}.png" alt="{array} complete reduction comparison"></a>'
            f'<figcaption>Figure {5 + 3 * number}. All available networks for {array}. The third panel uses the fitted model stubs; native validity flags exclude unsupported outputs. The fourth compares it with legacy coefficients restricted to the same scalar-map-eligible channel set. No signal rescaling or image registration. <a href="{array}.pdf">PDF</a>.</figcaption></figure>'
            f"<p>Common-pixel correlation: <strong>{result['signal_correlation']:.5f}</strong>; difference RMS: <strong>{result['difference_rms']:.3f} mJy/beam</strong> within 40 arcsec. Background RMS includes correlated residual structure and is not a noise-only sensitivity measurement.</p>"
            + table(
                [
                    "Quantity",
                    "Full legacy",
                    "Common legacy",
                    "Python",
                    "Python − common",
                ],
                rows,
            )
            + "<p>FITS: "
            + " · ".join(
                file_link(Path(s["path"]), label, output)
                for s, label in zip(
                    result["sources"],
                    ("full legacy", "common legacy", "Python"),
                    strict=True,
                )
            )
            + "</p>"
            + controls
            + "<details><summary>Exact counts by network</summary>"
            + "<p>Acquired = APT excluded + unusable flag + joint output not mapped + scalar-map eligible. “Used” means positive saved map weight and at least one unflagged sample in a chunk; later science cuts can reduce this number.</p>"
            + table(
                [
                    "Network",
                    "Acquired",
                    "APT excluded",
                    "Unusable flag",
                    "Joint output not mapped",
                    "Eligible",
                    "Used common",
                    "Used Python",
                ],
                network_rows,
            )
            + "</details>"
            + f"<p>The Python reduction averages {result['coverage']['mean_contributing_channels_per_chunk']['python']:.1f} contributing channels per chunk; {result['coverage']['contributing_in_all_chunks']['python']} contribute in every chunk. Contribution still requires positive weight and at least one unflagged sample, not validity of every sample in that chunk.</p>"
            + figure(
                f"{array}-populations",
                f"Figure {6 + 3 * number}. Actual map contributors. All three populations use the same v3 diagnostic labels; these are channel counts, not independent optical detector counts. Nonlinearity bins are descriptive.",
            )
            + "<details><summary>Exact population counts and quality metrics</summary>"
            + population_tables(populations[str(number)])
            + "</details>"
            + figure(
                f"{array}-sweep-example",
                f"Figure {7 + 3 * number}. A representative two-channel/two-resonator group excluded from scalar maps. Chosen nearest the median paired residual ratio among this class, not as a best-fit example. Lines show saved model predictions on measured frequencies.",
            )
            + "<details><summary>Whole fitted-product census, including both Tune blocks</summary>"
            + "<p>These are model-local components, not independently established optical detector identities. Block 1 calibrates this timestream; block 0 remains in the fitted product. A group with insufficient independent samples remains in the plan, with no fabricated fit. It does not abort neighboring fits.</p>"
            + table(
                [
                    "Block",
                    "Planned groups",
                    "Insufficient data",
                    "Fit groups",
                    "Converged groups",
                    "Components",
                    "Routed groups",
                    "1-probe/1-resonator groups",
                    "Other groups",
                ],
                block_rows,
            )
            + "<h3>Whole final-block model composition (before APT and map selection)</h3><p>This table includes all fitted components, including those excluded from the maps. Every fit uses the nonlinear model. The a_nl percentiles describe converged fitted components; they are not separate solving modes or acceptance thresholds. Routing means a group is present in the stub, not that its components have accepted APT identities or contribute to a map.</p>"
            + table(
                [
                    "Model composition",
                    "Components",
                    "Converged",
                    "Routed",
                    "a_nl 10th/50th/90th percentile",
                ],
                matrix_rows,
            )
            + f"<p>Independently of APT identity, {result['group_readout_coverage']['groups_with_fewer_probes_than_components']} routed groups ({result['group_readout_coverage']['components_in_these_groups']} components) have fewer simultaneous probes than fitted components. Independent loss and frequency recovery for every component therefore lacks sufficient instantaneous measurements. An APT binding alone does not resolve this constraint; the full group ledger retains it separately. Groups with enough probes still require numerical rank and validity checks.</p>"
            + "</details></section>"
        )
    (output / "metrics.json").write_text(json.dumps(results, indent=2))
    (output / "native-coverage.json").write_text(json.dumps(run_stats, indent=2))
    examples = solve_examples(prepared, legacy_manifest, output)
    example_plots.build(prepared, legacy_manifest, output)

    def link(path, label):
        return file_link(path, label, output)

    dependencies = " · ".join(
        [
            link(prepared / "prepared.json", "resolved calibration context"),
            link(Path(receipt["model_index"]), "model index.yaml"),
            link(Path(receipt["stub_index"]), "stub index.yaml"),
            link(prepared / "channel-coverage.ecsv", "all acquired channels"),
            link(prepared / "component-coverage.ecsv", "all fitted components"),
            link(prepared / "plan-coverage.ecsv", "all planned groups"),
            link(prepared / "legacy-common/config.yaml", "common legacy config"),
            link(physical.parents[1] / "config.yaml", "executed Python-model config"),
            link(physical.parents[1] / "citlali.log", "Python-model reduction log"),
            link(physical.parents[1] / "execution.json", "native execution receipt"),
            link(prepared.parent / "fit-affine.yaml", "fitting config"),
            link(prepared.parent / "fit-retry-command.json", "fitting command"),
            link(prepared.parent / "input-hashes.json", "raw input hashes"),
            link(
                prepared.parent / "native-build.json",
                "native source revisions and comparison fixes",
            ),
        ]
    )
    background_changes = "; ".join(
        f"{r['array']}: {r['background']['legacy_rms']:.2f} → "
        f"{r['background']['python_rms']:.2f} mJy/beam"
        for r in results
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>152434 full-array reduction comparison</title>'
        "<style>body{max-width:1400px;margin:32px auto;padding:0 24px;font:17px/1.6 Georgia,serif;color:#20252a}h1,h2,h3,th,nav,figcaption{font-family:system-ui,sans-serif}h1{font-size:30px}h2{border-bottom:1px solid #bbb;padding-bottom:6px;margin-top:50px}a{color:#075b83}img{width:100%}figure{margin:24px 0}figcaption{font-size:14px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #ddd;padding:9px;text-align:left}th{background:#f3f5f5}nav{padding:12px 0;border-block:1px solid #ddd}code{font-size:14px}</style>"
        "<h1>152434: full acquired-array reduction comparison</h1><p>All 11 locally available networks; both Tune blocks; the complete pointing integration. Networks 6 and 10 are absent from the available acquisition. The same affine, nonlinear Python fit configuration is used across arrays.</p>"
        '<nav><a href="#overview">Coverage</a> · <a href="#map-quality">Maps and RMS</a>'
        + (
            '<a href="#coordinate-control"> · x′ map control</a>'
            if coordinate_control
            else ""
        )
        + ' · <a href="#model-quality">Model quality</a> · <a href="#timestream-examples">Singlet and joint timestreams</a> · <a href="#a1100">a1100</a> · <a href="#a1400">a1400</a> · <a href="#a2000">a2000</a> · <a href="#provenance">Products and interpretation</a></nav>'
        + (
            gap_headline(gap_result)
            if gap_result
            else pca_headline(pca_result)
            if pca_result
            else ""
        )
        + '<h2 id="overview">1. Whole-observation accounting</h2><p>The full legacy and Python-model runs use the byte-identical beammap APT. Only rows with flag = 0 can enter the reduction. Python model validity excludes additional rows. The common-eligibility legacy control changes only the APT flag column to apply those additional exclusions; positions, flux conversion, column units and metadata are unchanged. Raw data, telescope coordinates, pointing correction, cleaning and native weighting settings are retained. No zero-model-flag-only policy or network-0-only selection is imposed.</p>'
        + figure(
            "population-flow",
            "Figure 1. Original beammap APT, additional scalar-map restrictions and actual contributions. The unusable flag includes fitting, probe selection and solving constraints; Figure 3 separates them.",
        )
        + "<details><summary>Exact APT identities and population counts</summary>"
        + table(
            ["Run", "APT file", "Rows / flag = 0", "SHA-256"],
            [
                [
                    POPULATIONS[n],
                    Path(a["path"]).name,
                    f"{a['rows']} / {a['flag_zero']}",
                    a["sha256"],
                ]
                for n, a in apts["sources"].items()
            ],
        )
        + '<p><a href="populations.json">APT paths, hashes, thresholds and population summaries</a> · <a href="population-ledger.ecsv">Every channel, APT flag, fit group, a_nl, quality flags and exact contributing chunks</a>.</p>'
        + table(
            [
                "Array",
                "Acquired channels",
                "Beammap APT good",
                "Scalar-map eligible",
                "Full legacy contributing",
                "Common-eligibility legacy contributing",
                "Python contributing",
                "Peak ratio to common legacy",
                "Correlation with common legacy",
                "Difference RMS (mJy/beam)",
            ],
            overview,
        )
        + "</details>"
        + f"<p><strong>Map quality:</strong> the common-annulus background RMS changes as follows: {background_changes}. These fluctuations include residual structure. Completing the reduction does not establish a scientific improvement, and the fitted PPT is retained for comparison rather than adopted as a pointing calibration.</p>"
        + '<p><a href="full-comparison/maps.png">Three-array legacy/Python map comparison</a> · <a href="full-comparison/maps.pdf">PDF</a>. The array sections below also show the common-APT control, so a change of detector selection is visible separately from a change of model.</p>'
        + "<p><strong>Scope of joint recovery:</strong> all joint fits and coefficients are retained, but the historical APT has no accepted model-component-to-beam associations for these new fits. Outputs requiring those bindings are excluded by the native solver flags and shown separately as identity gaps. This is not a validation of full multiplet map recovery.</p>"
        + map_quality_section()
        + (coordinate_section(coordinate_control) if coordinate_control else "")
        + (pca_section(pca_result, pca_leakage) if pca_result else "")
        + (gap_section(gap_result) if gap_result else "")
        + sweep_quality_section(quality)
        + timestream_section(examples)
        + "".join(sections)
        + '<h2 id="provenance">6. Products and interpretation</h2><p>'
        + dependencies
        + '</p><p><a href="metrics.json">Full numerical metrics</a> · <a href="native-coverage.json">Saved native weight/flag accounting</a>.</p>'
        + "<p>The first full-array attempt exposed a PCA division by zero when one channel retained only one valid sample in a chunk. The comparison build rejects that channel/chunk before cleaning and guards covariance pairs without two shared samples. It also omits joint-output computation when the APT has no component bindings; channel results and flags are unchanged. All seven previously completed control map files are numerically identical after these fixes. Source patches, executable hashes and regression evidence are retained with the comparison.</p>"
        + "<p>The original legacy control is the local development executable with the isolated packet-counter bounds correction documented in the earlier comparison. These are fresh full-array trials with iterative source feedback disabled in both versions. Both common-APT and Python-model reductions use the same pinned v3 engine. The full-legacy/Python comparison changes both model and eligibility; the common-APT control isolates that eligibility change, subject to each reduction's subsequent science cuts. Absolute flux calibration is held fixed; brighter peaks or lower background RMS alone do not establish improved optical sensitivity. The native consumer uses the full selected model schema and keeps sample flags; unresolved Python/native parity remains a separate release check.</p>"
        + "<p>Figures compare unchanged sky-coordinate grids and a common coverage threshold. Difference RMS includes source and background changes. “Contributing channels” is derived from native chunk weights and flagged fractions; it is not a count of independent optical detectors or a mapping-speed estimate.</p></html>"
    )
    if gap_result:
        path = output / "index.html"
        path.write_text(reorganize_html(path.read_text(), gap_result))


def main():
    """Render the report after all reductions complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "legacy", "common", "physical", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument(
        "--legacy-manifest",
        type=Path,
        required=True,
        help="Engine-control manifest naming the original tune.txt files.",
    )
    parser.add_argument(
        "--legacy-current",
        type=Path,
        help="Same v3 engine with original APT and legacy coefficients; separates engine and eligibility effects.",
    )
    report(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
