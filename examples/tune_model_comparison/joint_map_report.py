"""Audit and display the broader joint-component maps against scalar controls."""

# ruff: noqa: INP001, E501, T201 -- scientific report and reproducible local evidence

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from array_coordinate_control import ARRAYS, common_maps, save
from array_populations import a_bin
from astropy.table import Table
from build_xprime_control import digest
from compare_maps import read_map
from netCDF4 import Dataset

ORDER = ("scalar-numerical", "joint-numerical", "scalar-physical", "joint-physical")
ALL_ORDER = ("legacy-broader", *ORDER)
BROAD_ORDER = ("legacy-broader", "joint-numerical", "joint-physical")
BROAD_LABELS = ("Legacy reference", "V3 numerical", "V3 physical")
LABELS = (
    "Scalar numerical",
    "Broader numerical",
    "Scalar physical",
    "Broader physical",
)
COLORS = ("#a2b0b9", "#237d98", "#c5b5d2", "#75449b")


def completed_run(folder):
    """Require exactly one successful realization; retain failed attempt receipts."""
    paths = [
        p.parent
        for p in folder.glob("redu*/execution.json")
        if json.loads(p.read_text())["exit_code"] == 0
    ]
    if len(paths) != 1:
        message = f"Expected one successful realization in {folder}, found {len(paths)}"
        raise ValueError(message)
    return paths[0]


def population_audit(base, apt):
    """Count APT rows separately from fitted components and retain missing evidence."""
    old = Table.read(base / "prepared/channel-coverage.ecsv")
    components = json.loads(
        (base / "joint-control/component-bindings.json").read_text()
    )
    added = {
        (r["nw"], r["channel"]): r for r in components if r["status"] == "joint_added"
    }
    rows = []
    for channel in old:
        key = (int(channel["nw"]), int(channel["channel"]))
        status = str(channel["status"])
        if key in added:
            status = "joint_added"
        elif status == "joint_apt_binding_missing":
            matches = [
                r
                for r in components
                if r["nw"] == key[0] and r.get("channel") == key[1]
            ]
            reasons = {r["status"] for r in matches}
            status = (
                next(iter(reasons))
                if len(reasons) == 1
                else "no_unique_component_binding"
            )
        rows.append(
            {
                "nw": key[0],
                "channel": key[1],
                "array": int(channel["array"]),
                "uid": int(channel["apt_uid"]),
                "status": status,
            }
        )
    np.testing.assert_equal(
        [(int(r["nw"]), int(r["kids_tone"])) for r in apt],
        [(r["nw"], r["channel"]) for r in rows],
    )
    Table(rows=rows).write(
        base / "joint-control/channel-disposition.ecsv", overwrite=True
    )
    return rows


def audit(base):  # noqa: C901, PLR0912, PLR0915 -- sequential checks on native artifacts
    """Check the exact binding, weights, masks and full-integration map products."""
    apt_path = base / "joint-control/joint-apt.ecsv"
    apt = Table.read(apt_path)
    common = Table.read(base / "prepared/common-apt.ecsv")
    selection = json.loads((base / "joint-control/selection.json").read_text())
    original = Table.read(selection["original_apt"])
    for name in original.colnames:
        if name not in ("flag", "tone_freq", "uid"):
            np.testing.assert_equal(apt[name], original[name])
    np.testing.assert_equal(apt["uid"], original["uid"])
    eligible = np.asarray(apt["flag"]) == 0
    scalar = np.asarray(common["flag"]) == 0
    added = eligible & ~scalar
    assert np.all(eligible <= (np.asarray(original["flag"]) == 0))
    assert np.all(eligible[scalar])
    np.testing.assert_equal(apt["tone_freq"][scalar], common["tone_freq"][scalar])
    disposition = population_audit(base, apt)
    arrays = np.asarray(apt["array"], int)
    component_rows = {}
    bindings = json.loads((base / "joint-control/component-bindings.json").read_text())
    groups = {
        (n["nw"], g["group_id"]): g
        for n in yaml.safe_load(apt.meta["kids_readout"])["networks"]
        for g in n["groups"]
    }
    for binding in bindings:
        if binding["status"] != "joint_added":
            continue
        row = dict(binding)
        group = groups[(row["nw"], row["group_id"])]
        index = group["resonator_ids"].index(row["resonator_id"])
        row.update(
            a_nl=group["a_nl"][index],
            a_nl_bin=a_bin(group["a_nl"][index]),
            group_fit_flag=group["fit_flag"],
        )
        component_rows[(row["nw"], row["channel"])] = row
    runs, maps, weights, scalar_flags, scalar_signals, all_flags = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    added_masks = {}
    folders = {
        n: base
        / (
            "gap-control/v3-" + n.split("-")[1]
            if n.startswith("scalar")
            else "joint-control/" + n.split("-")[1]
        )
        for n in ORDER
    }
    folders["legacy-broader"] = base / "joint-control/legacy"
    for name, folder in folders.items():
        run = completed_run(folder)
        print(f"Auditing {name}: {run}", flush=True)
        [ptc_path] = (run / "152434/raw").glob("*_ptc_timestream.nc")
        [rtc_path] = (run / "152434/raw").glob("*_rtc_timestream.nc")
        with Dataset(ptc_path) as post, Dataset(rtc_path) as pre:
            np.testing.assert_equal(post["apt_nw"][:], apt["nw"])
            np.testing.assert_equal(post["apt_kids_tone"][:], apt["kids_tone"])
            flags = np.asarray(post["flags"][:], bool)
            assert not np.any(np.asarray(pre["flags"][:], bool) & ~flags)
            sf = np.asarray(post["kids_sample_flag"][:], int)
            y = np.asarray(post["signal"][:])
            assert np.isfinite(y).all()
            exclusion = int(post["kids_sample_flag"].map_exclusion_mask)
            assert not np.any(((sf & exclusion) != 0) & ~flags)
            assert (
                yaml.safe_load(post.kids_readout)["filled_samples_map_eligible"]
                is False
            )
            weights[name] = np.asarray(post["weights"][:])
            # Read each netCDF variable once; sparse column indexing otherwise
            # issues thousands of independent backend reads.
            scalar_signals[name] = np.asarray(pre["signal"][:])[:, scalar]
            scalar_flags[name] = np.asarray(pre["flags"][:], bool)[:, scalar]
            all_flags[name] = sf[:, added]
            added_masks[name] = flags[:, added]
            if name.startswith("joint"):
                policy = name.split("-")[1]
                for i in np.flatnonzero(added):
                    row = component_rows[(int(apt["nw"][i]), int(apt["kids_tone"][i]))]
                    row[policy + "_samples"] = len(flags)
                    row[policy + "_accepted_samples"] = int((~flags[:, i]).sum())
                    row[policy + "_solver_flag_union"] = int(
                        np.bitwise_or.reduce(sf[:, i])
                    )
                    row[policy + "_solver_excluded_samples"] = int(
                        ((sf[:, i] & exclusion) != 0).sum()
                    )
            counts = []
            for a, array in enumerate(ARRAYS):
                take = arrays == a
                joint = take & added
                counts.append(
                    {
                        "array": array,
                        "eligible": int(
                            (
                                take
                                & (scalar if name.startswith("scalar") else eligible)
                            ).sum()
                        ),
                        "scalar_contributors": int(
                            (~flags[:, take & scalar]).any(axis=0).sum()
                        ),
                        "added_row_contributors": int(
                            (~flags[:, joint]).any(axis=0).sum()
                        ),
                        "contributors": int((~flags[:, take]).any(axis=0).sum()),
                        "accepted_samples": int((~flags[:, take]).sum()),
                        "added_row_accepted_samples": int((~flags[:, joint]).sum()),
                        "added_row_solver_flag_counts": {
                            str(bit): int(((sf[:, joint] & bit) != 0).sum())
                            for bit in (
                                1,
                                2,
                                4,
                                8,
                                16,
                                32,
                                64,
                                128,
                                256,
                                512,
                                1024,
                                2048,
                            )
                        },
                        "added_rows_entirely_rejected": int(
                            flags[:, joint].all(axis=0).sum()
                        ),
                    }
                )
            runs[name] = {
                "path": str(run),
                "n_samples": len(flags),
                "policy": post["kids_sample_flag"].sample_acceptance,
                "arrays": counts,
                "sources": {
                    key: {"path": str(path), "sha256": digest(path)}
                    for key, path in [
                        ("ptc", ptc_path),
                        ("rtc", rtc_path),
                        ("execution", run / "execution.json"),
                    ]
                },
            }
        maps[name] = []
        for array in ARRAYS:
            [path] = (run / "152434/raw").glob(f"*_{array}_*.fits")
            maps[name].append(read_map(path))
    Table(rows=list(component_rows.values())).write(
        base / "joint-control/joint-contributors.ecsv", overwrite=True
    )
    checks = {}
    for policy in ("numerical", "physical"):
        left, right = f"scalar-{policy}", f"joint-{policy}"
        configs = []
        for name in (left, right):
            config = copy.deepcopy(
                yaml.safe_load((Path(runs[name]["path"]) / "config.yaml").read_text())
            )
            config["runtime"].pop("output_dir")
            for item in config["inputs"][0]["cal_items"]:
                if item["type"] == "array_prop_table":
                    item["filepath"] = "APT"
            configs.append(config)
        assert configs[0] == configs[1]
        left_weights, right_weights = (
            weights[left][:, scalar],
            weights[right][:, scalar],
        )
        changed_weights = np.any(left_weights != right_weights, axis=0)
        relative_weights = (
            np.divide(
                right_weights,
                left_weights,
                out=np.ones_like(left_weights),
                where=left_weights != 0,
            )
            - 1
        )
        valid = ~(scalar_flags[left] | scalar_flags[right])
        delta = np.abs(scalar_signals[left] - scalar_signals[right])
        np.testing.assert_allclose(
            scalar_signals[left][valid],
            scalar_signals[right][valid],
            atol=3e-7,
            rtol=1e-9,
        )
        checks[policy] = {
            "same_configuration_except_apt_and_output": True,
            "scalar_weights_bitwise_equal": bool(not changed_weights.any()),
            "scalar_weights_changed": int(changed_weights.sum()),
            "scalar_weight_max_relative_change": float(
                np.max(np.abs(relative_weights))
            ),
            "changed_scalar_weight_rows": [
                {
                    "nw": int(apt["nw"][scalar][i]),
                    "channel": int(apt["kids_tone"][scalar][i]),
                }
                for i in np.flatnonzero(changed_weights)
            ],
            "weight_rule": "Same sensitivity weights and median outlier rule; the median can change with the eligible population.",
            "scalar_valid_precleaning_max_abs_mJy_beam": float(delta[valid].max()),
            "scalar_compared_samples": int(valid.sum()),
        }
    for name in ("joint-numerical", "joint-physical"):
        np.testing.assert_equal(weights["legacy-broader"], weights[name])
    checks["broader_legacy_v3_weights_bitwise_equal"] = True
    numerical_flags, physical_flags = (
        all_flags["joint-numerical"],
        all_flags["joint-physical"],
    )
    # The projection adds nonfinite_output only when policy flags would
    # otherwise admit a NaN. Physical acceptance already excludes these
    # outside-domain samples. The original solving flags must still agree.
    np.testing.assert_equal(numerical_flags & ~4, physical_flags & ~4)
    derived = numerical_flags != physical_flags
    np.testing.assert_equal(numerical_flags[derived], physical_flags[derived] | 4)
    assert np.all((physical_flags[derived] & (256 | 2048)) != 0)
    assert np.all(added_masks["joint-numerical"][derived])
    assert np.all(added_masks["joint-physical"][derived])
    checks["joint_primary_solver_flags_identical"] = True
    checks["joint_additional_nonfinite_projection_flags"] = int(derived.sum())
    checks["projection_flag_difference_samples_excluded_by_both_policies"] = True
    metrics = []
    for a, array in enumerate(ARRAYS):
        aligned, _, annulus = common_maps([maps[n][a] for n in ALL_ORDER])
        take = arrays == a
        metrics.append(
            {
                "array": array,
                "original_apt_good": int(
                    (take & (np.asarray(original["flag"]) == 0)).sum()
                ),
                "scalar_eligible": int((take & scalar).sum()),
                "joint_added": int((take & added).sum()),
                "broader_eligible": int((take & eligible).sum()),
                "rms_mJy_beam": {
                    n: float(np.std(m["signal"][annulus]))
                    for n, m in zip(ALL_ORDER, aligned, strict=True)
                },
                "common_annulus_pixels": int(annulus.sum()),
                "disposition": dict(
                    Counter(r["status"] for r in disposition if r["array"] == a)
                ),
            }
        )
    result = {
        "order": list(ORDER),
        "broader_order": list(BROAD_ORDER),
        "arrays": metrics,
        "runs": runs,
        "checks": checks,
        "joint_solver_flags_identical_between_policies": bool(not derived.any()),
        "apt": {"path": str(apt_path), "sha256": digest(apt_path)},
        "selection": selection,
        "limits": "Existing unique acquisition-channel/APT correspondence; no independent component beam or flux recalibration. Joint state covariance is retained but the map accumulator uses per-row weights. Annulus RMS is not validated sensitivity or mapping speed.",
    }
    (base / "report/joint-control.json").write_text(json.dumps(result, indent=2) + "\n")
    return result, maps


def figures(base, result, maps):  # noqa: C901, PLR0912, PLR0915 -- related scientific panels
    """Show coverage and maps on shared scales, without intensity rescaling."""
    output = base / "report"
    pos = np.arange(3)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.8), layout="constrained")
    for i, (name, label, color) in enumerate(zip(ORDER, LABELS, COLORS, strict=True)):
        bars = axes[0].bar(
            pos + (i - 1.5) * 0.21,
            [r["rms_mJy_beam"][name] for r in result["arrays"]],
            width=0.2,
            color=color,
            label=label,
        )
        axes[0].bar_label(bars, fmt="%.1f", rotation=90, padding=3, fontsize=9)
    axes[0].set(
        ylabel="Annulus RMS (mJy/beam)",
        title="Map fluctuations on identical covered pixels",
    )
    axes[0].set_ylim(
        0, max(v for r in result["arrays"] for v in r["rms_mJy_beam"].values()) * 1.3
    )
    axes[0].legend(frameon=False, fontsize=9)
    scalar = [r["scalar_eligible"] for r in result["arrays"]]
    added = [r["joint_added"] for r in result["arrays"]]
    axes[1].bar(pos, scalar, color=COLORS[0], label="Scalar eligible")
    axes[1].bar(
        pos, added, bottom=scalar, color=COLORS[1], label="Added joint components"
    )
    for i, (a, b) in enumerate(zip(scalar, added, strict=True)):
        axes[1].text(i, a / 2, f"{a:,}", ha="center", va="center")
        axes[1].text(i, a + b / 2, f"+{b}", ha="center", va="center", color="white")
    axes[1].set(
        ylabel="Eligible APT rows",
        title="Same beammap-good gate; broader fitted population",
        ylim=(0, 3100),
    )
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set(xticks=pos, xticklabels=ARRAYS)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    save(fig, output, "joint-map-summary")
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), layout="constrained")
    for a, array in enumerate(ARRAYS):
        aligned, covered, _ = common_maps([maps[n][a] for n in ORDER])
        limit = np.percentile(
            np.abs(np.concatenate([m["signal"][covered] for m in aligned])), 99
        )
        for k, (name, m) in enumerate(zip(ORDER, aligned, strict=True)):
            ax = axes[a, k]
            data = np.where(covered, m["signal"], np.nan)
            im = ax.pcolormesh(
                m["x"],
                m["y"],
                data,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="auto",
                rasterized=True,
            )
            counts = result["runs"][name]["arrays"][a]
            ax.set(
                xlim=(-40, 40),
                ylim=(-40, 40),
                aspect="equal",
                title=f"{LABELS[k]}\n{counts['contributors']:,} contributors",
            )
            if k == 0:
                ax.set_ylabel(f"{array}\nElevation offset (arcsec)")
            if a == len(ARRAYS) - 1:
                ax.set_xlabel("Azimuth offset (arcsec)")
        fig.colorbar(im, ax=axes[a, :], shrink=0.8, label="mJy/beam")
    save(fig, output, "joint-maps")
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), layout="constrained")
    for a, array in enumerate(ARRAYS):
        aligned, covered, _ = common_maps([maps[n][a] for n in BROAD_ORDER])
        limit = np.percentile(
            np.abs(np.concatenate([m["signal"][covered] for m in aligned])), 99
        )
        for ax, name, label, m in zip(
            axes[a], BROAD_ORDER, BROAD_LABELS, aligned, strict=True
        ):
            im = ax.pcolormesh(
                m["x"],
                m["y"],
                np.where(covered, m["signal"], np.nan),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="auto",
                rasterized=True,
            )
            count = result["runs"][name]["arrays"][a]["contributors"]
            ax.set(
                xlim=(-40, 40),
                ylim=(-40, 40),
                aspect="equal",
                title=f"{label}\n{count:,} contributors",
            )
            if ax is axes[a, 0]:
                ax.set_ylabel(f"{array}\nElevation offset (arcsec)")
            if a == len(ARRAYS) - 1:
                ax.set_xlabel("Azimuth offset (arcsec)")
        fig.colorbar(im, ax=axes[a], shrink=0.8, label="mJy/beam")
    save(fig, output, "joint-legacy-maps")
    for a, array in enumerate(ARRAYS):
        fig, axes = plt.subplots(1, 3, figsize=(8, 3.2), layout="constrained")
        aligned, covered, _ = common_maps([maps[n][a] for n in BROAD_ORDER])
        limit = np.percentile(
            np.abs(np.concatenate([m["signal"][covered] for m in aligned])), 99
        )
        for ax, _name, label, m in zip(
            axes, BROAD_ORDER, BROAD_LABELS, aligned, strict=True
        ):
            im = ax.pcolormesh(
                m["x"],
                m["y"],
                np.where(covered, m["signal"], np.nan),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="auto",
                rasterized=True,
            )
            ax.set(
                xlim=(-40, 40),
                ylim=(-40, 40),
                aspect="equal",
                title=label,
                xlabel="Azimuth offset (arcsec)",
            )
            ax.title.set_size(10)
        axes[0].set_ylabel("Elevation offset (arcsec)")
        fig.colorbar(im, ax=axes, shrink=0.8, label=f"{array} (mJy/beam)")
        save(fig, output, f"joint-legacy-{array}")
    for a, array in enumerate(ARRAYS):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), layout="constrained")
        aligned, covered, _ = common_maps([maps[n][a] for n in ORDER])
        limit = np.percentile(
            np.abs(np.concatenate([m["signal"][covered] for m in aligned])), 99
        )
        for ax, name, label, m in zip(axes.flat, ORDER, LABELS, aligned, strict=True):
            im = ax.pcolormesh(
                m["x"],
                m["y"],
                np.where(covered, m["signal"], np.nan),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="auto",
                rasterized=True,
            )
            counts = result["runs"][name]["arrays"][a]
            ax.set(
                xlim=(-40, 40),
                ylim=(-40, 40),
                aspect="equal",
                title=f"{label}: {counts['contributors']:,} contributors",
                xlabel="Azimuth offset (arcsec)",
                ylabel="Elevation offset (arcsec)",
            )
            ax.title.set_size(10)
        fig.colorbar(im, ax=axes, shrink=0.8, label=f"{array} (mJy/beam)")
        save(fig, output, f"joint-maps-{array}")


def html_section(result):
    """Put the broader comparison first and state the scientific limits."""
    rows = "".join(
        "<tr><td>"
        + r["array"]
        + "</td><td>"
        + str(r["joint_added"])
        + "</td>"
        + "".join(f"<td>{r['rms_mJy_beam'][n]:.2f}</td>" for n in ORDER)
        + "</tr>"
        for r in result["arrays"]
    )
    legacy_rows = "".join(
        "<tr><td>"
        + r["array"]
        + "</td>"
        + "".join(f"<td>{r['rms_mJy_beam'][name]:.2f}</td>" for name in BROAD_ORDER)
        + "</tr>"
        for r in result["arrays"]
    )
    changes = "; ".join(
        r["array"]
        + ": "
        + ", ".join(
            f"{policy} {100 * (r['rms_mJy_beam'][f'joint-{policy}'] / r['rms_mJy_beam'][f'scalar-{policy}'] - 1):+.1f}%"
            for policy in ("numerical", "physical")
        )
        for r in result["arrays"]
    )
    contributing_added = sum(
        r["added_row_contributors"] for r in result["runs"]["joint-physical"]["arrays"]
    )
    return (
        '<section id="joint-maps"><h2>Broader population: jointly fitted components in the maps</h2>'
        f"<p><strong>The full 152434 reduction now includes {contributing_added:,} additional joint components with valid map contributions.</strong> The figures compare the broader V3 maps against the legacy-model reference and the original scalar controls. Eligibility, actual contributions and sample failures are counted separately below.</p>"
        "<p>The final Tune-block Python fits and acquisition model stubs are unchanged. Each added component follows the saved ModelFit plan → KidsFind finding → unique saved channel match → original observation APT UID. The native solver evaluates the same nonlinear group model; the adapter selects its component response at the recorded probe frequency. It does not substitute a single-resonator fit.</p>"
        "<p>Both V3 runs return optical detuning from the same nonlinear model. Numerical and physical name sample-acceptance policies: numerical keeps finite domain/stability warnings; physical additionally excludes those samples. Both reject hard solving failures and keep filled samples out of the maps. Scalar controls mean the earlier independent-channel route, not a population composed exclusively of singlets.</p>"
        f"<p>The full-integration audit confirms identical underlying joint-solving flags. Projection adds a nonfinite-output bit to {result['checks']['joint_additional_nonfinite_projection_flags']:,} numerical-policy samples already outside the allowed domain; all are excluded under both policies. Numerical acceptance does not recover a joint state that the bounded solver did not return.</p>"
        "<p>The broader APT retains all 3,096 scalar-control rows and adds 1,045 supported joint components: 4,141 eligible rows out of 4,646 beammap-good rows. Beam positions, flux factors and APT sensitivities are inherited unchanged. The Citlali population-median outlier rule is retained; it changes three scalar weights in a1400 by about 2%, recorded in the audit. The original controls remain below; their 3,096-row count does not describe this broader run.</p>"
        "<h3>Broader population: legacy and V3</h3><p>The legacy-model reference uses the corrected engine with the same broader APT eligibility, beam and sensitivity values. Its final per-row weights match both broader V3 policies exactly. Legacy outputs are effective per-channel responses; V3 joint outputs are component responses. Actual contributions can differ because failed solutions remain rejected and the scalar proximity rule still applies to legacy channels.</p>"
        '<figure><a href="joint-legacy-maps.pdf"><img src="joint-legacy-maps.png" alt="Legacy and V3 maps on the broader eligible population"></a><figcaption>Legacy reference, V3 numerical and V3 physical on all three arrays. Each array shares one intensity scale; panel counts are actual contributors.</figcaption></figure>'
        "<table><thead><tr><th>Array</th><th>Broader legacy RMS</th><th>Broader V3 numerical RMS</th><th>Broader V3 physical RMS</th></tr></thead><tbody>"
        + legacy_rows
        + "</tbody></table><h3>Effect of adding joint components to V3</h3>"
        + f"<p><strong>Measured RMS change relative to each scalar control:</strong> {changes}. These are changes in map fluctuations, not validated sensitivity gains.</p>"
        '<figure><img src="joint-map-summary.png" alt="Broader population and matched-policy map RMS"><figcaption>Broader and scalar controls at each sample-acceptance policy. Population bars show eligibility, while map panels give actual contributors.</figcaption></figure>'
        "<table><thead><tr><th>Array</th><th>Added components</th><th>Scalar numerical</th><th>Broader numerical</th><th>Scalar physical</th><th>Broader physical</th></tr></thead><tbody>"
        + rows
        + "</tbody></table>"
        '<figure><a href="joint-maps.pdf"><img src="joint-maps.png" alt="Four controls across all three arrays"></a><figcaption>The full integration, with the same intensity scale within each array and identical covered pixels. RMS uses the common 20-40 arcsec annulus. Adding detectors changes the PCA basis and coverage; no image registration or signal rescaling is applied.</figcaption></figure>'
        "<p><strong>Interpretation:</strong> this tests broader model consumption using existing acquisition-to-APT correspondences. It does not independently validate component beam positions or flux response. The solver retains joint covariance; the map accumulator still uses per-row weights, so correlated components are not independent sensitivity gains. Sample failures remain rejected. Annulus RMS alone is not mapping speed or validated optical sensitivity. The previously reported measured Python/native parity discrepancy also remains a release limitation.</p>"
        '<p><a href="joint-control.json">Native checks, exact counts and hashes</a> · <a href="../joint-control/channel-disposition.ecsv">Every acquired APT row</a> · <a href="../joint-control/component-bindings.ecsv">Every fitted multiplet component</a> · <a href="../joint-control/joint-contributors.ecsv">Added components, nonlinearity and actual sample acceptance</a> · <a href="../joint-control/selection.json">Selection receipt</a></p></section>'
    )


def integrate_html(text, broader):
    """Keep the broader evidence first when any older report section is rebuilt."""
    text = re.sub(r'<section id="joint-maps">.*?</section>', "", text, flags=re.DOTALL)
    text = text.replace('<a href="#joint-maps">Broader joint maps</a> · ', "")
    text = text.replace(
        "The separate 1,140 joint-required rows retain fitted models; this comparison has not supplied accepted component-to-beam bindings.",
        "The original scalar selection excluded 1,140 joint-required rows despite retaining fitted models. The broader comparison now admits 1,045 through saved component/channel/APT correspondences; 95 remain without a unique supported binding.",
    ).replace(
        "Joint signals below are diagnostics: their component/APT bindings have not been supplied to the saved map trial.",
        "These saved Python diagnostic examples predate the broader maps. The new binding ledger identifies exactly which components now enter map making through the existing acquisition/APT correspondence.",
    )
    text = text.replace("Current policy comparison", "Scalar policy controls").replace(
        "Current performance: numerical versus physical acceptance",
        "Scalar controls: numerical versus physical acceptance",
    )
    text = (
        text.replace(
            "Current acceptance accounting", "Scalar-control acceptance accounting"
        )
        .replace("the current three-run comparison", "the corrected scalar controls")
        .replace(
            "Current contributing channel/APT rows",
            "Scalar-control contributing channel/APT rows",
        )
        .replace(
            "The current build, configs, source hashes and measured checks are linked in",
            "The scalar-control build, configs, source hashes and measured checks are linked in",
        )
    )
    text = text.replace(
        "Multiplet component-to-beam/APT identities, independent ",
        "Remaining unmatched component identities, independent ",
    )
    text = text.replace(
        "They do not restore the missing joint-component",
        "Those scalar controls do not restore the missing joint-component",
    )
    text = text.replace(
        "<nav>", '<nav><a href="#joint-maps">Broader joint maps</a> · ', 1
    )
    text = (
        text.replace(
            "The corrected runs remove the earlier large a1100 map excess.",
            "The corrected scalar controls removed the earlier large a1100 map excess. The broader maps above additionally test joint components and have their own measured fluctuations.",
        )
        .replace(
            "This review separates the current corrected maps, the fitted-model evidence, and the earlier investigation.",
            "This section retains the corrected scalar controls used before adding joint components.",
        )
        .replace(
            "Current summary and exact denominators",
            "Scalar-control summary and exact denominators",
        )
    )
    text = text.replace(
        "Current result: gap correction and explicit acceptance",
        "Scalar control: gap correction and explicit acceptance",
    )
    text = text.replace(
        "Full intended population", "Scalar-control intended population"
    )
    text = text.replace(
        "the comparison APT has no accepted component-to-beam",
        "the scalar-control APT has no component-to-beam",
    )
    text = text.replace("</nav>", "</nav>" + broader, 1)
    number = 0

    def caption(match):
        nonlocal number
        number += 1
        content = re.sub(r"^Figure \d+\.\s*", "", match[1])
        return f"<figcaption>Figure {number}. {content}</figcaption>"

    return re.sub(r"<figcaption>(.*?)</figcaption>", caption, text, flags=re.DOTALL)


def update_html(text, result):
    """Replace the broader summary with newly audited results."""
    return integrate_html(text, html_section(result))


def render(base):
    """Regenerate evidence and figures only from completed saved reductions."""
    result, maps = audit(base.resolve())
    figures(base, result, maps)
    report = base / "report/index.html"
    report.write_text(update_html(report.read_text(), result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    render(**vars(parser.parse_args()))
