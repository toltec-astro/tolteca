"""Scientific figures for contributor selection and paired sweep fit quality."""

# ruff: noqa: INP001, PLR2004 -- standalone report and figure layout constants

import json

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.lines import Line2D

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_kids.model_fit import ModelFit
from tolteca_kids.product import open_kids_product
from tolteca_kidsproc.models import KidsSweepGainWithLinTrend

ARRAYS = ("a1100", "a1400", "a2000")
COLORS = {
    "good": "#387d6c",
    "joint": "#d28d25",
    "unusable": "#b94a48",
    "bad": "#b7b7b7",
    "legacy": "#586776",
    "python": "#514397",
}


def save(fig, output, name):
    """Write the same scientific figure as a preview and vector PDF."""
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def clean(ax):
    """Keep guides unobtrusive and remove unused chart borders."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.18)


def stacked(ax, rows, categories, labels, *, legend=True):
    """Plot absolute counts; retain small classes in the downloadable ledger."""
    left = np.zeros(len(rows))
    total = max(sum(r.get(k, 0) for k, _, _ in categories) for r in rows)
    for key, label, color in categories:
        values = np.array([r.get(key, 0) for r in rows])
        ax.barh(
            np.arange(len(rows)),
            values,
            left=left,
            label=label,
            color=color,
            height=0.62,
        )
        for y, value in enumerate(values):
            if value >= max(1, 0.09 * total):
                ax.text(
                    left[y] + value / 2,
                    y,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white"
                    if color not in {COLORS["bad"], COLORS["joint"], "#e1c465"}
                    else "#202020",
                )
        left += values
    for y, value in enumerate(left):
        ax.text(value + total * 0.02, y, f"{int(value):,}", va="center", fontsize=10)
    ax.set(
        yticks=range(len(rows)),
        yticklabels=labels,
        xlim=(0, max(1, total * 1.17)),
        xlabel="Acquired channel / APT rows",
    )
    ax.invert_yaxis()
    clean(ax)
    if legend:
        ax.legend(
            loc="upper left", bbox_to_anchor=(0, -0.25), frameon=False, fontsize=10
        )


def populations(audit, output):
    """Show the beammap gate, added map restrictions and final populations."""
    arrays = [audit["arrays"][str(i)] for i in range(3)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    stacked(
        axes[0],
        arrays,
        [
            ("apt_good", "Beammap flag = 0", COLORS["good"]),
            ("apt_bad", "Beammap flag != 0", COLORS["bad"]),
        ],
        ARRAYS,
    )
    axes[0].set_title("a. Original beammap APT")
    stacked(
        axes[1],
        [r["eligibility"] for r in arrays],
        [
            ("eligible", "Scalar-map eligible", COLORS["good"]),
            ("joint_apt_binding_missing", "Joint output not mapped", COLORS["joint"]),
            ("fit_unusable", "Unusable flag: inspect cause", COLORS["unusable"]),
        ],
        ARRAYS,
    )
    axes[1].set_title("b. APT-good rows only")
    labels = []
    for i, row in enumerate(arrays):
        for j, (name, label, color) in enumerate(
            [
                ("legacy", "Full legacy", COLORS["legacy"]),
                ("common", "Legacy with v3 eligibility", "#88a6ba"),
                ("python", "V3 models", COLORS["python"]),
            ]
        ):
            y = i * 4 + j
            n = row["populations"][name]["channels"]
            axes[2].barh(y, n, color=color, label=label if i == 0 else None)
            axes[2].text(n + 45, y, f"{n:,}", va="center", fontsize=10)
        labels.append(i * 4 + 1)
    axes[2].set(
        yticks=labels,
        yticklabels=ARRAYS,
        xlim=(0, 3100),
        xlabel="Contribute in at least one chunk",
    )
    axes[2].invert_yaxis()
    axes[2].legend(
        loc="upper left", bbox_to_anchor=(0, -0.25), frameon=False, fontsize=10
    )
    axes[2].set_title("c. Saved map contributions")
    clean(axes[2])
    fig.suptitle(
        "152434: beammap selection is not model quality or map contribution",
        fontsize=15,
    )
    fig.subplots_adjust(wspace=0.38, bottom=0.3, top=0.84)
    save(fig, output, "population-flow")
    for i, array in enumerate(ARRAYS):
        ps = arrays[i]["populations"]
        rows = [ps[k] for k in ("legacy", "common", "python")]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        labels = ["Legacy", "Legacy restricted", "V3"]
        configs = [
            (
                "group_kind",
                "Selected group membership",
                [
                    ("singlet_one_probe", "One resonator / one probe", "#427c9d"),
                    (
                        "singlet_multiple_probes",
                        "One resonator / several probes",
                        "#77b6a5",
                    ),
                    ("multiplet", "Multi-resonator group", COLORS["joint"]),
                    ("unresolved", "No selected group", COLORS["bad"]),
                ],
            ),
            (
                "a_nl_bin",
                "Fitted nonlinearity at each probe",
                [
                    ("[0,0.1)", "0 <= a_nl < 0.1", "#387d6c"),
                    ("[0.1,0.5)", "0.1 <= a_nl < 0.5", "#427c9d"),
                    ("[0.5,1)", "0.5 <= a_nl < 1", "#8064a2"),
                    ("[1,infinity)", "a_nl >= 1", "#9f3e63"),
                    ("blended", "Blended: several a_nl values", COLORS["joint"]),
                    ("unresolved", "Unresolved", COLORS["bad"]),
                ],
            ),
            (
                "quality_state",
                "Python diagnostic flags",
                [
                    ("no_flags", "No listed flags", "#387d6c"),
                    ("warnings_only", "Warning-only", "#427c9d"),
                    ("joint_required", "Joint required", COLORS["joint"]),
                    ("unusable", "Unusable", COLORS["unusable"]),
                ],
            ),
        ]
        for ax, (field, title, categories) in zip(axes, configs, strict=True):
            stacked(ax, [r[field] for r in rows], categories, labels)
            ax.set_title(title)
        fig.suptitle(
            f"{array}: actual map contributors, labeled by the same v3 fit", fontsize=15
        )
        fig.subplots_adjust(wspace=0.4, bottom=0.39, top=0.84)
        save(fig, output, f"{array}-populations")

        # A separate portrait layout keeps labels readable in the printed report.
        fig, axes = plt.subplots(3, 1, figsize=(6.7, 8.2))
        for ax, (field, title, categories) in zip(axes, configs, strict=True):
            stacked(ax, [r[field] for r in rows], categories, labels, legend=False)
            ax.set_title(title)
            ax.set_xlabel("")
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(0, -0.2),
                ncol=2,
                frameon=False,
                fontsize=9,
            )
        fig.suptitle(f"{array}: actual map contributors", fontsize=14)
        fig.subplots_adjust(left=0.23, right=0.97, hspace=1.45, bottom=0.12, top=0.91)
        save(fig, output, f"{array}-populations-print")


def coverage(quality, output):
    """Separate fitted sweeps from the much smaller scalar-map population."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    fields = [
        "apt_good",
        "has_fitted_response",
        "has_converged_response",
        "has_selected_response",
        "selected_optimizer_converged",
        "map_eligible",
    ]
    labels = [
        "Beammap good",
        "Has a fitted response",
        "Has a converged response",
        "Has selected probe response",
        "Selected response converged",
        "Scalar-map eligible",
    ]
    for i, ax in enumerate(axes):
        row = quality["arrays"][str(i)]
        values = [row[k] for k in fields]
        ax.barh(
            range(len(fields)),
            values,
            color=[
                COLORS["bad"],
                "#77a8bc",
                "#427c9d",
                "#77a8bc",
                "#427c9d",
                COLORS["good"],
            ],
        )
        for y, n in enumerate(values):
            ax.text(
                n + row["apt_good"] * 0.02,
                y,
                f"{n:,}  ({n / row['apt_good']:.1%})",
                va="center",
                fontsize=10,
            )
        ax.set(
            yticks=range(len(fields)),
            yticklabels=labels if i == 0 else [""] * len(fields),
            xlim=(0, row["apt_good"] * 1.4),
            xlabel="APT-good channel rows",
        )
        ax.invert_yaxis()
        ax.set_title(ARRAYS[i])
        clean(ax)
    fig.suptitle(
        "Sweep modeling coverage versus downstream map eligibility", fontsize=15
    )
    fig.tight_layout(w_pad=2)
    save(fig, output, "model-coverage")
    reasons = [
        ("joint_output_not_mapped", "Joint-model output not mapped", COLORS["joint"]),
        ("recovery_unresolved", "Insufficient simultaneous probes", "#b49b6d"),
        ("no_model_at_probe", "No fitted response", COLORS["bad"]),
        ("optimizer_not_converged", "Optimizer did not converge", COLORS["unusable"]),
        ("probe_binding_unresolved", "Probe outside fitted windows", "#8064a2"),
    ]
    totals = [
        sum(r["exclusion_reason"].get(key, 0) for r in quality["arrays"].values())
        for key, _, _ in reasons
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(range(len(reasons)), totals, color=[r[2] for r in reasons])
    for y, n in enumerate(totals):
        ax.text(n + 15, y, f"{n:,}", va="center")
    ax.set(
        yticks=range(len(reasons)),
        yticklabels=[r[1] for r in reasons],
        xlim=(0, max(totals) * 1.15),
        xlabel="APT-good channel rows excluded from scalar maps",
    )
    ax.invert_yaxis()
    ax.set_title(
        f"Why {sum(totals):,} beammap-good channels did not enter the scalar-map trial"
    )
    clean(ax)
    fig.tight_layout()
    save(fig, output, "model-exclusions")


def quality_plots(quality, responses, output):
    """Write a compact overview and readable per-array print figures."""
    _quality_plot(quality, responses, output, ARRAYS)
    for array in ARRAYS:
        _quality_plot(quality, responses, output, (array,))


def _quality_plot(quality, responses, output, arrays):
    single = len(arrays) == 1
    fig, axes = plt.subplots(
        2, len(arrays), figsize=(6.5, 8) if single else (13, 8.8), squeeze=False
    )
    colors = {
        "eligible": COLORS["good"],
        "joint_apt_binding_missing": COLORS["joint"],
        "fit_unusable": COLORS["unusable"],
    }
    labels = {
        "eligible": "Scalar-map eligible",
        "joint_apt_binding_missing": "Joint output not mapped",
        "fit_unusable": "Unusable flag (not all failed fits)",
    }
    selected = responses[(responses["apt_flag"] == 0) & responses["selected_for_probe"]]
    for column, array in enumerate(arrays):
        i = ARRAYS.index(array)
        r = selected[selected["array"] == i]
        old, new = np.asarray(r["legacy_fraction"]), np.asarray(r["v3_fraction"])
        keep = np.isfinite(old) & np.isfinite(new) & (old > 0) & (new > 0)
        ax = axes[0, column]
        limits = (
            min(old[keep].min(), new[keep].min()) * 0.7,
            max(old[keep].max(), new[keep].max()) * 1.4,
        )
        for status, color in colors.items():
            for multi, marker in [(False, "o"), (True, "^")]:
                take = (
                    keep
                    & (r["status"] == status)
                    & ((r["fit_resonators"] > 1) == multi)
                )
                ax.scatter(
                    old[take],
                    new[take],
                    s=13,
                    color=color,
                    marker=marker,
                    alpha=0.55,
                    linewidths=0,
                    rasterized=True,
                )
        ax.plot(limits, limits, color="#444", lw=1, ls="--")
        ax.set(
            xscale="log",
            yscale="log",
            xlim=limits,
            ylim=limits,
            xlabel="Legacy residual / sweep excursion",
            ylabel="V3 residual / sweep excursion",
        )
        s = quality["arrays"][str(i)]["selected_responses"]
        ax.set_title(f"{array}: {s['finite_pairs']:,} paired channels")
        ax.text(
            0.04,
            0.96,
            f"V3 smaller: {s['v3_smaller_residual'] / s['finite_pairs']:.1%}\n"
            f"Median paired ratio: {s['median_paired_ratio']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
        ax.grid(alpha=0.15)
        ax = axes[1, column]
        for name, values, color in [
            ("Legacy tune.txt", old[keep], COLORS["legacy"]),
            ("V3 group model", new[keep], COLORS["python"]),
        ]:
            ordered = np.sort(values)
            ax.step(
                ordered,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                color=color,
                label=name,
            )
        ax.set(
            xscale="log",
            xlabel="Residual RMS / measured sweep excursion",
            ylabel="Fraction of paired channels",
            ylim=(0, 1.03),
        )
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=10, loc="lower right")
    handles = [
        Line2D([], [], color=c, marker="o", ls="", label=labels[k])
        for k, c in colors.items()
    ]
    handles += [
        Line2D([], [], color="black", marker=m, ls="", label=label)
        for m, label in [
            ("o", "One-resonator fit group"),
            ("^", "Multi-resonator fit group"),
        ]
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2 if single else 3,
        frameon=False,
        fontsize=9 if single else 10,
    )
    fig.suptitle(
        (
            "152434 Tune block 1: legacy versus nonlinear group model"
            if single
            else "152434 Tune block 1: legacy versus full nonlinear sweep model"
        ),
        fontsize=12 if single else 15,
    )
    fig.tight_layout(rect=(0, 0.14 if single else 0.1, 1, 0.95))
    save(fig, output, f"{arrays[0]}-sweep-quality" if single else "sweep-model-quality")


def examples(prepared, legacy_manifest, responses, output):
    """Show typical excluded joint groups, chosen by median paired residual ratio."""
    receipt = json.loads((prepared / "prepared.json").read_text())
    model = DataProdIndex.read(receipt["model_index"])
    legacy = {
        r["interface"]: r
        for r in json.loads(legacy_manifest.read_text())["model_equivalence"]
    }
    records = []
    for i, array in enumerate(ARRAYS):
        eligible = responses[
            (responses["array"] == i)
            & (responses["apt_flag"] == 0)
            & responses["selected_for_probe"]
            & (responses["status"] == "joint_apt_binding_missing")
            & (responses["fit_channels"] == 2)
            & (responses["fit_resonators"] == 2)
        ]
        ratios = np.asarray(eligible["v3_fraction"] / eligible["legacy_fraction"])
        pick = eligible[np.argmin(abs(ratios - np.median(ratios)))]
        nw, gid = int(pick["nw"]), str(pick["group_id"])
        interface = f"toltec{nw}"
        old = Table.read(legacy[interface]["labeled"], format="ascii.ecsv")
        old_lookup = {int(c): j for j, c in enumerate(old["chan"])}
        with open_kids_product(model, interface=interface) as tree:
            [block] = [b for b in tree.toltec_kids.blocks if b.meta.block == 1]
            view = ModelFit.view(block.tree)
            samples, fit = view.samples.load(), view.fit.load()
            group = samples.isel(sample=np.flatnonzero(samples.group_id.values == gid))
            chans = np.unique(group.chan.values)
            fig, axes = plt.subplots(
                len(chans), 2, figsize=(11, 3.2 * len(chans)), squeeze=False
            )
            for row, chan in enumerate(chans):
                s = group.isel(sample=np.flatnonzero(group.chan.values == chan)).sortby(
                    "frequency"
                )
                old_model = KidsSweepGainWithLinTrend.from_table(
                    old[[old_lookup[int(chan)]]]
                )
                prediction = old_model.eval(s.frequency.values)
                center = float(view.chan_prop.sel(chan=chan).fp)
                x = (s.frequency.values - center) / 1e3
                for j, (field, values) in enumerate(
                    [("I", prediction.real), ("Q", prediction.imag)]
                ):
                    ax = axes[row, j]
                    scale = 1e6
                    ax.plot(
                        x,
                        s[field].values / scale,
                        ".",
                        ms=3,
                        color="black",
                        label="Measured sweep",
                    )
                    ax.plot(
                        x,
                        values / scale,
                        color=COLORS["legacy"],
                        label="Legacy tune.txt",
                    )
                    ax.plot(
                        x,
                        s[f"{field}_model"].values / scale,
                        color=COLORS["python"],
                        label="V3 full group",
                    )
                    ax.set(
                        xlabel="Frequency - acquired probe (kHz)",
                        ylabel=f"{field} (recorded IQ / 10⁶)",
                        title=f"Network {nw}, channel {chan}",
                    )
                    ax.grid(alpha=0.18)
            rows = fit.isel(row=np.flatnonzero(fit.group_id.values == gid))
            ids = np.unique(rows.resonator_id)
            strengths = [
                float(
                    rows.a_nl.values[np.flatnonzero(rows.resonator_id.values == r)[0]]
                )
                for r in ids
            ]
            fig.suptitle(
                f"{array}: M = 2 channels, N = 2 resonators; "
                f"a_nl = {strengths[0]:.3g}, {strengths[1]:.3g}\n"
                f"Group {gid}: retained fit, excluded from scalar maps",
                fontsize=14,
            )
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
            fig.tight_layout(rect=(0, 0.05, 1, 0.91))
            save(fig, output, f"{array}-sweep-example")
        records.append(
            {
                "array": array,
                "nw": nw,
                "group_id": gid,
                "channels": chans.tolist(),
                "a_nl": strengths,
                "selection": (
                    "closest to median paired residual ratio among APT-good selected "
                    "2-channel/2-resonator responses excluded only from joint mapping"
                ),
                "selection_pool": len(eligible),
            }
        )
    (output / "sweep-example-selection.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
