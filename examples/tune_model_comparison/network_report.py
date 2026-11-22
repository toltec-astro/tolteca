#!/usr/bin/env python
"""Render network coverage and conditional nonlinear sensitivity from saved runs."""

from __future__ import annotations

import argparse
import html
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from astropy.table import Table
from markdown_it import MarkdownIt
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from tolteca_kidsproc._nonlinear import nonlinear_detuning

FAILURE_MASK = 65535 ^ 32
LOSS_ATOL = 1e-12
FREQUENCY_ATOL_HZ = 0.02


def sensitivity(r, xd, b):
    """Return stationary slope, loss coupling and conditional b-error response.

    ``r`` and ``xd`` are loss and effective detuning x-prime. The 5% and 10% quantities
    are assumed fractional calibration errors in b, not measured uncertainties.
    The noise ratio assumes isotropic IQ noise and correct fixed gain/coupling.
    """
    d = r**2 + xd**2
    slope = 1 + 2 * b * xd / d**2
    mixing = 2 * b * r / d**2
    return {
        "D": slope,
        "mu": mixing,
        "u": xd / r,
        "bias_5pct": 0.05 * abs((slope - 1) / slope),
        "bias_10pct": 0.10 * abs((slope - 1) / slope),
        "readout_noise_ratio": np.hypot(slope, mixing),
    }


def channel_sensitivity(folder):
    """Assess the measured independent-channel solutions without selecting them."""
    request = json.loads((folder / "request.json").read_text())
    report = Table.read(request["ecsv"], format="ascii.ecsv")
    shape = request["samples"], request["channels"]
    i, q = [np.fromfile(request[k], dtype="float64").reshape(shape) for k in "IQ"]
    flags = np.fromfile(folder / "native/stub/flags.bin", dtype="uint16").reshape(shape)
    results = []
    for row in report:
        chan = int(row["chan"])
        good = (flags[:, chan] & FAILURE_MASK) == 0
        if not good.any():
            continue
        df = row["f_in"] - row["f_g"]
        gain = complex(row["H0_re"], row["H0_im"])
        if "H1_re" in report.colnames:
            gain += complex(row["H1_re"], row["H1_im"]) * df
        else:
            gain *= np.exp((row["a"] - 2j * np.pi * row["tau"]) * df)
        z = (i[good, chan] + 1j * q[good, chan]) / request["normalization"]
        w = complex(row["c_re"], row["c_im"]) / (1 - z / gain)
        quantities = sensitivity(w.real, w.imag, row["b"])
        y_reference, _ = nonlinear_detuning(
            (row["f_in"] / row["fr"] - 1) / (2 * row["r"]),
            row["a_nl"],
            branch="increasing",
        )
        item = {
            "chan": chan,
            "accepted_fraction": float(good.mean()),
            "a_fit": float(row["a_nl"]),
            "u_reference": float(2 * y_reference),
        }
        for name, value in quantities.items():
            for label, quantile in (
                ("min", 0),
                ("median", 0.5),
                ("p95", 0.95),
                ("max", 1),
            ):
                item[f"{name}_{label}"] = float(np.quantile(value, quantile))
        results.append(item)
    table = pd.DataFrame(results)
    table.to_csv(folder / "nonlinear-sensitivity.csv", index=False)
    return table


def component_coverage(folder, component_file):
    """Retain every fitted component and mark its actual native output coverage."""
    components = pd.read_csv(
        component_file, dtype={"group_id": str, "resonator_id": str}
    )
    channels = pd.read_csv(
        component_file.parent / "channels.csv", dtype={"group_id": str}
    )
    request = json.loads((folder / "request.json").read_text())
    summary = yaml.safe_load((folder / "native/stub/summary.yaml").read_text())
    groups = {str(g["group_id"]): g for g in summary["joint"]}
    flags = np.fromfile(folder / "native/stub/flags.bin", dtype="uint16").reshape(
        request["samples"], request["channels"]
    )
    fractions, routes, reasons = [], [], []
    for row in components.itertuples():
        gid = str(row.group_id)
        if gid in groups:
            mask = np.fromfile(
                folder / "native/stub" / f"group-{gid}/flags.bin", dtype="uint16"
            )
            good = (mask & FAILURE_MASK) == 0
            route = "joint"
        elif row.multiplicity == 1:
            selected = channels.loc[channels.group_id == gid, "chan"].to_numpy(
                dtype=int
            )
            mask = flags[:, selected]
            good = (
                np.any((mask & FAILURE_MASK) == 0, axis=1)
                if len(selected)
                else np.zeros(len(flags), bool)
            )
            route = "independent" if len(selected) else "unselected"
        else:
            mask = np.array([], dtype="uint16")
            good = np.zeros(len(flags), bool)
            route = "unselected"
        fractions.append(float(good.mean()))
        routes.append(route)
        reasons.append(";".join(map(str, np.unique(mask).tolist())))
    components["native_fraction"] = fractions
    components["native_route"] = routes
    components["native_sample_flag_values"] = reasons
    components.to_csv(folder / "component-coverage.csv", index=False)
    return components


def summary_row(family, block, table):
    """Count components separately from channels and samples."""
    return {
        "Gain": family,
        "Block / data": block,
        "Components": len(table),
        "Singlet": int((table.multiplicity == 1).sum()),
        "Multiplet": int((table.multiplicity > 1).sum()),
        "Fit converged": int((table.fit_status == "optimizer_converged").sum()),
        "Any valid solving": int((table.native_fraction > 0).sum()),
        "All samples valid": int((table.native_fraction == 1).sum()),
        "No valid solving": int((table.native_fraction == 0).sum()),
    }


def figure_html(fig):
    """Use the maintained ModelFit viewer's scientific Plotly presentation."""
    fig.update_layout(
        template="plotly_white",
        font={"size": 16},
        margin={
            "l": 70,
            "r": 30,
            "b": 120 if fig.layout.legend.orientation == "h" else 65,
            "t": 70,
        },
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def collect_results(root):
    """Collect full-network denominators and measured native coverage."""
    summary, datasets, working, cells = [], {}, {}, []
    for family, name in (("Affine", "coverage"), ("Delay", "coverage-delay")):
        base = root / name
        if not (base / "measured/native/stub/summary.yaml").exists():
            msg = f"Native measured replay is incomplete: {family}"
            raise ValueError(msg)
        for block in (0, 1):
            folder = base / f"block{block}"
            table = component_coverage(folder, folder / "components.csv")
            datasets[family, str(block)] = table
            summary.append(summary_row(family, f"{block}: controlled", table))
        measured = component_coverage(base / "measured", base / "block1/components.csv")
        datasets[family, "measured"] = measured
        summary.append(summary_row(family, "1: measured", measured))
        working[family] = channel_sensitivity(base / "measured")
        for label, table in [
            ("0", datasets[family, "0"]),
            ("1", datasets[family, "1"]),
            ("measured", measured),
        ]:
            for multiplicity, select in (
                ("Singlet", table.multiplicity == 1),
                ("Multiplet", table.multiplicity > 1),
            ):
                for bin_name, low, high in (
                    ("0-0.01", 0, 0.01),
                    ("0.01-0.2", 0.01, 0.2),
                    ("0.2-0.75", 0.2, 0.75),
                    (">0.75", 0.75, np.inf),
                ):
                    part = table.loc[
                        select
                        & (table.a_nl >= low if low == 0 else table.a_nl > low)
                        & (table.a_nl <= high)
                    ]
                    cells.append(
                        {
                            "Gain": family,
                            "Data": label,
                            "Multiplicity": multiplicity,
                            "a_nl (display bin)": bin_name,
                            "Components": len(part),
                            "Any valid": int((part.native_fraction > 0).sum()),
                            "All valid": int((part.native_fraction == 1).sum()),
                        }
                    )
    pd.DataFrame(summary).to_csv(root / "summary.csv", index=False)
    pd.DataFrame(cells).to_csv(root / "matrix.csv", index=False)
    return summary, datasets, working, cells


def inspection_links(root):
    """Link the exact fit viewers, component records and bound stub products."""
    rows = []
    for family, name, version in (
        ("Affine", "coverage", "0000-models"),
        ("Delay", "coverage-delay", "0001-models"),
    ):
        base = root / name
        fit = (
            Path("dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc")
            / version
            / "artifacts/toltec0/blocks"
        )
        for block in (0, 1):
            viewer = (
                fit
                / str(block)
                / "tolteca_kids.model_fit_plot:ModelFitPlot/model_fit.html"
            )
            rows.append(
                {
                    "Gain / data": f"{family}, Tune block {block}",
                    "Sweep and fit": f'<a href="{viewer}">Interactive model viewer</a>',
                    "Coverage": (
                        f'<a href="{name}/block{block}/component-coverage.csv">'
                        "Every component</a>"
                    ),
                }
            )
        prepared = json.loads((base / "measured/prepared.json").read_text())
        index = Path(prepared["index"]).relative_to(root)
        rows.append(
            {
                "Gain / data": f"{family}, measured scan 2",
                "Sweep and fit": f'<a href="{index}">Published stub index.yaml</a>',
                "Coverage": (
                    f'<a href="{name}/measured/component-coverage.csv">'
                    "Every component</a>"
                ),
            }
        )
    return pd.DataFrame(rows).to_html(index=False, border=0, escape=False)


def outcome_table(datasets):
    """Separate missing selection from selected models with failed samples."""
    rows = []
    for (family, label), table in datasets.items():
        unselected = table.native_route == "unselected"
        rows.append(
            {
                "Gain": family,
                "Data": label,
                "Unselected components": int(unselected.sum()),
                "Selected, no valid sample": int(
                    (~unselected & (table.native_fraction == 0)).sum()
                ),
                "Some samples valid": int(
                    ((table.native_fraction > 0) & (table.native_fraction < 1)).sum()
                ),
                "All samples valid": int((table.native_fraction == 1).sum()),
            }
        )
    return pd.DataFrame(rows).to_html(index=False, border=0)


def parity_table(root):
    """Show completed comparison scope and strict absolute-tolerance misses."""
    rows = []
    for family, name in (("Affine", "coverage"), ("Delay", "coverage-delay")):
        for part in ("block0", "block1", "measured"):
            folder = root / name / part
            path = folder / "solving.json"
            if not path.exists():
                rows.append({"Gain": family, "Data": part, "Complete": False})
                continue
            data = json.loads(path.read_text())
            native = yaml.safe_load((folder / "native/ecsv/summary.yaml").read_text())
            joint = [g["parity"]["ecsv"] for g in data["joint"]]
            request = json.loads((folder / "request.json").read_text())
            checks = [data["independent"]["ecsv"], *joint]
            r_errors = [c["r"]["max_abs_difference"] or 0 for c in checks]
            f_errors = [c["fr"]["max_abs_difference"] or 0 for c in joint]
            rows.append(
                {
                    "Gain": family,
                    "Data": part,
                    "Complete": len(joint) == len(native["joint"]),
                    "Joint groups checked": len(joint),
                    "Samples per joint group": min(256, request["samples"]),
                    "Flag mismatches (groups/routes)": sum(
                        not c["flags_equal"] for c in checks
                    ),
                    "Max |delta r|": max(r_errors),
                    "Max |delta fr| (Hz)": max(f_errors, default=0),
                    "r tolerance misses": sum(v > LOSS_ATOL for v in r_errors),
                    "fr tolerance misses": sum(v > FREQUENCY_ATOL_HZ for v in f_errors),
                    "Finite-mask mismatches": sum(
                        not c[k]["same_finite_mask"]
                        for c in checks
                        for k in ("r", "fr" if "fr" in c else "x")
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(root / "parity.csv", index=False)
    return table.to_html(index=False, border=0)


def parity_exceptions(root):
    """Separate acceptance differences, failure-bit differences and precision."""
    rows = []
    for family, name in (("Affine", "coverage"), ("Delay", "coverage-delay")):
        folder = root / name / "measured"
        path = folder / "solving.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for group in data["joint"]:
            check = group["parity"]["ecsv"]
            if (
                check["flags_equal"]
                and check["r"]["within_tolerance"]
                and check["fr"]["within_tolerance"]
            ):
                continue
            gid = group["group_id"]
            with np.load(folder / f"python-group-{gid}.npz") as arrays:
                python_flags = arrays["flags"]
            native_flags = np.fromfile(
                folder / "native/ecsv" / f"group-{gid}/flags.bin", dtype="uint16"
            )[: len(python_flags)]
            pgood = (python_flags & FAILURE_MASK) == 0
            ngood = (native_flags & FAILURE_MASK) == 0
            rows.append(
                {
                    "Gain": family,
                    "Group": gid,
                    "Channels": ", ".join(map(str, group["channels"])),
                    "Python accepted": int(pgood.sum()),
                    "Native accepted": int(ngood.sum()),
                    "Different acceptance": int((pgood != ngood).sum()),
                    "Different flag bits": int((python_flags != native_flags).sum()),
                    "Max |delta r|": check["r"]["max_abs_difference"],
                    "Max |delta fr| (Hz)": check["fr"]["max_abs_difference"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(root / "parity-exceptions.csv", index=False)
    return table.to_html(index=False, border=0, na_rep="--")


def render(root):
    """Build the local review page from completed native output."""
    summary, datasets, working, cells = collect_results(root)
    dist = make_subplots(
        rows=1, cols=2, subplot_titles=("Tune block 0", "Tune block 1")
    )
    for family, color in (("Affine", "#228be6"), ("Delay", "#e67700")):
        for block in (0, 1):
            d = datasets[family, str(block)]
            dist.add_trace(
                go.Histogram(
                    x=d.a_nl,
                    name=family,
                    marker_color=color,
                    opacity=0.55,
                    xbins={"start": 0, "end": 2, "size": 0.025},
                    showlegend=block == 0,
                ),
                row=1,
                col=block + 1,
            )
    dist.update_layout(
        barmode="overlay",
        title="Fitted nonlinearity: all components; overflow retained in the tables",
    )
    dist.update_xaxes(title_text="a_nl (dimensionless)", range=[0, 2])
    dist.update_yaxes(title_text="Components")
    zone = make_subplots(
        rows=1, cols=2, subplot_titles=("5% assumed b error", "10% assumed b error")
    )
    u = np.linspace(-2.5, 2.5, 201)
    a = np.linspace(0, 1.5, 151)
    uu, aa = np.meshgrid(u, a)
    slope = 1 + 4 * aa * uu / (1 + uu**2) ** 2
    for col, eps in enumerate((0.05, 0.1), 1):
        error = (
            100
            * eps
            * abs(
                np.divide(
                    slope - 1, slope, out=np.full_like(slope, np.nan), where=slope > 0
                )
            )
        )
        zone.add_trace(
            go.Heatmap(
                x=u,
                y=a,
                z=error,
                zmin=0,
                zmax=20,
                colorscale="Cividis",
                showscale=col == len((0.05, 0.1)),
                colorbar={"title": "Amplitude error bound (%)"},
            ),
            row=1,
            col=col,
        )
        zone.add_trace(
            go.Contour(
                x=u,
                y=a,
                z=error,
                contours={"start": 5, "end": 5, "size": 1, "coloring": "none"},
                line={"color": "white", "width": 2},
                showscale=False,
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        for family, marker in (("Affine", "circle"), ("Delay", "x")):
            w = working[family]
            zone.add_trace(
                go.Scattergl(
                    x=w.u_reference,
                    y=w.a_fit,
                    text=w.chan,
                    mode="markers",
                    name=family,
                    marker={
                        "color": "#e03131" if family == "Affine" else "#212529",
                        "size": 5,
                        "symbol": marker,
                    },
                    showlegend=col == 1,
                    hovertemplate=(
                        "Channel %{text}<br>x\N{PRIME}/r=%{x:.4g}"
                        "<br>a=%{y:.4g}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    zone.update_xaxes(title_text="x\N{PRIME} / r", range=[-2.5, 2.5])
    zone.update_yaxes(title_text="a_nl fitted from the sweep", range=[0, 1.5])
    zone.update_layout(
        title="Fixed-loss calibration-error scenario; sweep reference states",
        height=540,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.22},
    )
    rates = go.Figure()
    for family, color in (("Affine", "#228be6"), ("Delay", "#e67700")):
        d = datasets[family, "measured"]
        rates.add_trace(
            go.Scattergl(
                x=d.a_nl,
                y=d.native_fraction,
                mode="markers",
                name=family,
                marker={"color": color, "size": 6},
                customdata=np.c_[d.group_id, d.resonator_id, d.multiplicity],
                hovertemplate=(
                    "Group %{customdata[0]}, component %{customdata[1]}"
                    "<br>Multiplicity %{customdata[2]}<br>a=%{x:.4g}"
                    "<br>Accepted fraction=%{y:.4g}<extra></extra>"
                ),
            )
        )
    rates.update_layout(
        title="Measured solving coverage: failed and unselected components retained"
    )
    rates.update_xaxes(title_text="Fitted a_nl")
    rates.update_yaxes(
        title_text="Fraction of 7,802 measured samples", range=[-0.03, 1.03]
    )
    quality = runpy.run_path(str(Path(__file__).with_name("network_quality.py")))
    quality_table, quality_figure = quality["quality_panels"](root)
    example_channel, example_figure = quality["worked_example"](root)
    response = runpy.run_path(str(Path(__file__).with_name("network_response.py")))
    response_table, response_figure = response["response_panels"](root)
    sky_figure = response["skydip_figure"](root)
    populations = runpy.run_path(
        str(Path(__file__).with_name("network_populations.py"))
    )
    population_table, population_figure = populations["panels"](root)
    (root / "plotly.min.js").write_text(get_plotlyjs())
    evidence = json.loads((root / "evidence.json").read_text())
    records = [
        {
            "Study": (
                f'<a href="{html.escape(e["copy"])}">{html.escape(e["title"])}</a>'
            ),
            "Retained result / limit": html.escape(e["finding"]),
        }
        for e in evidence
    ]
    page = Path(__file__).with_suffix(".html").read_text()
    for key, value in {
        "{{POPULATION_TABLE}}": population_table,
        "{{POPULATION_FIGURE}}": figure_html(population_figure),
        "{{RESPONSE_TABLE}}": response_table,
        "{{RESPONSE_FIGURE}}": figure_html(response_figure),
        "{{SKYDIP_FIGURE}}": figure_html(sky_figure),
        "{{QUALITY_TABLE}}": quality_table,
        "{{QUALITY_FIGURE}}": figure_html(quality_figure),
        "{{EXAMPLE_CHANNEL}}": str(example_channel),
        "{{EXAMPLE_FIGURE}}": figure_html(example_figure),
        "{{SUMMARY}}": pd.DataFrame(summary).to_html(index=False, border=0),
        "{{MATRIX}}": pd.DataFrame(cells).to_html(index=False, border=0),
        "{{DISTRIBUTION}}": figure_html(dist),
        "{{RATES}}": figure_html(rates),
        "{{ZONE}}": figure_html(zone),
        "{{OUTCOMES}}": outcome_table(datasets),
        "{{INSPECTION}}": inspection_links(root),
        "{{PARITY}}": parity_table(root),
        "{{EXCEPTIONS}}": parity_exceptions(root),
        '<div id="evidence"></div>': pd.DataFrame(records).to_html(
            index=False, border=0, escape=False
        ),
    }.items():
        page = page.replace(key, value)
    (root / "index.html").write_text(page)
    readme = root / "README.md"
    if not readme.exists():
        readme = Path(__file__).with_name("README.md")
    style = page.split("<style>", 1)[1].split("</style>", 1)[0]
    (root / "readme.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Methods and evidence</title>'
        f"<style>{style}</style>"
        + MarkdownIt("commonmark").enable("table").render(readme.read_text())
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root.resolve())
