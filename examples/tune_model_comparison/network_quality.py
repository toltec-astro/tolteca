#!/usr/bin/env python
"""Measure uncleaned solved-frequency spectra without claiming optical sensitivity."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.table import Table
from plotly.subplots import make_subplots

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kidsproc.analysis import PSDAnalysis

FAILURE_MASK = 65535 ^ 32
SUMMARY_BAND_HZ = (5, 20)
BINS = [
    ("0-0.01", 0, 0.01),
    ("0.01-0.2", 0.01, 0.2),
    ("0.2-0.75", 0.2, 0.75),
    (">0.75", 0.75, np.inf),
]
COLORS = ["#495057", "#228be6", "#e67700", "#ae3ec9"]


def spectrum(values, rate):
    """Use the maintained PSD analysis with one declared Welch configuration."""
    return PSDAnalysis.compute_psd(
        np.asarray(values),
        f_smp=rate,
        window="hann",
        nperseg=1024,
        noverlap=512,
        detrend="constant",
        scaling="density",
        return_onesided=True,
        axis=0,
    )


def _selected_bin(table, low, high):
    return (table.a_nl >= low if low == 0 else table.a_nl > low) & (table.a_nl <= high)


def measure(folder):  # noqa: PLR0915 - explicit spectral evidence workflow
    """Save component spectra, preserving all exclusions and counting no probe twice."""
    request = json.loads((folder / "request.json").read_text())
    prepared = json.loads((folder / "prepared.json").read_text())
    with open_toltec_kids(prepared["data_item"]["filepath"]) as raw:
        rate = float(raw["Header.Toltec.SampleFreq"].item())
        packets = raw["Data.Toltec.Ts"].to_numpy()[:, 3].astype(np.uint32)
        assert np.all(np.diff(packets) == 1)
    report = Table.read(request["ecsv"], format="ascii.ecsv")
    coverage = pd.read_csv(
        folder / "component-coverage.csv", dtype={"group_id": str, "resonator_id": str}
    )
    channels = pd.read_csv(
        folder.parent / "block1/channels.csv", dtype={"group_id": str}
    )
    shape = request["samples"], request["channels"]
    native = folder / "native/stub"
    x = np.fromfile(native / "x.bin", dtype="float64").reshape(shape)
    flags = np.fromfile(native / "flags.bin", dtype="uint16").reshape(shape)
    good_channels = np.flatnonzero(np.all((flags & FAILURE_MASK) == 0, axis=0))
    frequency_series = np.asarray(report["f_in"])[good_channels] / (
        1 + x[:, good_channels]
    )
    nu, channel_power = spectrum(frequency_series, rate)
    lookup = {chan: col for col, chan in enumerate(good_channels)}
    groups = {str(g["group_id"]): g for g in report.meta["joint_models"]["groups"]}
    group_spectra = {}
    for gid, group in groups.items():
        path = native / f"group-{gid}"
        gf = np.fromfile(path / "flags.bin", dtype="uint16")
        if np.all((gf & FAILURE_MASK) == 0):
            fr = np.fromfile(path / "fr.bin", dtype="float64").reshape(
                len(gf), len(group["resonator_ids"])
            )
            group_spectra[gid] = spectrum(fr, rate)[1]
    spectra = np.full((len(nu), len(coverage)), np.nan)
    band = (nu >= SUMMARY_BAND_HZ[0]) & (nu <= SUMMARY_BAND_HZ[1])
    statuses, estimates = [], []
    for col, row in enumerate(coverage.itertuples()):
        gid = str(row.group_id)
        count = 0
        if row.native_route == "joint" and gid in group_spectra:
            j = list(map(str, groups[gid]["resonator_ids"])).index(
                str(row.resonator_id)
            )
            spectra[:, col] = group_spectra[gid][:, j]
            count = 1
        elif row.native_route == "independent":
            selected = channels.loc[channels.group_id == gid, "chan"]
            columns = [lookup[ch] for ch in selected if ch in lookup]
            if columns:
                # A diagnostic summary of existing probe estimates, not a coadd.
                spectra[:, col] = np.median(channel_power[:, columns], axis=1)
                count = len(columns)
        statuses.append("measured" if count else "no_complete_valid_estimate")
        estimates.append(count)
    coverage["psd_status"] = statuses
    coverage["probe_estimates_summarized"] = estimates
    coverage["frequency_asd_5_20_hz"] = np.sqrt(np.median(spectra[band], axis=0))
    coverage["optical_snr"] = np.nan
    coverage["mapping_speed"] = np.nan
    coverage["sensitivity_status"] = "requires_optical_response_and_science_template"
    coverage.to_csv(folder / "output-quality.csv", index=False)
    np.savez_compressed(
        folder / "output-spectra.npz",
        frequency=nu,
        power=spectra,
        group=coverage.group_id.to_numpy(str),
        resonator=coverage.resonator_id.to_numpy(str),
    )
    method = {
        "sample_rate_hz": rate,
        "samples": request["samples"],
        "duration_s": request["samples"] / rate,
        "nperseg": 1024,
        "noverlap": 512,
        "window": "hann",
        "detrend": "constant",
        "scaling": "density",
        "summary_band_hz": [5, 20],
        "exclusions": "Only completely valid ordered estimates; no gap filling",
        "cleaning": "none",
        "optical_calibration": "not supplied",
        "multiple_singlet_probes": "median of their PSDs; not a combined timestream",
    }
    (folder / "output-quality-method.json").write_text(json.dumps(method, indent=2))
    return coverage, nu, spectra


def quality_panels(root):
    """Return an output-quality matrix and uncleaned frequency-noise spectra."""
    rows = []
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Affine: singlets",
            "Affine: multiplets",
            "Delay: singlets",
            "Delay: multiplets",
        ),
    )
    for plot_row, (family, name) in enumerate(
        (("Affine", "coverage"), ("Delay", "coverage-delay")), 1
    ):
        table, nu, power = measure(root / name / "measured")
        for plot_col, (kind, member) in enumerate(
            (
                ("Singlet", table.multiplicity == 1),
                ("Multiplet", table.multiplicity > 1),
            ),
            1,
        ):
            for (label, low, high), color in zip(BINS, COLORS, strict=True):
                part = table.loc[member & _selected_bin(table, low, high)]
                valid = part.psd_status == "measured"
                values = part.loc[valid, "frequency_asd_5_20_hz"]
                rows.append(
                    {
                        "Gain": family,
                        "Group type": kind,
                        "Fitted a bin": label,
                        "Fitted": len(part),
                        "Any valid": int((part.native_fraction > 0).sum()),
                        "Complete valid estimate": int(valid.sum()),
                        "Median ASD [Hz/sqrt(Hz)], 5-20 Hz": values.median(),
                    }
                )
                if valid.any():
                    median = np.sqrt(np.median(power[:, part.index[valid]], axis=1))
                    fig.add_trace(
                        go.Scattergl(
                            x=nu[1:],
                            y=median[1:],
                            name=f"a: {label}",
                            line={"color": color},
                            legendgroup=label,
                            showlegend=plot_row == 1 and plot_col == 1,
                        ),
                        row=plot_row,
                        col=plot_col,
                    )
    fig.update_xaxes(type="log", dtick="D2", title_text="Temporal frequency [Hz]")
    fig.update_yaxes(type="log", dtick="D2", title_text="Frequency ASD [Hz/√Hz]")
    fig.update_layout(
        title="Measured uncleaned frequency fluctuations: median component spectra",
        height=800,
        legend={"orientation": "h", "y": -0.16},
    )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(root / "output-quality-matrix.csv", index=False)
    return matrix.to_html(
        index=False, border=0, na_rep="--", float_format=lambda v: f"{v:.3g}"
    ), fig


def worked_example(root):
    """Show a representative valid singlet without selecting it for low noise."""
    folder = root / "coverage/measured"
    sensitivity = pd.read_csv(folder / "nonlinear-sensitivity.csv")
    sensitivity = sensitivity.loc[sensitivity.accepted_fraction == 1]
    # Choose near the median fitted nonlinearity among complete valid singlets.
    channel = int(
        sensitivity.iloc[
            (sensitivity.a_fit - sensitivity.a_fit.median()).abs().argsort().iloc[0]
        ].chan
    )
    request = json.loads((folder / "request.json").read_text())
    report = Table.read(request["ecsv"], format="ascii.ecsv")
    row = report[channel]
    shape = request["samples"], request["channels"]
    x, r = [
        np.fromfile(folder / f"native/stub/{key}.bin", dtype="float64").reshape(shape)[
            :, channel
        ]
        for key in ("x", "r")
    ]
    i, q = [
        np.fromfile(request[key], dtype="float64").reshape(shape)[:, channel]
        for key in "IQ"
    ]
    gain = complex(row["H0_re"], row["H0_im"]) + complex(row["H1_re"], row["H1_im"]) * (
        row["f_in"] - row["f_g"]
    )
    wd = complex(row["c_re"], row["c_im"]) / (
        1 - (i + 1j * q) / request["normalization"] / gain
    )
    rate = json.loads((folder / "output-quality-method.json").read_text())[
        "sample_rate_hz"
    ]
    t = np.arange(len(x)) / rate
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Effective and optical detuning",
            "Solved resonance-frequency change",
            "Measured loss versus optical detuning",
            "Measured detuning spectra",
        ),
    )
    for label, values, color in (
        ("x\N{PRIME} (effective)", wd.imag, "#e67700"),
        ("x_opt (returned x)", x, "#228be6"),
    ):
        fig.add_trace(
            go.Scattergl(
                x=t, y=values - np.median(values), name=label, line={"color": color}
            ),
            row=1,
            col=1,
        )
        nu, power = spectrum(values, rate)
        fig.add_trace(
            go.Scattergl(
                x=nu[1:],
                y=np.sqrt(power[1:]),
                name=label,
                line={"color": color},
                showlegend=False,
            ),
            row=2,
            col=2,
        )
    fr = row["f_in"] / (1 + x)
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=fr - np.median(fr),
            name="f_r",
            line={"color": "#228be6"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    binned_trajectory = runpy.run_path(
        str(Path(__file__).with_name("network_response.py"))
    )["binned_trajectory"]

    sky = json.loads((root / "references/skydip-network.json").read_text())["summary"]
    beta = sky["beta_median"]
    dx, dr = x - np.mean(x), r - np.mean(r)
    bx, br = binned_trajectory(dx, dr, rate)
    fig.add_trace(
        go.Scattergl(
            x=dx,
            y=dr,
            mode="markers",
            name="Measured samples",
            marker={"color": "#adb5bd", "size": 2, "opacity": 0.3},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bx,
            y=br,
            mode="lines+markers",
            name="1 s means",
            line={"color": "#228be6"},
            marker={"size": 4},
            customdata=np.arange(len(bx)),
            hovertemplate=(
                "Window %{customdata}<br>Δx=%{x:.4g}<br>Δr=%{y:.4g}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    extent = np.array([np.min(dx), np.max(dx)])
    fig.add_trace(
        go.Scatter(
            x=extent,
            y=beta * extent,
            name="Skydip median direction",
            line={"color": "#e67700", "dash": "dash"},
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Time [s]", row=1)
    fig.update_xaxes(
        title_text="Δx_opt (dimensionless)", tickformat=".1e", row=2, col=1
    )
    fig.update_xaxes(
        title_text="Temporal frequency [Hz]", type="log", dtick="D2", row=2, col=2
    )
    fig.update_yaxes(
        title_text="Median-subtracted detuning", tickformat=".1e", row=1, col=1
    )
    fig.update_yaxes(title_text="Δf_r [Hz]", row=1, col=2)
    fig.update_yaxes(title_text="Δr (dimensionless)", tickformat=".1e", row=2, col=1)
    fig.update_yaxes(
        title_text="ASD [1/√Hz]",
        type="log",
        dtick="D2",
        exponentformat="power",
        row=2,
        col=2,
    )
    fig.update_layout(
        title=f"145609 scan 2, channel {channel}: effective and optical detuning",
        height=800,
        legend={"orientation": "h", "y": -0.16},
    )
    return channel, fig
