"""Render measured evidence for each qualified detector without new fitting."""
# ruff: noqa: INP001 -- standalone example helpers are loaded by runpy.

import json
import runpy
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import zarr
from plotly.subplots import make_subplots


def render_cases(root, output, table, sources, spectra, frequency):  # noqa: PLR0913, PLR0917, PLR0915 -- six measured panels, with separate input and output owners.
    """Save small per-detector figures; load only the selected case in the browser."""
    active = runpy.run_path(str(Path(__file__).with_name("network_active.py")))
    response = runpy.run_path(str(Path(__file__).with_name("network_response.py")))
    selected = table[table.population == "active_suitable"].sort_values(
        ["family", "matched_scans", "match_residual_median_hz"],
        ascending=[True, False, True],
    )
    destination = output / "cases"
    destination.mkdir(exist_ok=True)
    cases = []
    for family, name, run in active["FAMILIES"]:
        folder = root / name / "measured"
        product = (
            root / "dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc" / run
        )
        block = zarr.open_group(product / "array.zarr", mode="r")["toltec0/blocks/1"]
        if not isinstance(block, zarr.Group):
            msg = "Expected a saved acquired-block group"
            raise TypeError(msg)
        samples = active["frame"](
            block["model_fit/tolteca_kidsproc.analysis.model_fit/samples"]
        )
        records = {
            (item["group_id"], item["resonator_id"]): (item, x, r)
            for item, x, r in response["component_records"](folder)
        }
        for row in selected[selected.family == family].itertuples():
            _, x, loss = records[row.group_id, row.resonator_id]
            probe = int(row.probe)
            key = f"{family}-{row.group_id}-{row.resonator_id}"
            record = sources["response_by_uid"][int(row.apt_uid)]
            ss = samples[
                (samples.group_id == row.group_id) & (samples.chan == row.probe)
            ]
            f = ss.frequency.to_numpy()
            iq = ss.I.to_numpy() + 1j * ss.Q.to_numpy()
            model = ss.I_model.to_numpy() + 1j * ss.Q_model.to_numpy()
            _, raw = active["_coordinate_inputs"](folder)
            measured = raw[:, probe]
            w, _ = active["independent_coordinates"](folder, probe)
            xb, rb = response["binned_trajectory"](x, loss, 122.0703125)
            xpb, _ = response["binned_trajectory"](w.imag, loss, 122.0703125)
            xb, rb, xpb = (v - v.mean() for v in (xb, rb, xpb))
            points = record["points"]
            # Express the Skydip and trajectory coordinates at the same probe.
            sx = np.array([row.f_probe_hz / p["fr"] - 1 for p in points])
            sr = np.array([p["r"] for p in points])
            airmass = [p["airmass"] for p in points]
            sx, sr = sx - sx.mean(), sr - sr.mean()
            fig = make_subplots(
                rows=3,
                cols=2,
                subplot_titles=(
                    "Acquired calibration sweep and model",
                    "IQ sweep and fixed-probe timestream",
                    "Independent Skydip frequency/loss response",
                    "145609 trajectory: 1-s averages",
                    "Measured detuning time series",
                    "Uncleaned coordinate and signal-referred noise",
                ),
                vertical_spacing=0.1,
                horizontal_spacing=0.12,
            )

            def trace(  # noqa: PLR0913, PLR0917 -- explicit plot coordinates and trace style.
                x, y, name, color, plot_row, col, mode="lines", figure=fig, **options
            ):
                figure.add_trace(
                    go.Scattergl(
                        x=x,
                        y=y,
                        name=name,
                        mode=mode,
                        line={"color": color},
                        marker={"color": color, "size": 5},
                        showlegend=False,
                        **options,
                    ),
                    row=plot_row,
                    col=col,
                )

            trace(
                (f - row.f_probe_hz) / 1000,
                abs(iq),
                "Measured sweep",
                "#868e96",
                1,
                1,
                "markers",
            )
            trace(
                (f - row.f_probe_hz) / 1000, abs(model), "Fitted sweep", "#1971c2", 1, 1
            )
            trace(iq.real, iq.imag, "Measured sweep", "#868e96", 1, 2, "markers")
            trace(model.real, model.imag, "Fitted sweep", "#1971c2", 1, 2)
            trace(
                measured.real[::16],
                measured.imag[::16],
                "Measured timestream",
                "#e67700",
                1,
                2,
                "markers",
            )
            fig.add_trace(
                go.Scatter(
                    x=sx,
                    y=sr,
                    mode="markers",
                    name="Skydip",
                    marker={"color": airmass, "colorscale": "Viridis", "size": 8},
                    customdata=airmass,
                    hovertemplate="Δx=%{x:.4g}<br>Δr=%{y:.4g}<br>Airmass=%{customdata:.3f}<extra></extra>",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            trace(sx, row.beta_at_probe * sx, "Skydip response law", "#1971c2", 2, 1)
            trace(xb, rb, "Optical coordinate", "#1971c2", 2, 2, "markers")
            trace(xpb, rb, "Effective coordinate", "#e67700", 2, 2, "markers")
            line_x = np.array([min(xb.min(), xpb.min()), max(xb.max(), xpb.max())])
            trace(
                line_x,
                row.beta_at_probe * line_x,
                "Independent Skydip direction",
                "#212529",
                2,
                2,
            )
            times = np.arange(len(xb)) * 122 / 122.0703125
            trace(times, xb * row.f_probe_hz, "Optical coordinate", "#1971c2", 3, 1)
            trace(times, xpb * row.f_probe_hz, "Effective coordinate", "#e67700", 3, 1)
            powers = spectra[key]
            for values, label, color in (
                (np.sqrt(powers[0]), "Optical coordinate", "#1971c2"),
                (np.sqrt(powers[1]), "Loss", "#868e96"),
                (np.sqrt(powers[2]), "Effective coordinate", "#e67700"),
                (
                    np.sqrt(powers[2]) / abs(row.xprime_signal_transfer),
                    "Signal-referred effective coordinate",
                    "#212529",
                ),
            ):
                trace(frequency[1:], values[1:], label, color, 3, 2)
            for plot_row, col, xtitle, ytitle in (
                (1, 1, "Frequency - probe [kHz]", "|IQ| [acquisition units]"),
                (1, 2, "I [acquisition units]", "Q [acquisition units]"),
                (2, 1, "Δx_opt [dimensionless]", "Δr [dimensionless]"),
                (2, 2, "Δx_opt or Δx-prime [dimensionless]", "Δr [dimensionless]"),
                (3, 1, "Time [s]", "f_probe * centered detuning [Hz]"),
                (3, 2, "Temporal frequency [Hz]", "ASD [1/√Hz]"),
            ):
                fig.update_xaxes(
                    title_text=xtitle, row=plot_row, col=col, exponentformat="power"
                )
                fig.update_yaxes(
                    title_text=ytitle, row=plot_row, col=col, exponentformat="power"
                )
            fig.update_xaxes(type="log", row=3, col=2)
            fig.update_yaxes(type="log", row=3, col=2)
            fig.update_layout(
                template="plotly_white",
                height=1300,
                margin={"t": 45, "b": 65},
                font={"family": "system-ui", "size": 14},
            )
            fig.update_annotations(font_size=15)
            filename = f"{key}.json"
            (destination / filename).write_text(fig.to_json())
            cases.append(
                {
                    "file": filename,
                    "label": f"{family}: APT {int(row.apt_uid)} / channel {probe}",
                    "description": f"Group {row.group_id}, "
                    f"component {row.resonator_id}; "
                    f"a={row.a_nl:.3f}; Qr={row.Qr:.0f}; "
                    f"sweep residual/variation={row.residual_to_sweep_variation:.2%}; "
                    f"{row.matched_scans}/7 unique Skydip scans. "
                    f"Skydip β={row.beta_at_probe:.4f}; "
                    f"145609 1-s slope={row.beta_1s:.4f}.",
                }
            )
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (output / "inspect.html").write_text(
        Path(__file__).with_suffix(".html").read_text()
    )
