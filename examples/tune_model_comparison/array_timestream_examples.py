"""Measured singlet/joint examples using the maintained fitted-group solver.

Select from sweep evidence before examining timestream noise. Keep rejected
samples and frequency conventions explicit; do not claim component/APT identity.
"""

# ruff: noqa: INP001, T201 -- executable scientific report recipe

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.table import Table

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids.model_fit import _MODEL_SCHEMA, ModelFit
from tolteca_kids.model_joint import joint_models, joint_solver
from tolteca_kids.product import open_kids_product
from tolteca_kidsproc.recovery import SampleFlag
from tolteca_kidsproc.timestream import TimestreamSolver
from tolteca_kidsproc.timestream_diagnostics import coordinate_spectra, spectral_matrix

BAND = (5, 20)


def longest_valid_interval(valid):
    """Find a contiguous interval, without closing gaps or removing samples."""
    changes = np.diff(np.r_[False, np.asarray(valid, bool), False].astype(int))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    if not len(starts):
        return 0, 0
    i = int(np.argmax(stops - starts))
    return int(starts[i]), int(stops[i])


def select_cases(responses, groups, *, nw=0, limit=0.05):
    """Choose median-quality complete groups before any timestream inspection."""
    rows = responses[(responses["nw"] == nw) & responses["selected_for_probe"]]
    choices = []
    for count, name in ((1, "singlet"), (2, "joint")):
        candidates = []
        for group in groups:
            if len(group["channels"]) != count or len(group["resonator_ids"]) != count:
                continue
            r = rows[rows["group_id"] == group["group_id"]]
            if (
                len(r) != count
                or set(r["channel"]) != set(group["channels"])
                or not np.all(r["apt_flag"] == 0)
                or not np.all(r["fit_status"] == "optimizer_converged")
                or not np.all(
                    np.isfinite(r["v3_fraction"]) & (r["v3_fraction"] < limit)
                )
                or group["fit_flag"] != 0
            ):
                continue
            candidates.append((float(np.max(r["v3_fraction"])), group))
        if not candidates:
            msg = f"No complete {name} group passes the declared sweep policy"
            raise ValueError(msg)
        median = float(np.median([c[0] for c in candidates]))
        error, group = min(
            candidates, key=lambda c: (abs(c[0] - median), c[1]["group_id"])
        )
        choices.append(
            {
                "name": name,
                "nw": nw,
                "group": group,
                "selection": {
                    "pool": len(candidates),
                    "limit": limit,
                    "median_max_channel_residual": median,
                    "selected_max_channel_residual": error,
                    "rule": (
                        "closest to median of maximum channel residual; "
                        "APT-good, converged, no group warning; chosen before solving"
                    ),
                },
            }
        )
    return choices


def fingerprint(path):
    """Record the exact local input used."""
    path = Path(path)
    with path.open("rb") as f:
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.file_digest(f, "sha256").hexdigest(),
        }


def build(prepared, legacy_manifest, output):  # noqa: PLR0915 -- sequential evidence construction
    """Solve two preselected examples without refitting or changing map products."""
    receipt = json.loads((prepared / "prepared.json").read_text())
    index = DataProdIndex.read(receipt["model_index"])
    with open_kids_product(index, interface="toltec0") as tree:
        block = tree.toltec_kids.blocks[1]
        view = ModelFit.view(block.tree)
        quality = block.tree["model_fit/quality"].to_dataset(inherit=False).load()
        payload = joint_models(view, quality, include_single=True)
        sweep_accum = float(block.tree["sweep"].attrs["iq_accumlen"])
    responses = Table.read(output / "sweep-model-responses.ecsv")
    cases = select_cases(responses, payload["groups"])
    folder = output / "timestream-examples"
    folder.mkdir(exist_ok=True)
    # Freeze choices before looking at the target I/Q or recovery results.
    (folder / "selection.json").write_text(json.dumps(cases, indent=2) + "\n")
    inputs = json.loads((prepared.parent / "inputs.json").read_text())
    source = next(r["timestream"] for r in inputs if r["interface"] == "toltec0")
    legacy = json.loads(legacy_manifest.read_text())
    legacy_source = next(
        r for r in legacy["model_equivalence"] if r["interface"] == "toltec0"
    )
    old = Table.read(legacy_source["labeled"], format="ascii.ecsv")
    old.meta["model_schema"] = copy.deepcopy(_MODEL_SCHEMA)
    old["flag"] = np.asarray(old["flag"], dtype=np.uint32)
    old["f_in"].unit = "Hz"
    parameters = _MODEL_SCHEMA["parameters"]
    assert isinstance(parameters, list)
    for parameter in parameters:
        old[parameter["name"]].unit = parameter["unit"] or None
    with open_toltec_kids(source) as raw:
        rate = float(raw["Header.Toltec.SampleFreq"].item())
        norm = float(raw["Header.Toltec.AccumLen"].item()) / sweep_accum
        assert sweep_accum == old.meta["Header.Toltec.AccumLen"]
        ticks = raw["Data.Toltec.Ts"]
        names = [
            part.strip().split()[0] for part in ticks.attrs["long_name"].split(",")
        ]
        packets = ticks.values[:, names.index("PacketCount")].astype(np.uint32)
        if not np.all(np.diff(packets) == 1):
            msg = "This example requires continuous packets; do not close gaps"
            raise ValueError(msg)
        probes = raw["Header.Toltec.ToneFreq"].values[0] + float(
            raw["Header.Toltec.LoCenterFreq"].item()
        )
        iq = (
            raw["Data.Toltec.Is"].values.astype(float)
            + 1j * raw["Data.Toltec.Qs"].values
        )
    records = []
    for case in cases:
        name, group = case["name"], case["group"]
        channels = np.asarray(group["channels"])
        np.testing.assert_allclose(probes[channels], group["probes"], rtol=0, atol=1e-6)
        solver = joint_solver(group)
        observed = iq[:, channels]
        print(
            f"Solving {name} group {group['group_id']}, "
            f"channels {channels}, {len(iq)} samples",
            flush=True,
        )
        fr, loss, flags, residual = solver(
            observed.real, observed.imag, normalization=norm
        )
        old_r, old_x, old_flags = TimestreamSolver(old, channels)(
            observed.real, observed.imag, normalization=norm
        )
        valid = (
            (
                (flags & (np.iinfo(flags.dtype).max ^ int(SampleFlag.condition_high)))
                == 0
            )
            & np.isfinite(fr).all(axis=1)
            & np.isfinite(loss).all(axis=1)
        )
        common = valid & (old_flags == 0).all(axis=1)
        start, stop = longest_valid_interval(common)
        np.savez_compressed(
            folder / f"{name}-solved.npz",
            fr=fr,
            r=loss,
            flags=flags,
            legacy_r=old_r,
            legacy_x=old_x,
            legacy_flags=old_flags,
            residual=residual,
            I=observed.real,
            Q=observed.imag,
        )
        result = {
            **case,
            "samples": len(iq),
            "f_smp_hz": rate,
            "normalization": norm,
            "valid_samples": int(valid.sum()),
            "common_samples": int(common.sum()),
            "spectral_interval": [start, stop],
            "sample_flag_counts": {
                f.name: int(np.count_nonzero(flags & int(f)))
                for f in SampleFlag
                if int(f)
            },
            "legacy_f_in_minus_actual_hz": (
                np.asarray(old["f_in"])[channels] - probes[channels]
            ).tolist(),
        }
        if stop - start < 2048:  # noqa: PLR2004 -- two Welch windows
            result["status"] = "insufficient_continuous_valid_samples"
            records.append(result)
            continue
        evaluated = solver.evaluate(fr[start:stop], loss[start:stop])
        ref = solver.evaluate(
            np.median(fr[start:stop], axis=0, keepdims=True),
            np.median(loss[start:stop], axis=0, keepdims=True),
        )
        reconstructed = evaluated.I.values + 1j * evaluated.Q.values
        err = reconstructed - observed[start:stop] / norm
        result["reconstructed_iq_rms"] = np.sqrt(
            np.mean(abs(err) ** 2, axis=0)
        ).tolist()
        result["status"] = "complete"
        result["components"] = []
        # A declared coordinate probe is not a new optical/beammap identity.
        for j, rid in enumerate(group["resonator_ids"]):
            k = int(np.argmin(abs(np.asarray(group["probes"]) - group["fr"][j])))
            pair, local = (
                evaluated.isel(chan=k, resonator=j),
                ref.isel(chan=k, resonator=j),
            )
            xp, xo, r = pair.x_prime.values, pair.x_opt.values, evaluated.r.values[:, j]
            spectra = coordinate_spectra(
                xp, xo, r, rate, D=float(local.D.item()), mu=float(local.mu.item())
            )
            spectra.to_netcdf(folder / f"{name}-{j}-spectra.nc")
            values = np.column_stack(
                (old_x[start:stop, k], xp, xo, old_r[start:stop, k], r)
            )
            frequency, cross = spectral_matrix(values, rate)
            power = cross.diagonal(axis1=1, axis2=2).real
            band = (frequency >= BAND[0]) & (frequency <= BAND[1])
            np.savez_compressed(
                folder / f"{name}-{j}-comparison.npz",
                values=values,
                time=np.arange(start, stop) / rate,
                frequency=frequency,
                cross=cross,
                psd=power,
            )
            result["components"].append(
                {
                    "resonator_id": rid,
                    "coordinate_channel": int(channels[k]),
                    "coordinate_rule": (
                        "recorded probe nearest the fitted resonance; "
                        "not an optical identity"
                    ),
                    "a_nl": group["a_nl"][j],
                    "Qr": 0.5 / group["r"][j],
                    "band_asd": np.sqrt(np.mean(power[band], axis=0)).tolist(),
                    "coordinates": ["legacy_x", "x_prime", "x_opt", "legacy_r", "r"],
                    "band_hz": [5, 20],
                    "D": float(local.D.item()),
                    "mu": float(local.mu.item()),
                    "rank": int(solver.diagnostics["rank"]),
                    "condition": float(solver.diagnostics["condition"]),
                }
            )
        if len(group["resonator_ids"]) > 1:
            # fr is component-specific without choosing a coordinate probe.
            f, c = spectral_matrix((fr[start:stop] / np.asarray(group["fr"]) - 1), rate)
            np.savez_compressed(
                folder / f"{name}-component-cross.npz", frequency=f, cross=c
            )
        records.append(result)
        (folder / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"{name}: {valid.sum()}/{len(valid)} valid; "
            f"common contiguous interval {start}:{stop}",
            flush=True,
        )
    result = {
        "observation": 152434,
        "network": 0,
        "block": 1,
        "cases": records,
        "sources": {
            "raw": fingerprint(source),
            "legacy": fingerprint(legacy_source["original"]),
            "model_index": fingerprint(receipt["model_index"]),
        },
        "interpretation": (
            "Python fitted-group solving before telescope processing, decorrelation "
            "or flux calibration. Legacy channel-effective coordinates are not "
            "separated joint detector signals. No component/APT association or "
            "optical sensitivity is asserted."
        ),
    }
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "legacy_manifest", "output"):
        parser.add_argument(key, type=Path)
    build(**vars(parser.parse_args()))
