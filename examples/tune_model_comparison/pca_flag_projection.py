"""Test whether flagged stored values leak through a saved native PCA basis."""

# ruff: noqa: INP001, E501, PLR2004 -- measured native projection audit

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table
from netCDF4 import Dataset
from pca_trace import stage_path
from pca_trace_analysis import weighted_power


def center_valid(signal, flags):
    """Use the native per-channel mean on valid samples; retain stored values."""
    valid = ~flags
    counts = valid.sum(axis=0)
    means = np.divide(
        np.where(valid, signal, 0).sum(axis=0),
        counts,
        out=np.zeros(signal.shape[1]),
        where=counts > 0,
    )
    return signal - means


def masked_projection(centered, flags, basis):
    """Compare native projection with zeroed flags and valid-sample least squares."""
    valid = ~flags
    rhs = np.where(valid, centered, 0) @ basis
    gram = np.einsum("ti,ia,ib->tab", valid.astype(float), basis, basis, optimize=True)
    eigen_min = np.linalg.eigvalsh(gram)[:, 0]
    if np.any(eigen_min < 1e-10):
        msg = "Insufficient valid probes to estimate every saved PCA mode"
        raise ValueError(msg)
    amplitudes = np.linalg.solve(gram, rhs[..., None])[..., 0]
    ordinary = centered - (centered @ basis) @ basis.T
    zeroed = centered - rhs @ basis.T
    least_squares = centered - amplitudes @ basis.T
    leakage = (np.where(flags, centered, 0) @ basis) @ basis.T
    np.testing.assert_allclose(zeroed - ordinary, leakage, atol=1e-6, rtol=1e-9)
    return ordinary, zeroed, least_squares, leakage


def build(base, output):
    """Hold the 1855 valid targets fixed and vary only the handling of other probes."""
    base, output = base.resolve(), output.resolve()
    root = base / "pca-control"
    audit = json.loads((output / "pca-trace.json").read_text())
    bounds = np.asarray(audit["chunk_bounds"])
    apt = Table.read(root / "matched-apt.ecsv")
    selected = (np.asarray(apt["array"]) == 0) & (np.asarray(apt["flag"]) == 0)
    stop_col = np.flatnonzero(np.asarray(apt["array"]) == 0)[-1] + 1
    networks = np.asarray(apt["nw"][:stop_col], dtype=int)
    selected = selected[:stop_col]
    rows: list[dict[str, Any]] = []
    validation_error = 0.0
    trajectories = np.empty((audit["n_samples"], 6, 2, 4))
    for model_idx, name in enumerate(("legacy", "v3")):
        with (
            Dataset(stage_path(root / "capture" / name, "rtc")) as raw,
            Dataset(stage_path(root / "capture" / name, "ptc")) as post,
            Dataset(stage_path(root / "matched" / name, "rtc")) as fixed_raw,
            Dataset(stage_path(root / "matched" / name, "ptc")) as fixed_post,
        ):
            signal = np.asarray(raw["signal"][:, :stop_col])
            flags = np.asarray(raw["flags"][:, :stop_col], dtype=bool)
            after = np.asarray(post["signal"][:, :stop_col])
            strict = np.asarray(fixed_post["signal"][:, :stop_col])
            np.testing.assert_equal(
                signal[:, selected],
                np.asarray(fixed_raw["signal"][:, :stop_col])[:, selected],
            )
            weights = np.asarray(fixed_post["weights"][:, :stop_col])
            for chunk, (start, stop) in enumerate(bounds):
                sl = slice(start, stop)
                basis_all = np.asarray(post[f"evecs_nw_0_chunk_{chunk}"][:stop_col, :5])
                for nw in range(6):
                    cols = np.flatnonzero(networks == nw)
                    keep = selected[cols]
                    w = weights[chunk, cols][keep]
                    x = center_valid(signal[sl][:, cols], flags[sl][:, cols])
                    f = flags[sl][:, cols]
                    y0, yz, yl, leak = masked_projection(x, f, basis_all[cols])
                    saved = after[sl][:, cols]
                    validation_error = max(
                        validation_error,
                        float(np.max(abs(y0[:, keep] - saved[:, keep]))),
                    )
                    np.testing.assert_allclose(
                        y0[:, keep], saved[:, keep], atol=2e-7, rtol=1e-9
                    )
                    ys = strict[sl][:, cols]
                    variants = (y0[:, keep], yz[:, keep], yl[:, keep], ys[:, keep])
                    for k, y in enumerate(variants):
                        trajectories[sl, nw, model_idx, k] = y @ w / w.sum()
                    rows.append(
                        {
                            "model": name,
                            "nw": nw,
                            "chunk": chunk,
                            "n_samples": int(stop - start),
                            "sum_weights": float(w.sum()),
                            "powers": [weighted_power(y, w) for y in variants],
                            "leakage_power": weighted_power(leak[:, keep], w),
                            "partially_flagged_probes": int(
                                (f.any(axis=0) & ~f.all(axis=0)).sum()
                            ),
                            "fully_flagged_probes": int(f.all(axis=0).sum()),
                            "common_mean_powers": [
                                float(np.mean((y @ w / w.sum()) ** 2)) for y in variants
                            ],
                        }
                    )
    summary = []
    for name in ("legacy", "v3"):
        for nw in range(6):
            use = [r for r in rows if r["model"] == name and r["nw"] == nw]
            weight = [r["n_samples"] * r["sum_weights"] for r in use]
            summary.append(
                {
                    "model": name,
                    "nw": nw,
                    "rms_mJy_beam": np.sqrt(
                        np.average([r["powers"] for r in use], weights=weight, axis=0)
                    ).tolist(),
                    "common_mean_rms_mJy_beam": np.sqrt(
                        np.average(
                            [r["common_mean_powers"] for r in use],
                            weights=weight,
                            axis=0,
                        )
                    ).tolist(),
                    "leakage_rms_mJy_beam": float(
                        np.sqrt(
                            np.average(
                                [r["leakage_power"] for r in use], weights=weight
                            )
                        )
                    ),
                    "partially_flagged_probe_chunks": sum(
                        r["partially_flagged_probes"] for r in use
                    ),
                }
            )
    result = {
        "scope": "Captured native PCA on the same always-valid target channels, using matched-run weights",
        "variants": [
            "native captured basis and projection",
            "flagged values zeroed in projection",
            "valid-sample least squares with captured basis",
            "native strict population and refitted basis",
        ],
        "rows": summary,
        "chunks": rows,
        "native_replay_max_abs_mJy_beam": validation_error,
        "target_prePCA_samples_identical": True,
        "limits": "Zeroed/least-squares projections are diagnostic replays using the saved basis, not production changes or recalibrated science maps.",
    }
    (output / "pca-flag-projection.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    np.savez_compressed(
        output / "pca-flag-trajectories.npz", network_means=trajectories
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    build(**vars(parser.parse_args()))
