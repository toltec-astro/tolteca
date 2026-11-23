"""Audit the native matched PCA stages and separate input and basis changes."""

# ruff: noqa: INP001, E501 -- explicit scientific diagnostic

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from array_coordinate_control import common_maps
from astropy.table import Table
from build_xprime_control import digest
from compare_maps import read_map
from netCDF4 import Dataset
from pca_trace import stage_path
from scipy.signal import welch


def chunk_bounds(raw_indices, factor, n_samples):
    """Reconstruct append-order bounds from raw inner ranges and decimation."""
    lengths = (raw_indices[:, 1] - raw_indices[:, 0] + factor) // factor
    stops = np.cumsum(lengths, dtype=int)
    if stops[-1] != n_samples:
        msg = "Raw ranges do not account for the saved samples"
        raise ValueError(msg)
    return np.column_stack((np.r_[0, stops[:-1]], stops))


def remove_modes(centered, basis):
    """Apply the saved native detector-space projection, without refitting it."""
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)
    return centered - (centered @ basis) @ basis.T


def difference_terms(legacy, v3, legacy_basis, v3_basis):
    """Decompose post-cleaning difference using the legacy basis as reference."""
    input_term = remove_modes(v3 - legacy, legacy_basis)
    basis_term = (v3 @ legacy_basis) @ legacy_basis.T - (v3 @ v3_basis) @ v3_basis.T
    return input_term, basis_term


def weighted_power(values, weights):
    """Average squared samples with the exact saved map weights."""
    return float(
        np.sum(values * values * weights[None, :]) / (len(values) * weights.sum())
    )


def read_stage(path, columns):
    """Read only the selected detector rows and preserve native metadata."""
    with Dataset(path) as data:
        result = {
            key: np.asarray(data[key][:])[:, columns]
            for key in ("signal", "flags", "det_lat", "det_lon")
        }
        result.update(
            time=np.asarray(data["TelTime"][:]),
            raw_indices=np.asarray(data["raw_scan_indices"][:], dtype=int),
            written_indices=np.asarray(data["scan_indices"][:], dtype=int),
            fs=float(data["SAMPRATE"][...]),
            units=data["signal"].units,
        )
        if "weights" in data.variables:
            result["weights"] = np.asarray(data["weights"][:])[:, columns]
            result["basis"] = [
                np.asarray(data[f"evecs_nw_0_chunk_{j}"][:, :5])[columns]
                for j in range(len(result["raw_indices"]))
            ]
        return result


def analyze(base, output):  # noqa: C901, PLR0915 -- sequential evidence audit
    """Compare native stages with exactly equal channels, samples and weights."""
    base, output = base.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = base / "pca-control"
    apt = Table.read(root / "matched-apt.ecsv")
    columns = np.flatnonzero(
        (np.asarray(apt["array"]) == 0) & (np.asarray(apt["flag"]) == 0)
    )
    nw = np.asarray(apt["nw"][columns], dtype=int)
    config = yaml.safe_load((root / "matched/legacy/redu00/config.yaml").read_text())
    factor = config["timestream"]["raw_time_chunk"]["downsample"]["factor"]
    stages, sources = {}, {}
    for name in ("legacy", "v3"):
        for stage in ("rtc", "ptc"):
            path = stage_path(root / "matched" / name, stage)
            key = f"{name}_{stage}"
            stages[key] = read_stage(path, columns)
            sources[key] = {"path": str(path), "sha256": digest(path)}
    ref = stages["legacy_rtc"]
    for item in stages.values():
        assert item["units"] == "mJy/beam"
        assert np.isfinite(item["signal"]).all()
        assert not item["flags"].any()
        for key in ("time", "det_lat", "det_lon", "raw_indices"):
            np.testing.assert_array_equal(item[key], ref[key])
    weights = stages["legacy_ptc"]["weights"]
    np.testing.assert_array_equal(weights, stages["v3_ptc"]["weights"])
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    bounds = chunk_bounds(ref["raw_indices"], factor, len(ref["time"]))
    fs = ref["fs"] / factor
    np.testing.assert_allclose(np.diff(ref["time"]), 1 / fs, atol=3e-7, rtol=0)
    centered = {name: np.empty_like(ref["signal"]) for name in ("legacy", "v3")}
    # Store aggregate spectra and one complete pair of network-mean trajectories.
    spectral_sum = np.zeros((6, 4, 129))
    common_sum = np.zeros_like(spectral_sum)
    total_samples = 0
    channel_rows, chunk_rows = [], []
    trajectories = np.empty((len(ref["time"]), 6, 4))
    max_replay_error = 0.0
    for chunk, (start, stop) in enumerate(bounds):
        sl = slice(start, stop)
        length = stop - start
        for name, values in centered.items():
            raw = stages[f"{name}_rtc"]["signal"][sl]
            values[sl] = raw - raw.mean(axis=0)
        for network in range(6):
            ids = np.flatnonzero(nw == network)
            w = weights[chunk, ids]
            x = [centered[name][sl][:, ids] for name in ("legacy", "v3")]
            y = [
                stages[f"{name}_ptc"]["signal"][sl][:, ids] for name in ("legacy", "v3")
            ]
            bases = [
                stages[f"{name}_ptc"]["basis"][chunk][ids] for name in ("legacy", "v3")
            ]
            for before, after, basis in zip(x, y, bases, strict=True):
                error = np.max(abs(remove_modes(before, basis) - after))
                max_replay_error = max(max_replay_error, float(error))
                np.testing.assert_allclose(
                    remove_modes(before, basis), after, rtol=1e-10, atol=2e-7
                )
            parts = difference_terms(*x, *bases)
            np.testing.assert_allclose(
                parts[0] + parts[1], y[1] - y[0], atol=3e-7, rtol=1e-10
            )
            variants = (x[0], x[1], y[0], y[1])
            for k, values in enumerate(variants):
                freq, p = welch(values, fs=fs, nperseg=256, noverlap=128, axis=0)
                spectral_sum[network, k] += length * np.average(p, weights=w, axis=1)
                trajectory = values @ w / w.sum()
                trajectories[sl, network, k] = trajectory
                _, pmean = welch(trajectory, fs=fs, nperseg=256, noverlap=128)
                common_sum[network, k] += length * pmean
            for j, col in enumerate(ids):
                g = np.dot(x[0][:, j], x[1][:, j]) / np.dot(x[0][:, j], x[0][:, j])
                channel_rows.append(
                    {
                        "chunk": chunk,
                        "nw": network,
                        "channel": int(apt["kids_tone"][columns[col]]),
                        "gain_v3_over_legacy": float(g),
                        "pre_correlation": float(
                            np.corrcoef(x[0][:, j], x[1][:, j])[0, 1]
                        ),
                        "pre_std_legacy": float(np.std(x[0][:, j])),
                        "pre_std_v3": float(np.std(x[1][:, j])),
                        "post_std_legacy": float(np.std(y[0][:, j])),
                        "post_std_v3": float(np.std(y[1][:, j])),
                        "weight": float(w[j]),
                    }
                )
            overlap = np.linalg.svd(bases[0].T @ bases[1], compute_uv=False)
            powers = [weighted_power(v, w) for v in variants]
            swapped = [remove_modes(x[1], bases[0]), remove_modes(x[0], bases[1])]
            chunk_rows.append(
                {
                    "chunk": chunk,
                    "nw": network,
                    "n_samples": int(length),
                    "n_channels": len(ids),
                    "sum_weights": float(w.sum()),
                    "powers": powers,
                    "v3_with_legacy_basis_power": weighted_power(swapped[0], w),
                    "legacy_with_v3_basis_power": weighted_power(swapped[1], w),
                    "input_term_power": weighted_power(parts[0], w),
                    "basis_term_power": weighted_power(parts[1], w),
                    "cross_term_power": float(
                        2 * np.sum(parts[0] * parts[1] * w) / (length * w.sum())
                    ),
                    "difference_power": weighted_power(y[1] - y[0], w),
                    "subspace_overlap": float(np.mean(overlap**2)),
                    "largest_angle_deg": float(
                        np.degrees(np.arccos(np.clip(overlap.min(), -1, 1)))
                    ),
                }
            )
        total_samples += length
    spectral_sum /= total_samples
    common_sum /= total_samples
    Table(channel_rows).write(output / "pca-channel-chunks.ecsv", overwrite=True)
    np.savez_compressed(
        output / "pca-spectra.npz",
        frequency=freq,
        psd=spectral_sum,
        mean_psd=common_sum,
        time=ref["time"] - ref["time"][0],
        network_means=trajectories,
    )
    rows = []
    for network in range(6):
        chunks = [r for r in chunk_rows if r["nw"] == network]
        scale = np.array([r["n_samples"] * r["sum_weights"] for r in chunks])

        def average(key, chunks=chunks, scale=scale):
            return np.average([r[key] for r in chunks], weights=scale, axis=0)

        powers = average("powers")
        rows.append(
            {
                "nw": network,
                "channels": chunks[0]["n_channels"],
                "rms_mJy_beam": np.sqrt(powers).tolist(),
                "v3_with_legacy_basis_rms": float(
                    np.sqrt(average("v3_with_legacy_basis_power"))
                ),
                "legacy_with_v3_basis_rms": float(
                    np.sqrt(average("legacy_with_v3_basis_power"))
                ),
                **{
                    k: float(average(k))
                    for k in (
                        "input_term_power",
                        "basis_term_power",
                        "cross_term_power",
                        "difference_power",
                        "subspace_overlap",
                        "largest_angle_deg",
                    )
                },
            }
        )
    maps = []
    for name in ("legacy", "v3"):
        [path] = (root / "matched" / name / "redu00/152434/raw").glob("*_a1100_*.fits")
        maps.append(read_map(path))
    aligned, _, annulus = common_maps(maps)
    result = {
        "scope": "v2 tune.txt coefficients versus V3 x_opt in the same pinned Citlali engine",
        "n_channels": len(columns),
        "n_samples": len(ref["time"]),
        "n_chunks": len(bounds),
        "fs_Hz": fs,
        "sources": sources,
        "selection": json.loads((root / "selection.json").read_text()),
        "all_samples_unflagged": True,
        "weights_bitwise_equal": True,
        "pointing_time_bitwise_equal": True,
        "pca_replay_max_abs_mJy_beam": max_replay_error,
        "stage_order": [
            "legacy pre-PCA",
            "V3 pre-PCA",
            "legacy post-PCA",
            "V3 post-PCA",
        ],
        "chunk_bounds": bounds.tolist(),
        "written_chunk_indices": ref["written_indices"].tolist(),
        "chunk_index_note": "Exported start indices assume equal-sized chunks; analysis uses cumulative decimated raw-inner lengths and verifies exact native PCA replay.",
        "networks": rows,
        "chunks": chunk_rows,
        "matched_map_rms_mJy_beam": [
            float(np.std(m["signal"][annulus], ddof=1)) for m in aligned
        ],
        "matched_map_sources": [
            {"path": str(m["path"]), "sha256": digest(m["path"])} for m in maps
        ],
        "map_weight_equal": bool(
            np.array_equal(aligned[0]["weight"], aligned[1]["weight"])
        ),
        "map_coverage_equal": bool(
            np.array_equal(aligned[0]["coverage"], aligned[1]["coverage"])
        ),
        "common_annulus_pixels": int(annulus.sum()),
        "limits": "Pre-PCA is after filtering, decimation and held APT flux calibration. PSD includes source and correlated sky signal, not validated noise-only sensitivity. PCA bases are unweighted as in native cleaning; exact map weights are used for diagnostic averages. Exchanged bases are diagnostics, not proposed calibrations.",
    }
    (output / "pca-trace.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    analyze(**vars(parser.parse_args()))
