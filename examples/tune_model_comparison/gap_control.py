"""Prepare and audit the full intended 152434 population with explicit gap policy."""

# ruff: noqa: INP001, E501 -- scientific comparison constants

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from array_coordinate_control import ARRAYS, common_maps
from astropy.table import Table
from build_xprime_control import digest
from compare_maps import read_map
from netCDF4 import Dataset
from pca_flag_projection import center_valid, masked_projection
from pca_trace import capture_config, stage_path
from pca_trace_analysis import chunk_bounds

RUNS = {
    "legacy": ("legacy-common", "numerical"),
    "v3-physical": ("python-full", "physical"),
    "v3-numerical": ("python-full", "numerical"),
}


def prepare(base):
    """Use the original common APT; never exclude channels based on sample history."""
    base = base.resolve()
    apt = base / "prepared/common-apt.ecsv"
    for name, (model, acceptance) in RUNS.items():
        folder = base / "gap-control" / name
        folder.mkdir(parents=True, exist_ok=True)
        source = yaml.safe_load((base / "prepared" / model / "config.yaml").read_text())
        config = capture_config(source, folder, apt)
        config["kids"]["solver"]["sample_acceptance"] = acceptance
        # All weights come from the same APT sensitivity and flux calibration.
        assert (
            config["timestream"]["processed_time_chunk"]["weighting"]["type"]
            == "approximate"
        )
        destination = folder / "source.yaml"
        destination.write_text(yaml.safe_dump(config, sort_keys=False))


def audit(base, output):  # noqa: PLR0915 -- sequential native evidence checks
    """Check identities, fixed weights, retained flags, gap masks and PCA outputs."""
    base, output = base.resolve(), output.resolve()
    apt_path = base / "prepared/common-apt.ecsv"
    apt = Table.read(apt_path)
    n = len(apt)
    ids = np.asarray(apt["nw"], dtype=int)
    arrays = np.asarray(apt["array"], dtype=int)
    eligible = np.asarray(apt["flag"]) == 0
    root = base / "gap-control"
    records, data, maps = {}, {}, {}
    for name in RUNS:
        raw_path, post_path = (
            stage_path(root / name, stage) for stage in ("rtc", "ptc")
        )
        receipt_path = root / name / "redu00/execution.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["exit_code"] == 0
        with Dataset(raw_path) as raw, Dataset(post_path) as post:
            np.testing.assert_equal(post["apt_nw"][:], ids)
            np.testing.assert_equal(post["apt_kids_tone"][:], apt["kids_tone"])
            x = np.asarray(raw["signal"][:])
            y = np.asarray(post["signal"][:])
            f = np.asarray(post["flags"][:], dtype=bool)
            sf = np.asarray(post["kids_sample_flag"][:], dtype=int)
            weights = np.asarray(post["weights"][:])
            assert np.isfinite(x).all()
            assert np.isfinite(y).all()
            assert x.shape[1] == n
            assert not np.any(np.asarray(raw["flags"][:], bool) & ~f)
            exclusion_mask = int(post["kids_sample_flag"].map_exclusion_mask)
            assert not np.any(((sf & exclusion_mask) != 0) & ~f)
            policy = post["kids_sample_flag"].sample_acceptance
            assert policy == RUNS[name][1]
            metadata = yaml.safe_load(post.kids_readout)
            assert metadata["filled_samples_map_eligible"] is False
            bounds = chunk_bounds(
                np.asarray(post["raw_scan_indices"][:], int), 2, len(x)
            )
            # Verify the exact native masked least-squares projection on a1100.
            replay_error = 0.0
            means = np.empty((len(x), 6, 2))
            for chunk, (start, stop) in enumerate(bounds):
                basis = np.asarray(post[f"evecs_nw_0_chunk_{chunk}"][:, :5])
                for nw in range(6):
                    cols = np.flatnonzero(ids == nw)
                    ff = f[start:stop][:, cols]
                    xx = center_valid(x[start:stop][:, cols], ff)
                    yy = y[start:stop][:, cols]
                    _, _, expected, _ = masked_projection(xx, ff, basis[cols])
                    error = float(np.max(np.abs(expected[~ff] - yy[~ff])))
                    replay_error = max(replay_error, error)
                    np.testing.assert_allclose(
                        expected[~ff], yy[~ff], atol=3e-7, rtol=1e-9
                    )
                    w = np.where(ff, 0, weights[chunk, cols])
                    denom = w.sum(axis=1)
                    means[start:stop, nw, 0] = (xx * w).sum(axis=1) / denom
                    means[start:stop, nw, 1] = (yy * w).sum(axis=1) / denom
            rows = []
            for a, array in enumerate(ARRAYS):
                take = arrays == a
                live = ~f[:, take]
                rows.append(
                    {
                        "array": array,
                        "intended_channels": int((take & eligible).sum()),
                        "contributing_channels": int(live.any(axis=0).sum()),
                        "partially_flagged_channels": int(
                            (live.any(axis=0) & ~live.all(axis=0)).sum()
                        ),
                        "accepted_channel_samples": int(live.sum()),
                        "rejected_channel_samples": int(f[:, take & eligible].sum()),
                    }
                )
            data[name] = {
                "flags": f,
                "solver_flags": sf,
                "weights": weights,
                "time": np.asarray(post["TelTime"][:]),
                "means": means,
            }
            records[name] = {
                "policy": policy,
                "map_exclusion_mask": exclusion_mask,
                "arrays": rows,
                "masked_pca_replay_max_abs_mJy_beam": replay_error,
                "sources": {
                    "rtc": {"path": str(raw_path), "sha256": digest(raw_path)},
                    "ptc": {"path": str(post_path), "sha256": digest(post_path)},
                    "execution": {
                        "path": str(receipt_path),
                        "sha256": digest(receipt_path),
                    },
                },
            }
        maps[name] = []
        for array in ARRAYS:
            [path] = (root / name / "redu00/152434/raw").glob(f"*_{array}_*.fits")
            maps[name].append(read_map(path))
    ref = data["legacy"]
    for values in data.values():
        np.testing.assert_equal(values["weights"], ref["weights"])
        np.testing.assert_equal(values["time"], ref["time"])
    np.testing.assert_equal(
        data["v3-physical"]["solver_flags"], data["v3-numerical"]["solver_flags"]
    )
    np.testing.assert_equal(data["legacy"]["flags"], data["v3-numerical"]["flags"])
    map_rows = []
    for a, array in enumerate(ARRAYS):
        aligned, _, annulus = common_maps([maps[name][a] for name in RUNS])
        map_rows.append(
            {
                "array": array,
                "common_annulus_pixels": int(annulus.sum()),
                "rms_mJy_beam": [float(np.std(m["signal"][annulus])) for m in aligned],
                "map_weight_equal": [
                    bool(np.array_equal(aligned[0]["weight"], m["weight"]))
                    for m in aligned
                ],
                "map_coverage_equal": [
                    bool(np.array_equal(aligned[0]["coverage"], m["coverage"]))
                    for m in aligned
                ],
            }
        )
    result = {
        "runs": records,
        "order": list(RUNS),
        "maps": map_rows,
        "apt": {"path": str(apt_path), "sha256": digest(apt_path)},
        "n_samples": len(ref["time"]),
        "weights_bitwise_equal": True,
        "numerical_pair_sample_masks_bitwise_equal": True,
        "solver_flags_preserved_between_policies": True,
        "limits": "Same intended APT population and weights. Physical policy can change accepted samples and map coverage; numerical policy retains finite domain/stability violations, not failed solutions. RMS is not validated sensitivity.",
    }
    (output / "gap-control.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        output / "gap-trajectories.npz",
        time=ref["time"],
        **{name: values["means"] for name, values in data.items()},
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "audit"))
    parser.add_argument("base", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.base)
    else:
        audit(args.base, args.base / "report")
