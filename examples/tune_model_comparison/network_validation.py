#!/usr/bin/env python
"""Validate complete native transport and the known controlled trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.table import Table

from tolteca_kids.model_fit import ModelFit
from tolteca_kids.model_joint import joint_models, joint_solver
from tolteca_kids.product import open_sweep_product
from tolteca_kidsproc.recovery import SampleFlag

FAILURE_MASK = 65535 ^ int(SampleFlag.condition_high)


def transport(folder):
    """Compare every byte of native output from ECSV and stub model readers."""
    left, right = folder / "native/ecsv", folder / "native/stub"
    files = sorted(p.relative_to(left) for p in left.rglob("*.bin"))
    other = sorted(p.relative_to(right) for p in right.rglob("*.bin"))
    pairs = []
    for path in files:
        checksums = [
            hashlib.sha256((base / path).read_bytes()).hexdigest()
            for base in (left, right)
        ]
        pairs.append({"file": str(path), "equal": checksums[0] == checksums[1]})
    metadata = [
        yaml.safe_load((base / "summary.yaml").read_text()) for base in (left, right)
    ]
    result = {
        "same_files": files == other,
        "metadata_equal": metadata[0] == metadata[1],
        "binary_equal": all(p["equal"] for p in pairs),
        "files": pairs,
    }
    (folder / "transport.json").write_text(json.dumps(result, indent=2))
    return result


def controlled_truth(index, base):
    """Compare accepted controlled samples with the injected loss and frequency.

    Errors are recorded even if they exceed the parity tolerances. Failure flags
    remain failures; a good comparison among survivors cannot rehabilitate them.
    """
    results = []
    with open_sweep_product(index) as tree:
        for block_id in (0, 1):
            folder = base / f"block{block_id}"
            request = json.loads((folder / "request.json").read_text())
            report = Table.read(request["ecsv"], format="ascii.ecsv")
            native = folder / "native/stub"
            shape = request["samples"], request["channels"]
            r, x = [
                np.fromfile(native / f"{name}.bin", dtype="float64").reshape(shape)
                for name in ("r", "x")
            ]
            flags = np.fromfile(native / "flags.bin", dtype="uint16").reshape(shape)
            block = tree.toltec_kids.blocks[block_id]
            quality = block.tree[ModelFit.result_path()]["quality"].to_dataset()
            view = ModelFit.view(tree, block=block_id)
            groups = joint_models(view, quality, include_single=True)["groups"]
            for group in groups:
                results.extend(_group_truth(folder, group, report, (r, x, flags)))
    result = pd.DataFrame(results)
    result.to_csv(base / "controlled-truth.csv", index=False)
    return result


def _group_truth(folder, group, report, signals):
    solver = joint_solver(group)
    gid, channels = group["group_id"], group["channels"]
    r, x, flags = signals
    n = len(r)
    truth_r = solver.r * (1 + np.linspace(0, 0.003, n)[:, None])
    truth_fr = solver.fr * (1 + 2 * solver.r * np.linspace(0, 0.005, n)[:, None])
    rows = []
    joint = folder / "native/stub" / f"group-{gid}"
    if joint.exists():
        nr, nfr = [
            np.fromfile(joint / f"{name}.bin", dtype="float64").reshape(
                n, len(solver.r)
            )
            for name in ("r", "fr")
        ]
        good = (np.fromfile(joint / "flags.bin", dtype="uint16") & FAILURE_MASK) == 0
        for j, rid in enumerate(group["resonator_ids"]):
            rows.append(
                _errors(
                    folder.name,
                    gid,
                    rid,
                    "joint",
                    good,
                    (nr[:, j], nfr[:, j]),
                    (truth_r[:, j], truth_fr[:, j]),
                )
            )
    else:
        # A selected singlet may be represented by more than one readout channel.
        for chan in channels:
            good = (flags[:, chan] & FAILURE_MASK) == 0
            fr = float(report["f_in"][chan]) / (1 + x[:, chan])
            rows.append(
                _errors(
                    folder.name,
                    gid,
                    group["resonator_ids"][0],
                    f"channel:{chan}",
                    good,
                    (r[:, chan], fr),
                    (truth_r[:, 0], truth_fr[:, 0]),
                )
            )
    return rows


def _errors(block, gid, rid, route, good, actual, expected):  # noqa: PLR0913, PLR0917
    # Identifiers and the two physical axes remain explicit in this evidence row.
    dr, df = [abs(a[good] - b[good]) for a, b in zip(actual, expected, strict=True)]
    return {
        "block": block,
        "group_id": gid,
        "resonator_id": rid,
        "route": route,
        "accepted_samples": int(good.sum()),
        "max_abs_r_error": float(dr.max()) if good.any() else None,
        "max_abs_frequency_error_hz": float(df.max()) if good.any() else None,
    }


def validate(root):
    """Save transport and known-truth evidence for both complete network fits."""
    products = root / "dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc"
    result = {}
    for name, realization in (
        ("coverage", "0000-models"),
        ("coverage-delay", "0001-models"),
    ):
        base = root / name
        result[name] = {
            part: transport(base / part) for part in ("block0", "block1", "measured")
        }
        truth = controlled_truth(products / realization / "index.yaml", base)
        result[name]["truth"] = {
            "max_abs_r_error": float(truth.max_abs_r_error.max()),
            "max_abs_frequency_error_hz": float(truth.max_abs_frequency_error_hz.max()),
        }
    (root / "validation.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    validate(parser.parse_args().root.resolve())
