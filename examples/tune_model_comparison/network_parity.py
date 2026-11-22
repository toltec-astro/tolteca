#!/usr/bin/env python
"""Resume saved Python parity groups in parallel after complete native solving.

Reuse a folder only with its unchanged request, model and solver revision.
This resumes numerical validation, not a new fit or calibration realization.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.table import Table
from network_coverage import compare_native

from tolteca_kids.model_joint import joint_solver


def solve_one(request, group, root):
    """Solve one ordered group without sharing continuation state across groups."""
    shape = request["samples"], request["channels"]
    channels = group["channels"]
    count = min(256, shape[0])
    i, q = [
        np.memmap(request[key], mode="r", dtype="float64", shape=shape)[
            :count, channels
        ]
        for key in "IQ"
    ]
    fr, loss, flags, *_ = joint_solver(group)(
        i, q, normalization=request["normalization"]
    )
    gid = group["group_id"]
    temporary = root / f"python-group-{gid}.partial.npz"
    np.savez_compressed(temporary, fr=fr, r=loss, flags=flags)
    temporary.replace(root / f"python-group-{gid}.npz")
    return gid


def resume(root, *, workers):
    """Complete missing group arrays, then audit all saved Python/native results."""
    request = json.loads((root / "request.json").read_text())
    # Both native routes must be complete before any comparison is reported.
    for mode in ("ecsv", "stub"):
        assert (root / "native" / mode / "summary.yaml").exists()
    report = Table.read(request["ecsv"], format="ascii.ecsv")
    pending = [
        g
        for g in report.meta["joint_models"]["groups"]
        if not (root / f"python-group-{g['group_id']}.npz").exists()
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = [executor.submit(solve_one, request, group, root) for group in pending]
        for job in as_completed(jobs):
            sys.stdout.write(f"Completed group {job.result()}\n")
            sys.stdout.flush()
    compare_native(request, report, None, root, run_native=False, reuse_python=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    resume(args.root.resolve(), workers=args.workers)
