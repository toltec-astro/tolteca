"""Replay and time phase fitting without publishing products or instrument settings.

Run in the delivery environment with --data-dir pointing to the reference
network's raw files. Input reduction is outside the timed region; fit assembly
and commit computation are timed separately. --perturbations adds nine-scale
checks. The output directory contains JSON evidence and scalar result arrays.
"""

# ruff: noqa: INP001, T201, PLR2004 -- Standalone numerical benchmark with stdout results.

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
from tollan.utils.log import reset_logger

from tolteca_kids.drive_fit import computation as comp
from tolteca_kids.drive_fit.step import DriveFitStepConfig
from tolteca_kids.drive_fit_commit import DriveFitCommitStep, DriveFitCommitStepConfig
from tolteca_kidsproc import _phase_fit, _phase_fit_solver


def summarize(data, commit):
    """Select scalar arrays needed to compare fits and committed powers."""
    arrays = {
        name: value.values for name, value in data.data_vars.items() if value.ndim <= 2
    }
    arrays.update(
        {f"commit_{name}": value.values for name, value in commit.data_vars.items()}
    )
    return arrays


def run(args):  # noqa: C901, PLR0915 -- Standalone evidence collection.
    """Collect timings and optional perturbation comparisons on prepared sweeps."""
    sources = sorted(
        args.data_dir.rglob(f"toltec{args.network}_{args.obsnum:06d}_*targsweep.nc")
    )
    targets = sorted(
        args.data_dir.rglob(f"toltec{args.network}_{args.target:06d}_*targsweep.nc")
    )
    if not sources or len(targets) != 1:
        msg = "Need one drive-fit sequence and exactly one target sweep"
        raise ValueError(msg)
    configs = {
        "legacy": DriveFitStepConfig(phase_fit_method="legacy"),
        "analytic": DriveFitStepConfig(),
        "retry": DriveFitStepConfig(phase_fit_retry=True),
    }
    # Validate sequence identity/layout with the production reader before caching.
    comp.fit_sequence(sources, configs["legacy"])
    sweeps = sorted(
        [comp._read_sweep(path, None) for path in sources],
        key=lambda sweep: -sweep.meta.atten_drive,
    )
    attenuation = np.array([s.meta.atten_drive for s in sweeps])
    target = comp._read_sweep(targets[0], None)
    original_channel = comp.fit_channel
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_sha256": {
            module.__name__: hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (comp, _phase_fit, _phase_fit_solver)
        },
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sources, *targets]
        },
        "configuration": {
            name: cfg.model_dump(mode="json") for name, cfg in configs.items()
        },
        "timing": [],
        "perturbations": [],
    }

    def one(kind, scale, label, *, save):
        per_channel = []

        def channel(frequency, signal, **kwargs):
            start = time.perf_counter()
            result = original_channel(frequency, signal * scale, **kwargs)
            per_channel.append(time.perf_counter() - start)
            return result

        comp.fit_channel = channel
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                fitted = comp._fit(sweeps, attenuation, configs[kind])
                fit_seconds = time.perf_counter() - start
                # The current matcher prints a table; retain benchmark output only.
                with contextlib.redirect_stdout(io.StringIO()):
                    start = time.perf_counter()
                    committed = DriveFitCommitStep.compute(
                        target, fitted, DriveFitCommitStepConfig()
                    )
                    commit_seconds = time.perf_counter() - start
        finally:
            comp.fit_channel = original_channel
        data = fitted.data
        result = {
            "kind": kind,
            "scale": scale,
            "label": label,
            "fit_seconds": fit_seconds,
            "commit_seconds": commit_seconds,
            "median_channel_ms": float(1000 * np.median(per_channel)),
            "p99_channel_ms": float(1000 * np.percentile(per_channel, 99)),
            "max_channel_ms": float(1000 * np.max(per_channel)),
            "accepted": fitted.n_success,
            "fit_failures": int((data.fit_flag != 0).sum()),
            "successful_retried_fits": int(data.fit_retried.sum()),
            "attenuation_exact": committed.data.attrs["atten_drive_exact"],
            "attenuation_global": committed.data.attrs["atten_drive_global"],
            "warnings": len(caught),
        }
        if save:
            np.savez_compressed(
                args.output / f"{label}.npz", **summarize(data, committed.data)
            )
        print(json.dumps(result), flush=True)
        return result

    def checkpoint():
        (args.output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")

    modes = list(configs)
    for kind in modes:
        one(kind, 1.0, f"warmup_{kind}", save=False)
    for repeat in range(args.repeats):
        for offset in range(len(modes)):
            kind = modes[(repeat + offset) % len(modes)]
            evidence["timing"].append(
                one(kind, 1.0, f"timing_{repeat}_{kind}", save=False)
            )
            checkpoint()
    scales = [("base", 1.0)]
    if args.perturbations:
        scales += [
            (f"{sign}1e{exponent}", 1 + multiplier * 10.0 ** (-exponent))
            for exponent in (15, 14, 13, 12)
            for sign, multiplier in [("plus", 1), ("minus", -1)]
        ]
    for kind in modes:
        for suffix, scale in scales:
            evidence["perturbations"].append(
                one(kind, scale, f"{kind}_{suffix}", save=True)
            )
            checkpoint()


def main():
    """Parse local reference-data paths and benchmark controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network", type=int, default=0)
    parser.add_argument("--obsnum", type=int, default=145484)
    parser.add_argument("--target", type=int, default=145485)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--perturbations", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    reset_logger(level="WARNING", verbose=False)
    run(args)


if __name__ == "__main__":
    main()
