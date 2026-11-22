#!/usr/bin/env python
"""Replay a complete published network through ECSV, stubs and native solving.

Run after the packaged ``tolteca_kids`` command. This program reads saved fits;
it does not fit models, assign detector identities or alter calibration cuts.
"""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids.model_fit import ModelFit
from tolteca_kids.model_joint import joint_models, joint_solver
from tolteca_kids.product import open_sweep_product
from tolteca_kidsproc.model_flags import ModelFitFlag
from tolteca_kidsproc.recovery import SampleFlag
from tolteca_kidsproc.timestream import TimestreamSolver
from tolteca_reduce.stubs import _pack_models


def save(path, value):
    """Write a JSON evidence record, converting NumPy scalar values."""
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item()))


def flag_counts(flags, enum):
    """Count overlapping quality bits without treating them as exclusive."""
    return {f.name: int(np.count_nonzero(flags & int(f))) for f in enum}


def same_arrays(actual, expected, *, atol):
    """Report numerical agreement without concealing mismatched failures."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    finite = np.isfinite(actual) & np.isfinite(expected)
    return {
        "same_finite_mask": bool(
            np.array_equal(np.isfinite(actual), np.isfinite(expected))
        ),
        "max_abs_difference": float(np.max(abs(actual[finite] - expected[finite])))
        if finite.any()
        else None,
        "within_tolerance": bool(
            np.allclose(actual, expected, atol=atol, rtol=0, equal_nan=True)
        ),
    }


def require_parity(summary):
    """Make a transport/numerical mismatch fail the reproducible recipe."""

    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    key in {"flags_equal", "same_finite_mask", "within_tolerance"}
                    and not item
                ):
                    msg = f"Native/Python parity failed at {path}/{key}"
                    raise ValueError(msg)
                visit(item, f"{path}/{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, f"{path}/{i}")

    visit(summary, "solving")


def compare_transports(root, report):
    """Compare every native sample through ECSV and stub transport.

    This check is independent of the shorter Python comparison prefix. Identical
    calibration values must produce identical states and failure flags.
    """
    groups = report.meta["joint_models"]["groups"]
    result = {}
    scopes = [("independent", "", ("r", "x"))] + [
        (group["group_id"], f"group-{group['group_id']}", ("fr", "r"))
        for group in groups
    ]
    for name, suffix, fields in scopes:
        folders = [root / "native" / mode / suffix for mode in ("ecsv", "stub")]
        flags = [np.fromfile(path / "flags.bin", dtype="uint16") for path in folders]
        record = {"flags_equal": bool(np.array_equal(*flags))}
        for field in fields:
            values = [
                np.fromfile(path / f"{field}.bin", dtype="float64") for path in folders
            ]
            record[field] = same_arrays(*values, atol=0)
        result[name] = record
    return result


def compare_native(  # noqa: PLR0913 - explicit replay and saved-result controls
    request, report, native, root, *, run_native=True, reuse_python=False
):
    """Execute native routes and compare every saved channel and group."""
    config = root / "request.json"
    save(config, request)
    if run_native:
        with (root / "native.log").open("w") as log:
            subprocess.run(  # noqa: S603 - explicit local test executable, no shell
                [str(native), str(config)], check=True, stdout=log, stderr=log
            )
    shape = request["samples"], request["channels"]
    i, q = [np.fromfile(request[k], dtype="float64").reshape(shape) for k in "IQ"]
    r, x, flags = TimestreamSolver(report, report["chan"])(
        i, q, normalization=request["normalization"]
    )
    np.savez_compressed(root / "python-independent.npz", r=r, x=x, flags=flags)
    summary = {
        "native_transport": compare_transports(root, report),
        "native_samples": request["samples"],
        "independent": {},
        "joint": [],
    }
    for mode in ("ecsv", "stub"):
        folder = root / "native" / mode
        nr, nx = [
            np.fromfile(folder / f"{k}.bin", dtype="float64").reshape(shape)
            for k in ("r", "x")
        ]
        nf = np.fromfile(folder / "flags.bin", dtype="uint16").reshape(shape)
        summary["independent"][mode] = {
            "flags_equal": bool(np.array_equal(nf, flags)),
            "r": same_arrays(nr, r, atol=1e-12),
            "x": same_arrays(nx, x, atol=1e-12),
        }
    for group in report.meta["joint_models"]["groups"]:
        gid, channels = group["group_id"], group["channels"]
        sys.stdout.write(
            f"Solving group {gid}: {len(channels)} probes, "
            f"{len(group['resonator_ids'])} resonators\n"
        )
        sys.stdout.flush()
        checked = min(256, len(i))
        saved = root / f"python-group-{gid}.npz"
        if reuse_python and saved.exists():
            with np.load(saved) as arrays:
                fr, loss, gf = [arrays[key] for key in ("fr", "r", "flags")]
            assert fr.shape == (checked, len(group["resonator_ids"]))
        else:
            fr, loss, gf, *_ = joint_solver(group)(
                i[:checked, channels],
                q[:checked, channels],
                normalization=request["normalization"],
            )
            np.savez_compressed(saved, fr=fr, r=loss, flags=gf)
        item = {
            "group_id": gid,
            "components": len(group["resonator_ids"]),
            "channels": channels,
            "python_checked_samples": checked,
            "accepted_samples": int(
                np.count_nonzero((gf & (65535 ^ int(SampleFlag.condition_high))) == 0)
            ),
            "flags": flag_counts(gf, SampleFlag),
            "parity": {},
        }
        for mode in ("ecsv", "stub"):
            folder = root / "native" / mode / f"group-{gid}"
            nfr, nr = [
                np.fromfile(folder / f"{k}.bin", dtype="float64").reshape(
                    len(i), len(group["resonator_ids"])
                )[:checked]
                for k in ("fr", "r")
            ]
            all_flags = np.fromfile(folder / "flags.bin", dtype="uint16")
            nf = all_flags[:checked]
            item["native_accepted_samples"] = int(
                np.count_nonzero(
                    (all_flags & (65535 ^ int(SampleFlag.condition_high))) == 0
                )
            )
            item["native_sample_flags"] = flag_counts(all_flags, SampleFlag)
            item["parity"][mode] = {
                "flags_equal": bool(np.array_equal(nf, gf)),
                "fr": same_arrays(nfr, fr, atol=0.02),
                "r": same_arrays(nr, loss, atol=1e-12),
            }
        summary["joint"].append(item)
        save(root / "solving.json", summary)
    save(root / "solving.json", summary)
    require_parity(summary)
    return summary


def prepare_controls(view, quality, report, root):
    """Generate small controlled trajectories with the maintained forward model."""
    n = len(report)
    i, q = np.full((16, n), np.nan), np.full((16, n), np.nan)
    controls = []
    for group in joint_models(view, quality, include_single=True)["groups"]:
        solver = joint_solver(group)
        nr = len(group["resonator_ids"])
        states = np.empty((16, 2 * nr))
        for j in range(nr):
            states[:, 2 * j] = np.linspace(0, 0.003, 16)
            states[:, 2 * j + 1] = np.linspace(0, 0.005, 16)
        evaluated = solver.evaluate(
            solver.fr * (1 + 2 * solver.r * states[:, 1::2]),
            solver.r * (1 + states[:, ::2]),
        )
        channels = group["channels"]
        i[:, channels], q[:, channels] = evaluated.I.values, evaluated.Q.values
        controls.append(
            {
                "group_id": group["group_id"],
                "channels": channels,
                "resonator_ids": group["resonator_ids"],
            }
        )
    i.tofile(root / "I.bin")
    q.tofile(root / "Q.bin")
    save(
        root / "controls.json",
        {
            "trajectory": "dr/r_ref: 0..0.003; dfr/(2*r_ref*fr_ref): 0..0.005",
            "groups": controls,
            "unmodeled_channels": "NaN input retained",
        },
    )


def main():  # noqa: PLR0915 - sequential, inspectable replay recipe
    """Prepare each block, then its recorded consuming acquisition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("tune", type=Path)
    parser.add_argument("timestream", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--namespace", default="tune-comparison")
    parser.add_argument("--dataprod_path", type=Path)
    parser.add_argument(
        "--prepared", type=Path, help="Reuse the exact prepared-input receipt"
    )
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    index = DataProdIndex.read(args.index)
    reports = {
        int(item.meta["scope"].split("/")[-1]): fileloc(
            index.get_source_uri(item.source)
        ).path
        for item in index.product.data_items
        if item.meta.get("name") == "model"
    }
    census = []
    with open_sweep_product(index) as tree, open_toltec_kids(args.tune) as raw:
        for block_id, ecsv in sorted(reports.items()):
            folder = root / f"block{block_id}"
            folder.mkdir(exist_ok=True)
            view = ModelFit.view(tree, block=block_id)
            rows = view.fit.to_dataframe().reset_index(drop=True)
            components = rows.drop_duplicates(["group_id", "resonator_id"]).copy()
            sizes = components.groupby("group_id").size()
            components["multiplicity"] = components.group_id.map(sizes)
            components["identity_status"] = "model_local_only"
            components.to_csv(folder / "components.csv", index=False)
            rows.to_csv(folder / "associations.csv", index=False)
            view.chan_prop.to_dataframe().to_csv(folder / "channels.csv")
            group_fields = [
                name
                for name, value in view.joint.data_vars.items()
                if value.dims == view.joint.group_id.dims
            ]
            view.joint[group_fields].to_dataframe().to_csv(folder / "optimizer.csv")
            block = tree.toltec_kids.blocks[block_id]
            quality = block.tree[ModelFit.result_path()]["quality"].to_dataset()
            report = Table.read(ecsv, format="ascii.ecsv")
            stub = folder / "transport_test_stub.nc"
            _pack_models(raw, {block_id: report}).to_netcdf(stub)
            prepare_controls(view, quality, report, folder)
            item = {
                "block": block_id,
                "channels": len(report),
                "components": len(components),
                "groups": len(sizes),
                "singlet_components": int((components.multiplicity == 1).sum()),
                "multiplet_components": int((components.multiplicity > 1).sum()),
                "converged_components": int(
                    (components.fit_status == "optimizer_converged").sum()
                ),
                "a_quantiles": np.quantile(
                    components.a_nl, [0, 0.25, 0.5, 0.75, 0.95, 1]
                ).tolist(),
                "report_flags": flag_counts(np.asarray(report["flag"]), ModelFitFlag),
                "ecsv": str(ecsv),
                "stub": str(stub),
                "nonlinear_branch": report.meta["nonlinear_branch"],
            }
            census.append(item)
            save(root / "census.json", census)
            request = {
                "samples": 16,
                "channels": len(report),
                "I": str(folder / "I.bin"),
                "Q": str(folder / "Q.bin"),
                "ecsv": str(ecsv),
                "stub": str(stub),
                "block": block_id,
                "normalization": 1.0,
                "output": str(folder / "native"),
            }
            save(folder / "request.json", request)
            if not args.prepare_only:
                compare_native(request, report, args.native, folder)
    # Only the recorded final Tune block calibrates the subsequent timestream.
    report = Table.read(reports[max(reports)], format="ascii.ecsv")
    folder = root / "measured"
    folder.mkdir(exist_ok=True)
    with open_toltec_kids(args.timestream) as raw:
        i, q = [raw[f"Data.Toltec.{key}s"].values.astype("float64") for key in "IQ"]
        ticks = raw["Data.Toltec.Ts"]
        names = [
            part.strip().split()[0] for part in ticks.attrs["long_name"].split(",")
        ]
        packets = ticks.values[:, names.index("PacketCount")].astype(np.uint32)
        if np.any(np.diff(packets) != 1):
            msg = (
                "This array replay requires a gap-free acquisition; "
                "retain counter resets for chunked solving"
            )
            raise ValueError(msg)
        normalization = (
            float(raw["Header.Toltec.AccumLen"].item())
            / report.meta["Header.Toltec.AccumLen"]
        )
        probes = raw["Header.Toltec.ToneFreq"].values[0] + float(
            raw["Header.Toltec.LoCenterFreq"].item()
        )
        np.testing.assert_allclose(probes, report["f_in"], rtol=0, atol=1e-6)
    i.tofile(folder / "I.bin")
    q.tofile(folder / "Q.bin")
    # Use the maintained context preparation and product materialization recipe.
    prepare_stubs = runpy.run_path(str(Path(__file__).with_name("prepare_stubs.py")))[
        "prepare_stubs"
    ]

    prepared = (
        json.loads(args.prepared.read_text())
        if args.prepared
        else prepare_stubs(
            args.tune,
            args.timestream,
            model_index=args.index,
            output_path=args.dataprod_path or root.parent / "dataprod_toltec",
            namespace=args.namespace,
        )
    )
    save(folder / "prepared.json", prepared)
    stub = prepared["data_item"]["stubs"][0]["filepath"]
    request = {
        "samples": len(i),
        "channels": i.shape[1],
        "I": str(folder / "I.bin"),
        "Q": str(folder / "Q.bin"),
        "ecsv": str(reports[max(reports)]),
        "stub": stub,
        "block": 0,
        "normalization": normalization,
        "output": str(folder / "native"),
    }
    save(folder / "request.json", request)
    if not args.prepare_only:
        compare_native(request, report, args.native, folder)


if __name__ == "__main__":
    main()
