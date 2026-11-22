#!/usr/bin/env python
"""Replay fitting, stub preparation, solving and diagnostics in a fresh directory.

All science runs use the installed tolteca_kids command. This recipe generates
configuration and validates saved products; it contains no fitting or solving
implementation. Native verification uses the production library's test harness.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from prepare_stubs import prepare_stubs

from tolteca_datamodels.data_prod import DataProdIndex, resolve_index
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids.product import open_kids_product

HERE = Path(__file__).resolve().parent


def command(argv, log, *, workers):
    """Run a packaged command with explicit numerical worker settings and log."""
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        DASK_SCHEDULER="processes",
        DASK_NUM_WORKERS=str(workers),
    )
    with log.open("w") as output:
        subprocess.run(argv, env=env, check=True, stdout=output, stderr=output)  # noqa: S603 -- explicit argv, no shell


def solve_config(  # noqa: PLR0913 -- explicit source and destination identities
    root, namespace, *, tune_uid, stream_uid, model_index, evidence=None
):
    """Build a consumer configuration with exact raw and model indexes."""
    ns = root / "dataprod_toltec" / namespace
    obsnum, subobsnum, scannum = map(int, stream_uid.split("-")[-3:])
    return {
        "kids": {
            "input": {
                "select": (
                    f"obsnum == {obsnum} and subobsnum == {subobsnum} "
                    f"and scannum == {scannum}"
                ),
                "raise_on_errors": True,
            },
            "context": {
                "raw_indexes": [
                    str(ns / uid / "dp_raw_obs.yaml") for uid in (tune_uid, stream_uid)
                ],
                "models": [
                    {
                        "index": str(model_index),
                        "producer": "tolteca_kids.model_fit:ModelFit",
                    }
                ],
            },
            "timestream_solve": {"enabled": True, "backend": "python"},
            "timestream_reduce": {"psd": None},
            "timestream_diagnostics": {}
            if evidence is None
            else {"evidence": str(evidence)},
            "timestream_plot": {},
            "output": {
                "enabled": True,
                "registration": False,
                "path": str(root / "dataprod_toltec"),
                "namespace": namespace,
                "run_name": {"generator": "counter", "key": "solved"},
            },
        }
    }


def main():  # noqa: PLR0915 -- sequential reviewable replay
    """Create a reproducible comparison with explicit paths and product receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tune", type=Path)
    parser.add_argument("timestream", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["affine", "delay"],
        default=["affine", "delay"],
    )
    parser.add_argument(
        "--native", type=Path, help="Production-library tune_network test executable"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--affine_evidence", type=Path)
    parser.add_argument("--delay_evidence", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    cli = shutil.which("tolteca_kids")
    if cli is None:
        parser.error(
            "Install the pinned Python packages and their tolteca_kids command"
        )
    sources, uids = {}, []
    for path in (args.tune.resolve(), args.timestream.resolve()):
        with open_toltec_kids(path) as raw:
            uids.append(raw.toltec_kids.meta.uid_raw_obs)
        with path.open("rb") as stream:
            sources[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "raw_sources": sources,
        "families": {},
        "native": str(args.native) if args.native else None,
    }
    for family in args.families:
        config = yaml.safe_load((HERE / "fit.yaml").read_text())
        config["kids"]["model_fit"]["analysis"]["gain_family"] = family
        config["kids"]["output"].update(
            path=str(root / "dataprod_toltec"), namespace=family
        )
        path = root / f"fit-{family}.yaml"
        path.write_text(yaml.safe_dump(config))
        command(
            [cli, "--config_paths", str(path), "--", str(args.tune.resolve())],
            root / "logs" / f"fit-{family}.log",
            workers=args.workers,
        )
        ns = root / "dataprod_toltec" / family
        model = resolve_index(ns / uids[0] / "dp_raw_obs_proc")
        prepared = prepare_stubs(
            args.tune,
            args.timestream,
            model_index=model,
            output_path=root / "dataprod_toltec",
            namespace=family,
        )
        (root / f"prepared-{family}.json").write_text(json.dumps(prepared, indent=2))
        path = root / f"solve-{family}.yaml"
        evidence = getattr(args, f"{family}_evidence")
        if evidence is not None:
            destination = root / f"response-evidence-{family}.ecsv"
            shutil.copyfile(evidence, destination)
            evidence = destination
        path.write_text(
            yaml.safe_dump(
                solve_config(
                    root,
                    family,
                    tune_uid=uids[0],
                    stream_uid=uids[1],
                    model_index=model,
                    evidence=evidence,
                )
            )
        )
        command(
            [cli, "--config_paths", str(path), "--", str(args.timestream.resolve())],
            root / "logs" / f"solve-{family}.log",
            workers=args.workers,
        )
        result = resolve_index(ns / uids[1] / "dp_raw_obs_proc")
        index = DataProdIndex.read(result)
        with open_kids_product(index) as tree:
            ledger = (
                tree.toltec_kids.blocks[0]
                .tree["timestream_diagnostics"]
                .to_dataset(inherit=False)
            )
            record = {
                "model_index": str(model.relative_to(root)),
                "stub_index": str(Path(prepared["index"]).relative_to(root)),
                "solved_index": str(result.relative_to(root)),
                "fitted_components": ledger.sizes["component"],
                "numerical_complete": int(ledger.numerical_complete.sum()),
                "spectral_complete": int((ledger.spectral_status == "complete").sum()),
                "artifacts": len(index.product.data_items),
            }
        if args.native:
            command(
                [
                    sys.executable,
                    str(HERE / "network_coverage.py"),
                    str(model),
                    str(args.tune.resolve()),
                    str(args.timestream.resolve()),
                    str(args.native.resolve()),
                    str(root / f"native-{family}"),
                    "--namespace",
                    family,
                    "--dataprod_path",
                    str(root / "dataprod_toltec"),
                    "--prepared",
                    str(root / f"prepared-{family}.json"),
                ],
                root / "logs" / f"native-{family}.log",
                workers=args.workers,
            )
        manifest["families"][family] = record
        (root / "replay.json").write_text(json.dumps(manifest, indent=2))
        sys.stdout.write(json.dumps(record) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
