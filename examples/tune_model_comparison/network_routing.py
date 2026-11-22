#!/usr/bin/env python
"""Replay saved fits with fixed-probe routing in a separate comparison tree.

No fitting, sample selection, identity reconciliation or acceptance cuts change.
The source products and earlier solved arrays remain read-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr
from network_coverage import prepare_controls, save
from network_parity import resume
from network_quality import measure
from network_report import component_coverage
from network_validation import transport

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids.model_calibration import calibration_report
from tolteca_kids.model_fit_quality import ModelFitQualityConfig, model_fit_quality
from tolteca_kids.product import open_sweep_product
from tolteca_kidsproc.analysis.model_fit import KidsModelView
from tolteca_kidsproc.calibration import calibration_channels
from tolteca_reduce.stubs import _pack_models


def prepare(source, output):  # noqa: PLR0915 -- sequential saved-model replay
    """Write new reports/stubs and replay requests, preserving every saved fit."""
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "replay-source.json",
        {
            "source_root": str(source),
            "operation": "fixed-probe routing; no refit",
            "residual_limit": 0.05,
        },
    )
    for name in (
        "dataprod_toltec",
        "active-response",
        "response-qualification",
        "notation-before",
        "evidence.json",
        "plotly.min.js",
    ):
        (output / name).symlink_to(source / name)
    (output / "source").symlink_to(source, target_is_directory=True)
    changes = []
    for folder, realization in [
        ("coverage", "0000-models"),
        ("coverage-delay", "0001-models"),
    ]:
        original = source / folder
        base = output / folder
        base.mkdir()
        index = (
            source
            / "dataprod_toltec/network-145609/145609-0-1/dp_raw_obs_proc"
            / realization
            / "index.yaml"
        )
        with open_sweep_product(index) as tree:
            reports = {}
            for block_id in (0, 1):
                target = base / f"block{block_id}"
                target.mkdir()
                block = tree.toltec_kids.blocks[block_id]
                namespace = (
                    block.tree["model_fit/tolteca_kidsproc.analysis.model_fit"]
                    .copy(deep=False)
                    .load()
                )
                old = KidsModelView.from_namespace(namespace)
                previous_props = old.chan_prop.copy(deep=True)
                props = calibration_channels(
                    old.fit, previous_props.chan.values, previous_props.fp.values
                )
                namespace["chan_prop"] = xr.DataTree(dataset=props)
                view = KidsModelView.from_namespace(namespace)
                saved_quality = block.tree["model_fit/quality"].to_dataset().load()
                config = ModelFitQualityConfig.model_validate(
                    saved_quality.attrs["thresholds"]
                )
                quality = model_fit_quality(view, config)
                np.testing.assert_array_equal(
                    quality.group_flag, saved_quality.group_flag
                )
                report = calibration_report(view, block, quality=quality)
                ecsv = target / "kids_model.ecsv"
                report.write(ecsv, format="ascii.ecsv")
                reports[block_id] = report
                props.to_dataframe().to_csv(target / "channels.csv")
                for name in ("components.csv", "associations.csv", "optimizer.csv"):
                    shutil.copyfile(original / f"block{block_id}" / name, target / name)
                for c in props.chan.values:
                    before, after = previous_props.sel(chan=c), props.sel(chan=c)
                    if before.group_id.item() != after.group_id.item():
                        changes.append(
                            {
                                "family": folder,
                                "block": block_id,
                                "chan": int(c),
                                "before": before.group_id.item(),
                                "after": after.group_id.item(),
                                "probe_hz": after.fp.item(),
                                "reason": after.recovery_reason.item(),
                            }
                        )
                request = json.loads(
                    (original / f"block{block_id}/request.json").read_text()
                )
                with xr.open_dataset(request["stub"]) as raw:
                    _pack_models(raw, {block_id: report}).to_netcdf(
                        target / "models_stub.nc"
                    )
                prepare_controls(view, quality, report, target)
                request.update(
                    I=str(target / "I.bin"),
                    Q=str(target / "Q.bin"),
                    ecsv=str(ecsv),
                    stub=str(target / "models_stub.nc"),
                    output=str(target / "native"),
                )
                save(target / "request.json", request)
            measured = base / "measured"
            measured.mkdir()
            request = json.loads((original / "measured/request.json").read_text())
            prepared = json.loads((original / "measured/prepared.json").read_text())
            report = reports[1]
            with open_toltec_kids(prepared["data_item"]["filepath"]) as raw:
                _pack_models(raw, {0: report}).to_netcdf(measured / "models_stub.nc")
            request.update(
                ecsv=str(base / "block1/kids_model.ecsv"),
                stub=str(measured / "models_stub.nc"),
                output=str(measured / "native"),
            )
            save(measured / "request.json", request)
            # Raw identity only; the new stub is recorded in request.json.
            save(
                measured / "prepared.json",
                {"data_item": {"filepath": prepared["data_item"]["filepath"]}},
            )
    save(output / "routing-changes.json", changes)


def solve(output, native, *, workers=4):
    """Run both native readers, Python parity and component spectral accounting."""
    requests = sorted(output.glob("coverage*/**/request.json"))

    def native_run(request):
        with request.with_name("native.log").open("w") as log:
            subprocess.run(  # noqa: S603 -- Explicit local native test executable.
                [str(native), str(request)], check=True, stdout=log, stderr=log
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(native_run, requests))
    for request in requests:
        resume(request.parent, workers=workers)
        result = transport(request.parent)
        if not all(
            result[key] for key in ("same_files", "metadata_equal", "binary_equal")
        ):
            msg = f"ECSV/stub native transport differs: {request}"
            raise ValueError(msg)
        if request.parent.name == "measured":
            component_coverage(
                request.parent, request.parent.parent / "block1/components.csv"
            )
            measure(request.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--native",
        type=Path,
        help="Native tune_network executable; omit to prepare only",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve())
    if args.native:
        solve(args.output.resolve(), args.native.resolve(), workers=args.workers)
